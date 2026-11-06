# pyright: reportAttributeAccessIssue=false

import json
from collections import Counter, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import duckdb
import numpy
import pandas
from ase.db import connect as ase_connect
from ase.db.sqlite import SQLite3Database
from pint import Quantity, get_application_registry

import apitofsim.workflow.sql_files as sql_files
from apitofsim import ClusterData
from apitofsim.api import Histogram

from .base import SimulationMode
from .db_utils import (
    duckdb_connect_roview_cow,
    get_through_join_else,
    transaction_scope,
)

ureg = get_application_registry()
Q_ = ureg.Quantity


def guess_events_filename(filename):
    from apitofsim.recording import events_filename

    return events_filename(filename)


def guess_ase_db_filename(cluster_db_filename):
    ase_path = cluster_db_filename
    if ase_path.endswith(".duckdb"):
        ase_path = ase_path[: -len(".duckdb")]
    ase_path += ".ase.sqlite.db"
    return ase_path


@contextmanager
def connection_scope(database_type, filename, **kwargs):
    db = database_type(filename, **kwargs)
    try:
        yield db
    finally:
        db.close()


class SQLite3DatabaseWithUri(SQLite3Database):
    def _connect(self):
        assert self.filename is not None
        import sqlite3

        return sqlite3.connect(
            self.filename, timeout=20, uri=True, check_same_thread=False
        )


def connect_ase_sqlite_db_readonly(
    filename,
    create_indices=True,
    use_lock_file=True,
    append=True,
    serial=False,
    **db_kwargs,
):
    # TODO: These ClusterDatabase, etc. objects should probably be context managers too
    uri = f"file:{filename}?mode=ro"
    return SQLite3DatabaseWithUri(
        uri, create_indices, use_lock_file, serial=serial, **db_kwargs
    )


def connect_duckdb_cleanup(filename, *, readonly=False):
    if readonly:
        return duckdb_connect_roview_cow(filename, fallback="connect")
    else:
        return duckdb.connect(filename), None


class ClusterDatabase:
    TABLES = [sql_files.pathway]
    VIEWS = []

    def __init__(self, filename, *, readonly=False, ase_filename=None):
        self.closed = False
        if isinstance(filename, tuple):
            self.db, self.cleanup = filename
        else:
            self.db, self.cleanup = connect_duckdb_cleanup(filename, readonly=readonly)
        if ase_filename is not None:
            # TODO: These ClusterDatabase, etc. objects should probably be context managers too
            if readonly:
                self.ase_db = connect_ase_sqlite_db_readonly(ase_filename).__enter__()
            else:
                self.ase_db = ase_connect(ase_filename).__enter__()
        else:
            self.ase_db = None
        self._setup_db()

    def refresh_views(self):
        for view in self.VIEWS:
            self.db.execute(view)

    def create_temp_views(self):
        for view in self.VIEWS:
            self.db.execute(
                view.replace("create or replace view", "create or replace temp view")
            )

    def _setup_db(self):
        import os

        self.db.execute("SET threads=1")
        self.db.execute("SET preserve_insertion_order=false")
        if "DUCKDB_MEMORY_LIMIT" in os.environ:
            memory_limit = os.environ["DUCKDB_MEMORY_LIMIT"]
            self.db.execute(f"set memory_limit='{memory_limit}';")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "db"):
            self.db.close()
        if hasattr(self, "ase_db") and self.ase_db is not None:
            self.ase_db.__exit__(None, None, None)
        if hasattr(self, "cleanup") and self.cleanup is not None:
            self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()

    def create_tables(self, *, foreign_keys: bool = False):
        """Create a new schema, with foreign keys only when explicitly enabled.

        Primary keys, uniqueness and NOT NULL constraints are always retained.
        This does not change constraints in an existing database.
        """
        sql = "\n".join(self.TABLES)
        if not foreign_keys:
            sql = sql_files.strip_foreign_keys(sql)
        self.db.execute(sql)
        self.refresh_views()
        if self.ase_db is not None:
            self.db.execute(
                """
                alter table cluster add column ase_mol_id integer default null;
                """
            )

    def check_foreign_keys(self) -> dict[str, int]:
        """Return violated schema FK relationships and their orphan row counts.

        Checks the canonical schema even when FKs were omitted at creation.
        Nullable keys follow SQL MATCH SIMPLE: any NULL component exempts a row.
        All schema tables must be accessible, including attached cluster tables.
        This checks declared relationships only, not simulation semantics.
        """
        # Let DuckDB interpret the canonical DDL, including composite keys.
        with duckdb.connect(config={"threads": 1}) as schema:
            schema.execute("\n".join(self.TABLES))
            constraints = schema.execute(
                """select table_name, constraint_column_names,
                          referenced_table, referenced_column_names
                   from duckdb_constraints() where constraint_type = 'FOREIGN KEY'
                   order by table_name, constraint_index"""
            ).fetchall()

        def quote(name):
            return '"' + name.replace('"', '""') + '"'

        violations = {}
        with transaction_scope(self.db):
            for table, columns, parent, parent_columns in constraints:
                nonnull = " and ".join(f"c.{quote(c)} is not null" for c in columns)
                match = " and ".join(
                    f"c.{quote(c)} = p.{quote(p)}"
                    for c, p in zip(columns, parent_columns, strict=True)
                )
                row = self.db.execute(
                    f"select count(*) from {quote(table)} c where {nonnull} "
                    f"and not exists (select 1 from {quote(parent)} p where {match})"
                ).fetchone()
                assert row is not None
                count = row[0]
                if count:
                    key = f"{table}({', '.join(columns)}) -> {parent}({', '.join(parent_columns)})"
                    violations[key] = count
        return violations

    def clusters_query(
        self, parent=None, pathways=None, parents_only=False, children_only=False
    ):
        if parents_only and children_only:
            raise ValueError("Cannot set both parents_only and children_only to True")
        query = self.db.table("cluster")
        if (parent is None and pathways is None) and not (
            parents_only or children_only
        ):
            # Shortcut for efficiency
            return query
        if parents_only:
            relevant_fragment = "cluster_id"
        elif children_only:
            relevant_fragment = "product1_id, product2_id"
        else:
            relevant_fragment = "cluster_id, product1_id, product2_id"
        pathways_query = self.pathways_query(parent, pathways)
        # relevant_cluster_ids = self.db.table("pathway").select(duckdb.SQLExpression(f"unnest([{relevant_fragment}])").alias("relevant_cluster_id"))
        # if parent is not None:
        # relevant_cluster_ids = relevant_cluster_ids.filter(duckdb.ColumnExpression('cluster_id ') == duckdb.ConstantExpression(parent))
        # relevant_cluster_ids = relevant_cluster_ids.distinct().fetchdf()
        relevant_cluster_ids = (
            pathways_query.select(
                duckdb.SQLExpression(f"unnest([{relevant_fragment}])").alias(
                    "relevant_cluster_id"
                )
            )
            .distinct()
            .fetchdf()
        )
        return query.join(
            self.db.from_df(relevant_cluster_ids).set_alias("relevant"),
            condition="relevant.relevant_cluster_id = cluster.id",
        )

    def clusters_df(self, *args, **kwargs):
        return self.clusters_query(*args, **kwargs).fetchdf().replace({pandas.NA: None})

    def iter_clusters_dicts(self, *args, **kwargs):
        for cluster in self.clusters_df(*args, **kwargs).itertuples():
            yield cluster._asdict()

    @staticmethod
    def _cluster_obj_from_tuple(cluster):
        return ClusterData(
            Q_(cluster.atomic_mass, "amu"),
            Q_(cluster.electronic_energy, "hartree"),
            cluster.rotational_temperatures,
            cluster.vibrational_temperatures,
            cluster.charge,
        )

    def iter_clusters_objects(self, *args, **kwargs):
        for cluster in self.clusters_df(*args, **kwargs).itertuples():
            obj = self._cluster_obj_from_tuple(cluster)
            yield obj

    def clusters_dicts_indexed(self, *args, **kwargs):
        ret = {}
        for d in self.iter_clusters_dicts(*args, **kwargs):
            ret[d["id"]] = d
        return ret

    def clusters_objects_indexed(self, *args, include_name_lookup=False, **kwargs):
        name_lookup = {}
        ret = {}
        for cluster in self.clusters_df(*args, **kwargs).itertuples():
            ret[cluster.id] = self._cluster_obj_from_tuple(cluster)
            if include_name_lookup:
                name_lookup[cluster.id] = cluster.common_name
        if include_name_lookup:
            return ret, name_lookup
        else:
            return ret

    def pathways_query(self, parent=None, pathways=None, sort=False):
        if parent is not None and pathways is not None:
            raise ValueError("Cannot specify both parent and pathways")
        pathway_rel = self.db.table("pathway")
        if pathways is not None:
            import pyarrow as pa

            wanted_tbl = pa.table([pathways], names=["pathway_id"])
            pathway_rel = pathway_rel.join(
                self.db.from_arrow(wanted_tbl).set_alias("wanted"),
                condition="wanted.pathway_id = pathway.id",
            )
        if parent is not None:
            pathway_rel = pathway_rel.filter(
                duckdb.ColumnExpression("cluster_id")
                == duckdb.ConstantExpression(parent)
            )
        if sort:
            pathway_rel = pathway_rel.sort(
                "cluster_id",  # pyright: ignore[reportArgumentType]
                "product1_id",  # pyright: ignore[reportArgumentType]
                "product2_id",  # pyright: ignore[reportArgumentType]
            )
        return pathway_rel

    def pathways_ids(self, parent=None, **kwargs):
        query = self.pathways_query(parent, **kwargs)
        for pathway in query.fetchdf().itertuples():
            yield (
                pathway.id,
                pathway.cluster_id,
                pathway.product1_id,
                pathway.product2_id,
            )

    def pathways_objs(self, *args, indexed=None, **kwargs):
        if indexed is None:
            indexed = self.clusters_objects_indexed()
        for pathway_id, cluster_id, product1_id, product2_id in self.pathways_ids(
            *args, **kwargs
        ):
            yield (
                pathway_id,
                indexed[cluster_id],
                indexed[product1_id],
                indexed[product2_id],
            )

    def insert_cluster(
        self,
        name,
        atomic_mass,
        charge,
        electronic_energy,
        rotational_temperatures,
        vibrational_temperatures,
        import_info=None,
        *,
        ase_id=None,
        allow_duplicates=False,
    ):
        if ase_id is not None and self.ase_db is None:
            raise ValueError("ASE database not initialized, cannot insert ASE molecule")
        value_names = [
            "atomic_mass",
            "charge",
            "electronic_energy",
            "rotational_temperatures",
            "vibrational_temperatures",
        ]
        values = [
            atomic_mass,
            charge,
            electronic_energy,
            rotational_temperatures,
            vibrational_temperatures,
        ]
        existing = self.db.execute(
            """
            select
                id,
                atomic_mass,
                charge,
                electronic_energy,
                rotational_temperatures,
                vibrational_temperatures
            from cluster
            where common_name = ?
            """,
            (name,),
        ).fetchone()
        if existing is not None:
            for value_name, existing_value, new_value in zip(
                value_names, existing[1:], values
            ):
                if not numpy.array_equal(existing_value, new_value):
                    if not allow_duplicates:
                        raise ValueError(
                            f"Cluster with name '{name}' already exists with different {value_name}: existing={existing_value}, new={new_value}"
                        )
                    if "__" in name:
                        barename, num = name.rsplit("__", 1)
                        try:
                            num = int(num)
                        except ValueError:
                            barename = name
                            num = 0
                    else:
                        barename = name
                        num = 0
                    return self.insert_cluster(
                        f"{barename}__{num + 1}",
                        *values,
                        import_info=import_info,
                        allow_duplicates=True,
                    )
            return False, existing[0]

        id = self.db.execute(
            f"insert into cluster values (default, ?, ?, ?, ?, ?, ?, ?, {'?' if self.ase_db is not None else ''}) returning id",
            (
                name,
                atomic_mass,
                charge,
                electronic_energy,
                rotational_temperatures,
                vibrational_temperatures,
                import_info,
                *((ase_id,) if self.ase_db is not None else ()),
            ),
        ).fetchone()
        assert id is not None
        return True, id[0]

    def insert_pathway(self, parent_id, product1_id, product2_id):
        self.db.execute(
            "insert into pathway values (default, ?, ?, ?)",
            (parent_id, product1_id, product2_id),
        )

    def get_all_lookups(self, parent=None, pathways=None):
        if isinstance(parent, str):
            parent = self.db.execute(
                """
                select id
                from cluster
                where common_name = ?
                """,
                (parent,),
            ).fetchone()
            if parent is None:
                raise ValueError(f"No cluster found with name {parent}")
            parent = parent[0]
        if pathways is not None:
            pathways = list(pathways)
            if (
                len(pathways) > 0
                and isinstance(pathways[0], tuple)
                and isinstance(pathways[0][0], str)
            ):
                import pyarrow as pa

                wanted_tbl = pa.table(
                    list(zip(*pathways)), names=["pathway", "product1", "product2"]
                )
                pathway_common_names = self.db.sql(
                    """
                    select
                        p.id as pathway_id,
                        c.common_name as cluster_common_name,
                        p1.common_name as product1_common_name,
                        p2.common_name as product2_common_name,

                    from pathway p
                    inner join cluster c on c.id = p.cluster_id
                    inner join cluster p1 on p1.id = p.product1_id
                    inner join cluster p2 on p2.id = p.product2_id;
                    """
                )
                pathways = (
                    pathway_common_names.join(
                        self.db.from_arrow(wanted_tbl).set_alias("wanted"),
                        condition=(
                            "wanted.pathway = cluster_common_name "
                            "and ((wanted.product1 = product1_common_name and wanted.product2 = product2_common_name) "
                            "or (wanted.product1 = product2_common_name and wanted.product2 = product1_common_name))"
                        ),
                    )
                    .select("pathway_id")
                    .to_arrow_table()["pathway_id"]
                )
        cluster_indexed, name_lookup = self.clusters_objects_indexed(
            include_name_lookup=True, parent=parent, pathways=pathways
        )
        pathway_lookup = {}
        for pathway_id, cluster_id, product1_id, product2_id in self.pathways_ids(
            parent=parent, pathways=pathways, sort=True
        ):
            pathway_lookup[pathway_id] = (cluster_id, product1_id, product2_id)

        return cluster_indexed, name_lookup, pathway_lookup

    def insert_ase(self, cluster_id, ase_mol):
        if self.ase_db is None:
            raise ValueError("ASE database not initialized")
        ase_mol_id = self.ase_db.write(ase_mol)
        self.db.execute(
            "update cluster set ase_mol_id = ? where id = ?", (ase_mol_id, cluster_id)
        )

    def is_experiment_db(self) -> bool:
        return False


class SuperClusterDatabase(ClusterDatabase):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
        sql_files.vib_energy_index,
        sql_files.rf_basis,
    ]
    VIEWS = [
        sql_files.pathway_report,
    ]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

    def ensure_vib_index_cache(self):
        # Existing read-only databases can serve warmed indexes without any DDL.
        try:
            self.db.table("vib_energy_index")
        except duckdb.CatalogException:
            self.db.execute(sql_files.vib_energy_index)

    def get_histogram_params(self, histogram_id):
        row = (
            self.db.table("histogram_params")
            .filter(
                duckdb.ColumnExpression("id") == duckdb.ConstantExpression(histogram_id)
            )
            .select("bin_width", "max")
            .fetchone()
        )
        if row is None:
            return None
        bin_width, x_max = row
        return Q_(bin_width, "K"), Q_(x_max, "K")

    def available_dos_histograms_query(self):
        """Return distinct DOS configuration IDs and bin widths (kelvin).

        Availability includes cluster and products DOS. No pairing with rate
        configurations is implied by the rows or their order.
        """
        return self.db.sql("""
            select id as histogram_params_id, bin_width
            from histogram_params
            where id in (
                select histogram_params_id from cluster_dos
                union
                select histogram_params_id from products_dos
            )
        """)

    def available_rate_histograms_query(self):
        """Return distinct k_rate configuration IDs and bin widths (kelvin).

        These configurations are independent of the available DOS set.
        """
        return self.db.sql("""
            select id as histogram_params_id, bin_width
            from histogram_params
            where id in (select histogram_params_id from k_rate)
        """)

    def get_cluster_dos_through_join_else(
        self, cluster_indexed, histogram_id, cluster_dos_dict
    ):
        wanted_cluster_ids = numpy.array(list(cluster_indexed.keys()))
        for miss_info in get_through_join_else(
            self.db,
            self.db.table("cluster_dos"),
            "data",
            cluster_dos_dict,
            cluster_id=wanted_cluster_ids,
            histogram_params_id=histogram_id,
        ):
            cluster_id = miss_info["cluster_id"]
            cluster = cluster_indexed[cluster_id]
            if cluster.is_atom_like_product():
                cluster_dos_dict[cluster_id] = None
            else:
                yield cluster_id, cluster

    def get_product_dos_through_join_else(
        self, cluster_indexed, histogram_id, cluster_dos_dict, pathway_lookup
    ):
        from apitofsim.api import ProductsCluster

        wanted_p1 = []
        wanted_p2 = []
        for _, product1_id, product2_id in pathway_lookup.values():
            product1_id, product2_id = sorted((product1_id, product2_id))
            wanted_p1.append(product1_id)
            wanted_p2.append(product2_id)

        for miss_info in get_through_join_else(
            self.db,
            self.db.table("products_dos"),
            "data",
            cluster_dos_dict,
            cluster1_id=wanted_p1,
            cluster2_id=wanted_p2,
            histogram_params_id=histogram_id,
        ):
            p1 = miss_info["cluster1_id"]
            p2 = miss_info["cluster2_id"]
            yield ProductsCluster(cluster_indexed[p1], cluster_indexed[p2])

    def get_k_rate_through_join_else(self, histogram_id, k_total_dict, pathway_lookup):
        for miss_info in get_through_join_else(
            self.db,
            self.db.table("k_rate").filter(
                duckdb.ColumnExpression("histogram_params_id")
                == duckdb.ConstantExpression(histogram_id)
            ),
            "data",
            k_total_dict,
            pathway_id=pathway_lookup.keys(),
        ):
            pathway_id = miss_info["pathway_id"]
            cluster_id, product1_id, product2_id = pathway_lookup[pathway_id]
            yield pathway_id, cluster_id, product1_id, product2_id


ConfigRow = namedtuple("ConfigRow", ["id", "name", "config"])


class RateCurveProvenanceError(ValueError):
    """The selected result's rate data cannot be identified unambiguously."""


@dataclass(frozen=True)
class PathwayRateCurve:
    """Cached k(delta_E), where delta_E = internal energy - bonding energy.

    histogram.x and bonding_energy are per-particle joules; rates has units 1/s.
    bin_width and energy_max describe the rate mesh in kelvin.
    A missing histogram is explicit and never triggers computation.
    """

    pathway_id: int
    parent_id: int
    product1_id: int
    product2_id: int
    participating: bool
    dos_histogram_id: int
    rate_histogram_id: int
    bin_width: Quantity[float]
    energy_max: Quantity[float]
    bonding_energy: Quantity[float]
    histogram: Histogram | None

    @property
    def rates(self):
        return None if self.histogram is None else Q_(self.histogram.y, "1/s")

    def rate_at(self, delta_energy, *, strict=True):
        """Evaluate k with simulation bin-centre interpolation and thresholds.

        Excess energy at or below zero gives a zero rate.
        Above the upper bin edge, strict raises ApiTofRateConstantOverflow;
        non-strict clamps to the last value, as next_fragmentation_time does.
        Missing energy remains missing. NaNs propagate.
        """
        from apitofsim.api import (
            ApiTofRateConstantOverflow,
            energy_in_joules,
        )

        if self.histogram is None:
            raise ValueError(f"Missing rate curve for pathway {self.pathway_id}")
        energy = energy_in_joules(delta_energy)
        if energy is None:
            return None
        x = self.histogram.x.magnitude
        values = numpy.asarray(energy.magnitude)
        maximum = (x[1] - x[0]) * len(x)
        if strict and numpy.any(values > maximum):
            from apitofsim import consts

            error = ApiTofRateConstantOverflow(
                "Excess energy exceeds the rate histogram"
            )
            error.max = maximum / consts.boltzmann
            error.current = float(numpy.nanmax(values)) / consts.boltzmann
            raise error
        rate = numpy.interp(values, x, self.histogram.y)
        return Q_(numpy.where(values <= 0, 0, rate), "1/s")


class _EventQueries:
    @staticmethod
    def _scope_query(query, run_id, parent, result_id):
        for column, value in (
            ("experiment_run_id", run_id),
            ("cluster_id", parent),
            ("experiment_result_id", result_id),
        ):
            if value is not None:
                query = query.filter(
                    duckdb.ColumnExpression(column) == duckdb.ConstantExpression(value)
                )
        return query

    def results_query(self, run_id=None, parent=None, result_id=None):
        """Return results, scoped by run, root cluster ID and/or result ID.

        Tree executions are included from startup, even without successful completion.

        Single-pathway parents are resolved through their pathway. Multi-pathway
        results appear once, including results with no pathway_fragmentation rows.
        All supplied scope filters are intersected; omitted filters select all rows.
        """
        query = self.db.table("event_execution").select(
            "result_id as experiment_result_id, run_id as experiment_run_id, cluster_id, pathway_id, pathway_id is not null as is_single_pathway, status"
        )
        return self._scope_query(query, run_id, parent, result_id)

    def realizations_query(self, run_id=None, parent=None, result_id=None):
        """Return recorded realizations with result, run and parent IDs.

        Includes observed histories from complete, cancelled and failed executions.
        All supplied filters intersect; event identity is scoped to its execution.
        """
        self._ensure_event_reader()
        query = (
            self.db.table("realization")
            .set_alias("r")
            .join(
                self.results_query().set_alias("results"),
                "results.experiment_result_id = r.experiment_result_id",
                how="left",
            )
            .select("""
                r.id as realization_id, r.experiment_result_id,
                results.experiment_run_id, results.cluster_id,
                results.pathway_id, results.is_single_pathway
            """)
        )
        return self._scope_query(query, run_id, parent, result_id)

    def events_query(self, run_id=None, parent=None, result_id=None, *, sort=False):
        """Return raw event state, cohort IDs and optional subtype fields.

        Subtype fields are projected from the fixed IPC schema. With sort=True,
        order by numeric result, realization and event indices; timestamps may tie.
        The default leaves ordering to the caller.
        """
        self._ensure_event_reader()
        query = (
            self.db.table("event_details")
            .set_alias("e")
            .join(
                self.realizations_query(run_id, parent, result_id).set_alias("cohort"),
                "cohort.realization_id = e.realization_id",
            )
            .select("""
                e.id as event_id, e.* exclude (id),
                cohort.experiment_result_id, cohort.experiment_run_id,
                cohort.cluster_id, cohort.is_single_pathway
            """)
        )
        if sort:
            query = query.order("result_id, realization_index, event_index")
        return query

    def tree_events_query(self, run_id=None, parent=None, result_id=None):
        """Ordered events with persisted tree identity, without inferring history.

        cluster_id remains the execution root; current_cluster_id identifies the
        event's before-state species, including at fragmentation. Destinations use
        native -1 for an intentional stop and NULL for unavailable information.
        provenance reports missing/inconsistent data; it never treats a local
        node index as a database cluster ID. Includes partial tree executions.
        """
        events = self.events_query(run_id, parent, result_id)
        query = (
            events.set_alias("e")
            .join(
                self.db.table("cluster_tree_experiment_result").set_alias("tree"),
                "tree.id = e.experiment_result_id",
            )
            .join(
                self.db.table("cluster_tree_node").set_alias("node"),
                "node.experiment_result_id = e.experiment_result_id and node.node_index = e.particle_index",
                how="left",
            )
            .join(
                self.db.table("cluster_tree_transition").set_alias("transition"),
                "transition.experiment_result_id = e.experiment_result_id and transition.transition_index = e.transition_index",
                how="left",
            )
            .join(
                self.db.table("cluster_tree_node").set_alias("destination"),
                "destination.experiment_result_id = e.experiment_result_id and destination.node_index = transition.destination_node_index",
                how="left",
            )
            .select("""
                e.*, node.cluster_id as current_cluster_id,
                destination.cluster_id as next_cluster_id,
                transition.destination_node_index,
                case
                    when node.node_index is null then 'missing_node'
                    when e.event_type = 'fragmentation' then
                        case
                            when e.fragmentation_event_id is null then 'missing_subtype'
                            when e.pathway_id is null then 'missing_pathway'
                            when transition.transition_index is null then 'missing_transition'
                            when transition.source_node_index != e.particle_index
                              or transition.pathway_id != e.pathway_id then 'inconsistent_transition'
                            when e.next_particle_index is null then 'missing_destination'
                            when e.next_particle_index != coalesce(transition.destination_node_index, -1)
                                then 'inconsistent_destination'
                            else 'persisted'
                        end
                    else 'persisted'
                end as provenance
            """)
        )
        # Validate continuity in event order, including retained partial histories.
        return query.select("""
            *, case
                when event_type = 'init' then particle_index = 0
                else particle_index = lag(
                    case when event_type = 'fragmentation' then next_particle_index
                         else particle_index end
                ) over (partition by realization_id order by event_index)
            end as continuity_valid
        """).order("result_id, realization_index, event_index")

    def fragmentation_histories_query(self, run_id=None, parent=None, result_id=None):
        """One ordered row per observed split, retaining realization/node identity.

        Enabled branches with no observations live in cluster_tree_transition.
        This reader requires the new schema and performs no legacy reconstruction.
        """
        return (
            self.tree_events_query(run_id, parent, result_id)
            .filter("event_type = 'fragmentation'")
            .order("result_id, realization_index, event_index")
        )

    def recorded_output_query(self, run_id=None, parent=None, result_id=None):
        """Return one row of diagnostics for recorded output in the given scope.

        Counts describe database records, not simulation counters. Terminal events
        mean escape or fragmentation event-info records, regardless of subtype
        presence; a realization with multiple such events is counted once.
        Empty scopes return zero counts. With no filters, scope is this database.
        """
        realizations = self.realizations_query(run_id, parent, result_id).aggregate(
            "count(*) as realization_count"
        )
        events = self.events_query(run_id, parent, result_id).aggregate("""
            count(*) as event_count,
            count(*) filter (where event_type = 'escape') as escape_count,
            count(*) filter (where event_type = 'collision') as collision_event_count,
            count(collision_event_id) as collision_subtype_count,
            count(*) filter (where event_type = 'fragmentation') as fragmentation_event_count,
            count(fragmentation_event_id) as fragmentation_subtype_count,
            count(distinct realization_id) filter (
                where event_type in ('escape', 'fragmentation')
            ) as terminal_realization_count
        """)
        return realizations.join(events, "true")


class ExperimentDatabase(SuperClusterDatabase, _EventQueries):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
        sql_files.vib_energy_index,
        sql_files.rf_basis,
        sql_files.experiment,
        sql_files.event_recording,
    ]
    VIEWS = [
        sql_files.pathway_report,
        sql_files.experiment_report,
    ]

    def __init__(
        self, filename, *, record_events=False, events_filename=None, **kwargs
    ):
        from pathlib import Path

        from apitofsim.recording import events_filename as sidecar_name

        self.record_events = record_events
        self.readonly = kwargs.get("readonly", False)
        self._recording = None
        self._event_read_lock = None
        self._event_reader_ready = False
        self.events_path = (
            Path(events_filename).resolve() if events_filename is not None else None
        )
        super().__init__(filename, **kwargs)
        if self.events_path is None:
            source = (
                filename
                if isinstance(filename, (str, Path)) and str(filename) != ":memory:"
                else next(
                    (
                        row[2]
                        for row in self.db.execute("pragma database_list").fetchall()
                        if row[2]
                    ),
                    None,
                )
            )
            if source is not None:
                self.events_path = sidecar_name(source).resolve()
        physical = {
            row[0]
            for row in self.db.execute(
                "select table_name from duckdb_tables() where not internal"
            ).fetchall()
        }
        if "event_info" in physical or "realization" in physical:
            self.close()
            raise ValueError(
                "Legacy relational-event database: explicit migration is required before opening"
            )
        if "experiment_config" in physical and "event_recording_state" not in physical:
            self.close()
            raise ValueError(
                "Unsupported experiment schema: create a new database or migrate explicitly"
            )
        if "event_recording_state" in physical:
            try:
                self._ensure_event_reader()
            except BaseException:
                self.close()
                raise

    def create_tables(self, *, foreign_keys=False):
        super().create_tables(foreign_keys=foreign_keys)
        self._ensure_event_reader()

    def _release_event_reader(self):
        if self._event_reader_ready and not self.closed:
            self.db.unregister("_event_archive")
        if self._event_read_lock is not None:
            self._event_read_lock.release()
            self._event_read_lock = None
        self._event_reader_ready = False

    def _ensure_event_reader(self):
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.dataset as ds

        from apitofsim.recording import (
            EVENT_SCHEMA,
            RecordingTransaction,
            archive_metadata,
            recording_lock,
        )

        if self._recording is not None:
            raise RuntimeError(
                "Event queries are unavailable during a recording transaction; use live event delivery"
            )
        if self._event_reader_ready:
            return
        state = self.db.execute(
            "select schema_version,archive_id,generation,file_size from event_recording_state"
        ).fetchall()[0]
        if state[0] != 1:
            raise ValueError(
                "Unsupported event recording schema; explicit migration is required"
            )
        path = self.events_path
        if path is not None and Path(str(path) + ".journal").exists():
            with recording_lock(path):
                RecordingTransaction.recover(
                    path, generation=state[2], expected_archive_id=state[1]
                )
        if path is not None:
            self._event_read_lock = recording_lock(path, shared=True)
            self._event_read_lock.acquire()
        if state[3]:
            if path is None or not path.exists():
                self._release_event_reader()
                raise FileNotFoundError(
                    "Recorded events require the matching .events.arrow sidecar (or explicit events_filename)"
                )
            if path.stat().st_size != state[3] or archive_metadata(path)[0] != state[1]:
                self._release_event_reader()
                raise ValueError(
                    "Event sidecar identity/size does not match this database"
                )
            self.db.register("_event_archive", ds.dataset(str(path), format="ipc"))
        else:
            self.db.register(
                "_event_archive", pa.Table.from_batches([], schema=EVENT_SCHEMA)
            )
        self.db.execute(sql_files.event_views)
        self._event_reader_ready = True

    def close(self):
        if getattr(self, "_recording", None) is not None:
            raise RuntimeError(
                "Close the recording transaction before closing its database"
            )
        try:
            super().close()
        finally:
            self._release_event_reader()

    @contextmanager
    def recording_transaction(self):
        from apitofsim.recording import IpcRecording, RecordingTransaction

        if self.readonly:
            raise PermissionError("Cannot record through a read-only database")
        if self._recording is not None:
            yield self._recording
            return
        if self.events_path is None:
            raise ValueError(
                "Recording requires a file-backed database or an explicit events_filename"
            )
        self._release_event_reader()
        transaction = RecordingTransaction(
            IpcRecording(self.events_path, "append"), database=self
        )
        previous_recording = self.record_events
        primary_error = None
        try:
            with transaction:
                self._recording = transaction
                self.record_events = True
                try:
                    yield transaction
                except BaseException as error:
                    # Preserve native partial results only if finishing the
                    # archive and its metadata remains possible.
                    self.db.execute(
                        "update event_execution set status='failed' where status='running'"
                    )
                    try:
                        transaction.finish()
                    except BaseException as cleanup:
                        error.add_note(f"Event recording rollback: {cleanup}")
                        transaction.rollback()
                    raise
        except BaseException as error:
            primary_error = error
            raise
        finally:
            self.record_events = previous_recording
            self._recording = None
            self._event_reader_ready = False
            try:
                self._ensure_event_reader()
            except BaseException as cleanup:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Reopening event reader failed: {cleanup}")

    def begin_execution(
        self,
        run_id,
        cluster_id,
        pathway_id,
        pathway_ids,
        *,
        result_id=None,
        recorded=False,
    ):
        if result_id is None:
            result_id = self.db.execute(
                "select nextval('experiment_result_sequence')"
            ).fetchall()[0][0]
        if cluster_id is None and pathway_id is not None:
            cluster_id = self.db.execute(
                "select cluster_id from pathway where id=?", [pathway_id]
            ).fetchall()[0][0]
        self.db.execute(
            "insert into event_execution (result_id,run_id,cluster_id,pathway_id,pathway_ids,recorded,status) values(?,?,?,?,?,?,'running')",
            [result_id, run_id, cluster_id, pathway_id, list(pathway_ids), recorded],
        )
        return result_id

    def finish_execution(self, result_id, status, *, recording=None, failure_id=None):
        self.db.execute(
            "update event_execution set status=?,failure_id=?,event_count=?,batch_start=?,batch_count=? where result_id=?",
            [
                status,
                failure_id,
                recording.event_count if recording else 0,
                recording.batch_start if recording else 0,
                recording.batch_count if recording else 0,
                result_id,
            ],
        )

    def insert_run(
        self,
        config_id=None,
        simulation_mode=SimulationMode.SINGLE_CLUSTER,
        *,
        dos_histogram_id=None,
        rate_histogram_id=None,
    ):
        import orjson

        id = self.db.execute(
            "insert into experiment_run (id,experiment_config_id,simulation_mode,run_config,start_time) values (default, ?, ?, ?, current_timestamp) returning id",
            (
                config_id,
                simulation_mode.name,
                orjson.dumps(
                    {
                        "dos_histogram_id": dos_histogram_id,
                        "rate_histogram_id": rate_histogram_id,
                    }
                ).decode(),
            ),
        ).fetchone()
        assert id is not None
        return id[0]

    def record_run_preparation(
        self, run_id, config, dos_histogram_id, rate_histogram_id
    ):
        """Record the exact cache IDs and effective config before simulation.

        Custom runners using DerivedDataPreparer should call this after
        run_preliminaries, using its dos_histogram_id and rate_histogram_id.
        """
        from apitofsim.config import dump_to_raw

        metadata = dump_to_raw(
            {
                "config": config,
                "dos_histogram_id": dos_histogram_id,
                "rate_histogram_id": rate_histogram_id,
            }
        ).decode()
        self.db.execute(
            "update experiment_run set run_config = json_merge_patch(coalesce(run_config, '{}'), ?::json) where id = ?",
            (metadata, run_id),
        )

    def pathway_rate_curves(
        self, run_id, parent, result_id=None, *, all_pathways=False
    ):
        """Read one result's cached curves without preparing data or changing views.

        parent is a database cluster ID (the current parent for serial events).
        Omit result_id only if the run/parent identifies exactly one result.
        By default, return enabled pathways, including those with zero events.
        all_pathways also includes other outgoing pathways with participating=False.
        Missing cached curves have histogram=None; no data is computed on read.
        Bonding energies honour the selected single-pathway fragmentation_energy
        override, otherwise using the simulation's electronic-energy calculation.
        Explicit histogram IDs in run_config are required: legacy runs without
        provenance raise RateCurveProvenanceError rather than guessing a cache.
        """
        import orjson

        from apitofsim.api import (
            FragmentationPathway,
            Histogram,
            consts,
            energy_in_joules,
            scaled_rate_const,
        )
        from apitofsim.config import import_raw_config

        run = self.db.execute(
            """
            select r.run_config, c.config from experiment_run r
            left join experiment_config c on c.id = r.experiment_config_id
            where r.id = ?
        """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise RateCurveProvenanceError(f"No run {run_id}")
        metadata = orjson.loads(run[0] or "{}")
        # Older insert_run versions passed JSON bytes, stored as a JSON string.
        if isinstance(metadata, str):
            metadata = orjson.loads(metadata)
        config = import_raw_config(
            metadata["config"] if "config" in metadata else orjson.loads(run[1] or "{}")
        )
        dos_id = metadata.get("dos_histogram_id")
        rate_id = metadata.get("rate_histogram_id")
        if dos_id is None or rate_id is None:
            raise RateCurveProvenanceError("Missing DOS/rate histogram provenance")
        params = self.get_histogram_params(rate_id)
        if params is None or self.get_histogram_params(dos_id) is None:
            raise RateCurveProvenanceError("Recorded histogram configuration is absent")
        results = self.db.execute(
            """
            select id, pathway_id, null as cluster_id
            from single_pathway_experiment_result where experiment_run_id = ?
            union all
            select id, null, cluster_id
            from multi_pathway_experiment_result where experiment_run_id = ?
            union all
            select id, null, cluster_id
            from cluster_tree_experiment_result where experiment_run_id = ? and total_us is not null
        """,
            (run_id, run_id, run_id),
        ).fetchall()
        outgoing = sorted(self.pathways_ids(parent=parent))
        outgoing_ids = {p[0] for p in outgoing}
        if result_id is not None:
            results = [r for r in results if r[0] == result_id]
        else:
            results = [r for r in results if r[1] in outgoing_ids or r[2] == parent]
        if len(results) != 1:
            raise RateCurveProvenanceError("Select exactly one successful result")
        result_id, single_pathway, root = results[0]
        if single_pathway is not None:
            if single_pathway not in outgoing_ids:
                raise RateCurveProvenanceError("Result does not belong to this parent")
            enabled = {single_pathway}
        else:
            # record_result stores every enabled pathway, including zero counts.
            enabled = {
                r[0]
                for r in self.db.execute(
                    "select pathway_id from pathway_fragmentation where experiment_result_id = ?",
                    (result_id,),
                ).fetchall()
            }
            if not enabled:
                raise RateCurveProvenanceError("Missing enabled-pathway provenance")
            if root != parent and not enabled.intersection(outgoing_ids):
                raise RateCurveProvenanceError(
                    "Parent is not identified in this result"
                )
        clusters = self.clusters_objects_indexed(parent=parent)
        curves = []
        for pathway_id, parent_id, p1, p2 in outgoing:
            participating = pathway_id in enabled
            if not all_pathways and not participating:
                continue
            rows = self.db.execute(
                "select data from k_rate where pathway_id = ? and histogram_params_id = ?",
                (pathway_id, rate_id),
            ).fetchall()
            if len(rows) > 1:
                raise RateCurveProvenanceError(
                    f"Ambiguous rate cache for pathway {pathway_id}"
                )
            histogram = None
            if rows and rows[0][0] is not None:
                histogram = scaled_rate_const(
                    Histogram.from_mesh(*params, numpy.array(rows[0][0]))
                )
                if len(histogram.x) != len(histogram.y):
                    raise RateCurveProvenanceError(
                        f"Invalid rate cache for pathway {pathway_id}"
                    )
            override = (
                config.get("fragmentation_energy")
                if single_pathway == pathway_id
                else None
            )
            if override is not None:
                # Match the native preparation's existing Boltzmann constant.
                energy = energy_in_joules(override)
                assert energy is not None
                kelvin = energy.to("K", "boltzmann").magnitude
                bonding = Q_(kelvin * consts.boltzmann, "J")
            else:
                energy_kelvin = FragmentationPathway(
                    clusters[parent_id].into_cpp(),
                    clusters[p1].into_cpp(),
                    clusters[p2].into_cpp(),
                ).fragmentation_energy_kelvin()
                bonding = Q_(energy_kelvin * consts.boltzmann, "J")
            curves.append(
                PathwayRateCurve(
                    pathway_id,
                    parent_id,
                    p1,
                    p2,
                    participating,
                    dos_id,
                    rate_id,
                    *params,
                    bonding,
                    histogram,
                )
            )
        return curves

    def insert_config(self, name, config):
        from apitofsim.config import dump_to_raw

        if isinstance(config, dict):
            config = dump_to_raw(config).decode("utf-8")
        id = self.db.execute(
            "insert into experiment_config values (default, ?, ?::json) returning id",
            (name, config),
        ).fetchone()
        assert id is not None
        return id[0]

    def iter_configs(self, name=None):
        import orjson

        from apitofsim.config import import_raw_config

        query = self.db.table("experiment_config")
        if name is not None:
            if isinstance(name, (tuple, list)):
                query = query.filter(
                    duckdb.ColumnExpression("name").isin(
                        *(duckdb.ConstantExpression(n) for n in name)
                    )
                )
            else:
                query = query.filter(
                    duckdb.ColumnExpression("name") == duckdb.ConstantExpression(name)
                )
        for id, name, config in query.fetchall():
            yield ConfigRow(id, name, import_raw_config(orjson.loads(config)))

    def collision_profile_inputs(self, *, run_id=None, cluster_id=None, result_id=None):
        """Read config and ClusterData for a result, or a run_id/cluster_id pair.

        Works with read-only/attached databases. The run must reference a stored
        config; missing inputs raise ValueError. No derived data is prepared.
        """
        import orjson

        from apitofsim.config import import_raw_config

        if result_id is not None:
            if run_id is not None or cluster_id is not None:
                raise ValueError("Supply result_id or run_id and cluster_id")
            result = self.db.execute(
                "select experiment_run_id, cluster_id from experiment_result where id = ?",
                [result_id],
            ).fetchone()
            if result is None:
                raise ValueError(f"Unknown experiment result: {result_id}")
            run_id, cluster_id = result
        if run_id is None or cluster_id is None:
            raise ValueError("Supply result_id or run_id and cluster_id")
        row = self.db.execute(
            """select conf.config from experiment_run as run
            join experiment_config as conf on conf.id = run.experiment_config_id
            where run.id = ?""",
            [run_id],
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"No stored configuration for experiment run: {run_id}")
        clusters = (
            self.db.table("cluster")
            .filter(
                duckdb.ColumnExpression("id") == duckdb.ConstantExpression(cluster_id)
            )
            .fetchdf()
        )
        if clusters.empty:
            raise ValueError(f"Unknown cluster: {cluster_id}")
        cluster = self._cluster_obj_from_tuple(next(iter(clusters.itertuples())))
        return import_raw_config(orjson.loads(row[0])), cluster

    def start_tree_result(
        self, run_id, cluster_id, substance, cluster_ids, pathway_ids
    ):
        """Persist the exact native tree before starting execution.

        cluster_ids maps payload indices to database IDs; pathway_ids maps native
        transition indices to database pathway IDs. Node indices are result-local.
        """
        import pyarrow as pa

        topology = substance.topology_columns()
        node_count = len(topology["node_payload"])
        transition_count = len(topology["transition_source"])
        if len(pathway_ids) != transition_count:
            raise ValueError("Expected one pathway ID per tree transition")
        with transaction_scope(self.db):
            row = self.db.execute(
                "insert into cluster_tree_experiment_result (experiment_run_id, cluster_id) values (?, ?) returning id",
                [run_id, cluster_id],
            ).fetchone()
            assert row is not None
            result_id = row[0]
            nodes = pa.table(
                {
                    "experiment_result_id": numpy.full(
                        node_count, result_id, dtype=numpy.int64
                    ),
                    "node_index": numpy.arange(node_count, dtype=numpy.int64),
                    "cluster_id": pa.array(
                        [cluster_ids[i] for i in topology["node_payload"]],
                        type=pa.int64(),
                    ),
                }
            )
            self.db.from_arrow(nodes).insert_into("cluster_tree_node")
            if transition_count:
                destinations = topology["transition_destination"]
                transitions = pa.table(
                    {
                        "experiment_result_id": numpy.full(
                            transition_count, result_id, dtype=numpy.int64
                        ),
                        "transition_index": numpy.arange(
                            transition_count, dtype=numpy.int64
                        ),
                        "source_node_index": topology["transition_source"],
                        "pathway_id": pa.array(pathway_ids, type=pa.int64()),
                        "destination_node_index": pa.array(
                            destinations, mask=destinations < 0
                        ),
                    }
                )
                self.db.from_arrow(transitions).insert_into("cluster_tree_transition")
        return result_id

    def record_result(
        self,
        run_id,
        counters,
        timings=None,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
        tree_result_id=None,
        result_id=None,
    ):
        if result_id is None:
            result_id = self.db.execute(
                "select nextval('experiment_result_sequence')"
            ).fetchall()[0][0]
        if timings is not None:
            loop_usecs = int(timings.loop / timedelta(microseconds=1))
            total_usecs = int(timings.total / timedelta(microseconds=1))
        else:
            loop_usecs = 0
            total_usecs = 0
        if pathway_id is None and (cluster_id is None or pathway_ids is None):
            raise ValueError(
                "Either pathway_id or cluster_id and pathway_ids must be provided"
            )
        if pathway_id is not None:
            id = self.db.execute(
                "insert into single_pathway_experiment_result values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) returning id",
                (
                    result_id,
                    run_id,
                    pathway_id,
                    loop_usecs,
                    total_usecs,
                    int(counters.nwarnings),
                    int(counters.n_fragmented_total[0]),
                    int(counters.n_escaped_total),
                    int(counters.ncoll_total),
                    int(counters.counter_collision_rejections),
                ),
            ).fetchone()
        else:
            values = (
                run_id,
                cluster_id,
                loop_usecs,
                total_usecs,
                int(counters.nwarnings),
                int(counters.n_escaped_total),
                int(counters.ncoll_total),
                int(counters.counter_collision_rejections),
            )
            if tree_result_id is not None:
                self.db.execute(
                    """update cluster_tree_experiment_result set
                    loop_us = ?, total_us = ?, nwarnings = ?, n_escaped_total = ?,
                    ncoll_total = ?, counter_collision_rejections = ? where id = ?""",
                    [*values[2:], tree_result_id],
                )
                id = (tree_result_id,)
            else:
                id = self.db.execute(
                    "insert into multi_pathway_experiment_result values (?, ?, ?, ?, ?, ?, ?, ?, ?) returning id",
                    (result_id, *values),
                ).fetchone()
            assert pathway_ids is not None
            assert id is not None
            counter = Counter()
            for mapped_pathway_id, cnt in zip(
                pathway_ids, counters.n_fragmented_total, strict=True
            ):
                counter[mapped_pathway_id] += cnt
            if counter:
                self.db.execute(
                    "insert into pathway_fragmentation "
                    "(experiment_result_id, pathway_id, count) "
                    "select ?, unnest(?::INTEGER[]), unnest(?::UBIGINT[])",
                    [id[0], list(counter), [int(cnt) for cnt in counter.values()]],
                )
        assert id is not None
        if not self.db.execute(
            "select 1 from event_execution where result_id=?", [id[0]]
        ).fetchall():
            self.begin_execution(
                run_id, cluster_id, pathway_id, pathway_ids or (), result_id=id[0]
            )
            self.finish_execution(id[0], "complete")
        return id[0]

    def record_failure(
        self,
        run_id,
        exc_name,
        msg,
        overflow_requested=None,
        pathway_id=None,
        cluster_id=None,
    ):
        id = self.db.execute(
            """insert into experiment_failure
            (experiment_run_id, pathway_id, cluster_id, exc_name, msg, overflow_requested)
            values (?, ?, ?, ?, ?, ?) returning id""",
            (run_id, pathway_id, cluster_id, exc_name, msg, overflow_requested),
        ).fetchone()
        assert id is not None
        return id[0]

    def add_failure_diagnostics(self, failure_id, diagnostics):
        """Shallow-merge JSON diagnostics into an existing failure.

        Accept a dict of JSON values with finite numbers.
        Supplied keys replace old values, including whole nested objects.
        Preserve the primary error and event links; unknown IDs raise KeyError.
        Add the column on first use for older databases. Use the connection's
        current result database/schema and transaction; callers handle errors.
        """
        if not isinstance(diagnostics, dict):
            raise TypeError("diagnostics must be a JSON object")
        if (
            self.db.execute(
                "select id from experiment_failure where id = ?", [failure_id]
            ).fetchone()
            is None
        ):
            raise KeyError(f"Unknown failure ID {failure_id}")
        payload = json.dumps(
            self.get_failure_diagnostics(failure_id) | diagnostics, allow_nan=False
        )
        if "diagnostics" not in self.db.table("experiment_failure").columns:
            self.db.execute(
                "alter table experiment_failure add column diagnostics json"
            )
        self.db.execute(
            "update experiment_failure set diagnostics = ? where id = ?",
            [payload, failure_id],
        )

    def get_failure_diagnostics(self, failure_id):
        """Return the diagnostics dict, or {} for an unenriched or unknown ID."""
        if "diagnostics" not in self.db.table("experiment_failure").columns:
            return {}
        row = self.db.execute(
            "select diagnostics from experiment_failure where id = ?", [failure_id]
        ).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else {}

    def report_df(self, tbl_name):
        return self.db.table(tbl_name).fetchdf()

    def experiment_overview_query(self, experiment_run_id=None):
        """Return run summaries with raw timestamps, config JSON and run_config.

        Omit the ID for all runs; a missing ID returns an empty relation.
        is_single_pathway is derived from the stored simulation_mode.
        """
        query = self.db.sql("""
            select summary.*, config.config, run.run_config
            from experiment_summary as summary
            join experiment_run as run on run.id = summary.experiment_run_id
            join experiment_config as config on config.id = run.experiment_config_id
        """)
        if experiment_run_id is not None:
            query = query.filter(
                duckdb.ColumnExpression("experiment_run_id")
                == duckdb.ConstantExpression(experiment_run_id)
            )
        return query

    def experiment_clusters_query(self, experiment_run_id, cluster_id=None):
        """Return cluster report rows for a run, optionally limited to a cluster.

        Includes raw cluster facts and result-mode metadata from
        experiment_cluster_report. Missing IDs return an empty relation.
        """
        query = self.db.table("experiment_cluster_report").filter(
            duckdb.ColumnExpression("experiment_run_id")
            == duckdb.ConstantExpression(experiment_run_id)
        )
        if cluster_id is not None:
            query = query.filter(
                duckdb.ColumnExpression("cluster_id")
                == duckdb.ConstantExpression(cluster_id)
            )
        return query

    def forget(self, runs=False, configs=False, derived=False, all=False):
        if all:
            runs = True
            configs = True
            derived = True

        if configs:
            self.db.execute("truncate experiment_config")

        if runs or configs:
            for tbl in [
                "event_execution",
                "experiment_run",
                "single_pathway_experiment_result",
                "multi_pathway_experiment_result",
                "cluster_tree_transition",
                "cluster_tree_node",
                "cluster_tree_experiment_result",
                "pathway_fragmentation",
                "experiment_failure",
            ]:
                self.db.execute(f"truncate {tbl}")

        if derived:
            if "vib_energy_index" in self.db.sql("show tables").fetchnumpy()["name"]:
                self.db.execute("truncate vib_energy_index")
            for tbl in ["cluster_dos", "products_dos", "k_rate"]:
                self.db.execute(f"truncate {tbl}")

    def is_experiment_db(self) -> bool:
        return is_experiment_db(self.db)


def is_experiment_db(db) -> bool:
    try:
        db.table("experiment_config")
    except duckdb.CatalogException:
        return False
    else:
        return True


def auto_db_type(filename, *, readonly=False):
    db, cleanup = connect_duckdb_cleanup(filename, readonly=readonly)
    conn = (db, cleanup)
    if is_experiment_db(db):
        return ExperimentDatabase(
            conn, readonly=readonly, events_filename=guess_events_filename(filename)
        )
    else:
        return SuperClusterDatabase(conn)
