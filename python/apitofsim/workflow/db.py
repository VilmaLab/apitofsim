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
from .db_utils import duckdb_connect_roview_cow, get_through_join_else

ureg = get_application_registry()
Q_ = ureg.Quantity


def guess_ase_db_filename(cluster_db_filename):
    ase_path = cluster_db_filename
    if ase_path.endswith(".duckdb"):
        ase_path = ase_path[: -len(".duckdb")]
    ase_path += ".ase.sqlite.db"
    return ase_path


@contextmanager
def connection_scope(database_type, filename, **kwargs):
    db = database_type(filename, **kwargs)
    yield db
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
        if hasattr(self, "cleanup") and self.cleanup is not None:
            self.cleanup()
        if hasattr(self, "db"):
            self.db.close()
        if hasattr(self, "ase_db") and self.ase_db is not None:
            self.ase_db.__exit__(None, None, None)

    def __del__(self):
        self.close()

    def create_tables(self):
        sql = "\n".join(self.TABLES)
        self.db.execute(sql)
        self.refresh_views()
        if self.ase_db is not None:
            self.db.execute(
                """
                alter table cluster add column ase_mol_id integer default null;
                """
            )

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

    def is_realization_db(self) -> bool:
        return False


class SuperClusterDatabase(ClusterDatabase):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
    ]
    VIEWS = [
        sql_files.pathway_report,
    ]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

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


class ExperimentDatabase(SuperClusterDatabase):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
        sql_files.experiment,
    ]
    VIEWS = [
        sql_files.pathway_report,
        sql_files.experiment_report,
    ]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

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
            "insert into experiment_run values (default, ?, ?, ?, current_timestamp) returning id",
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
        row = self.db.execute(
            "select run_config, json_merge_patch('{}', ?::json) from experiment_run where id = ?",
            [metadata, run_id],
        ).fetchone()
        assert row is not None
        previous = json.loads(row[0]) if row[0] else {}
        effective = json.loads(row[1])
        if "config" in previous and any(
            previous.get(k) != v for k, v in effective.items()
        ):
            raise ValueError(
                "Session config/cache provenance is fixed; start_run() for different inputs"
            )
        self.db.execute(
            "update experiment_run set run_config = json_merge_patch(coalesce(run_config, '{}'), ?::json) where id = ?",
            (metadata, run_id),
        )

    def record_execution(self, run_id, result_id, configured, requested, completed):
        """Save counts per result/failure, retaining compatibility with older schemas."""
        metadata = json.dumps(
            {
                "executions": {
                    str(result_id): {
                        "configured_realizations": configured,
                        "requested_realizations": requested,
                        "completed_realizations": completed,
                    }
                }
            }
        )
        self.db.execute(
            "update experiment_run set run_config = json_merge_patch(coalesce(run_config, '{}'), ?::json) where id = ?",
            [metadata, run_id],
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
        """,
            (run_id, run_id),
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

    def record_result(
        self,
        run_id,
        counters,
        timings=None,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
    ):
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
                "insert into single_pathway_experiment_result values (default, ?, ?, ?, ?, ?, ?, ?, ?, ?) returning id",
                (
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
            id = self.db.execute(
                "insert into multi_pathway_experiment_result values (default, ?, ?, ?, ?, ?, ?, ?, ?) returning id",
                (
                    run_id,
                    cluster_id,
                    loop_usecs,
                    total_usecs,
                    int(counters.nwarnings),
                    int(counters.n_escaped_total),
                    int(counters.ncoll_total),
                    int(counters.counter_collision_rejections),
                ),
            ).fetchone()
            assert pathway_ids is not None
            assert id is not None
            counter = Counter()
            for pathway_id, cnt in zip(
                pathway_ids, counters.n_fragmented_total, strict=True
            ):
                counter[pathway_id] += cnt
            for pathway_id, cnt in counter.items():
                self.db.execute(
                    "insert into pathway_fragmentation values (default, ?, ?, ?)",
                    (id[0], pathway_id, int(cnt)),
                )
        assert id is not None
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
                "experiment_run",
                "single_pathway_experiment_result",
                "multi_pathway_experiment_result",
                "pathway_fragmentation",
                "experiment_failure",
            ]:
                self.db.execute(f"truncate {tbl}")

        if derived:
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


class RealizationDatabase(ExperimentDatabase):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
        sql_files.experiment,
        sql_files.realizations,
    ]
    VIEWS = [
        sql_files.pathway_report,
        sql_files.experiment_report,
        sql_files.event_report,
    ]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

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
        """Return successful results, scoped by run, parent cluster ID and/or result ID.

        Single-pathway parents are resolved through their pathway. Multi-pathway
        results appear once, including results with no pathway_fragmentation rows.
        All supplied scope filters are intersected; omitted filters select all rows.
        """
        query = self.db.table("experiment_result").select(
            "id as experiment_result_id, * exclude (id)"
        )
        return self._scope_query(query, run_id, parent, result_id)

    def realizations_query(self, run_id=None, parent=None, result_id=None):
        """Return recorded realizations with result, run and parent IDs.

        Database scope (no filters) includes unlinked realizations; narrower scopes
        include only realizations linked to matching successful results.
        """
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

        Subtypes are left joined: collision_event_id/fragmentation_event_id reveal
        missing records even when a present fragmentation has a NULL pathway_id.
        With sort=True, order by realization, simulation time, then event ID to
        break ties. The default leaves ordering to the caller.
        """
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
            query = query.order("realization_id, postime.t, event_id")
        return query

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

    def insert_realization(self, experiment_run_id):
        id = self.db.execute(
            "insert into realization values (default, ?) returning id",
            (experiment_run_id,),
        ).fetchone()
        assert id is not None
        return id[0]

    def update_realization_experiment(self, realization_id, experiment_result_id):
        self.db.execute(
            "update realization set experiment_result_id = ? where id = ?",
            (experiment_result_id, realization_id),
        )

    def is_realization_db(self) -> bool:
        return is_realization_db(self.db)


def is_realization_db(db) -> bool:
    try:
        db.table("realization")
    except duckdb.CatalogException:
        return False
    else:
        return True


def auto_db_type(filename, *, readonly=False):
    db, cleanup = connect_duckdb_cleanup(filename, readonly=readonly)
    conn = (db, cleanup)
    if is_realization_db(db):
        return RealizationDatabase(conn)
    elif is_experiment_db(db):
        return ExperimentDatabase(conn)
    else:
        return SuperClusterDatabase(conn)


class EventRecorder:
    def __init__(self, db, pathways):
        self.db = db
        self.pathways = pathways
        assert isinstance(db, RealizationDatabase)
        self.realization_ids = {}

    def _insert_event_info(self, state, realization_id, event_type):
        row = self.db.db.execute(
            "insert into event_info values (default, ?, ?, "
            "{'x': ?, 'y': ?, 'z': ?, 't': ?}, "
            "{'x': ?, 'y': ?, 'z': ?}, {'x': ?, 'y': ?, 'z': ?}, ?, ?, ?) returning id",
            (
                realization_id,
                event_type,
                *state.postime,
                *state.velocity,
                *state.omega,
                state.rot_energy,
                state.vibrational_energy,
                state.particle_index,
            ),
        ).fetchone()
        assert row is not None
        return row[0]

    def __call__(self, event):
        from apitofsim.api import (
            CollisionEvent,
            EscapeEvent,
            FragmentationEvent,
            InitEvent,
            PostCollisionEvent,
        )

        event_type = {
            InitEvent: "init",
            CollisionEvent: "collision",
            PostCollisionEvent: "postcollision",
            FragmentationEvent: "fragmentation",
            EscapeEvent: "escape",
        }[type(event)]
        state = event.state
        if state.realization not in self.realization_ids:
            self.realization_ids[state.realization] = self.db.insert_realization(None)
        realization_id = self.realization_ids[state.realization]
        self.db.db.execute("begin transaction")
        try:
            event_id = self._insert_event_info(state, realization_id, event_type)
            if isinstance(event, CollisionEvent):
                self.db.db.execute(
                    "insert into collision_event values (?, ?, ?, ?)",
                    (event_id, event.theta, event.u_norm, event.accepted),
                )
            elif isinstance(event, FragmentationEvent):
                self.db.db.execute(
                    "insert into fragmentation_event values (?, ?)",
                    (event_id, self.pathways[event.pathway_index]),
                )
            self.db.db.execute("commit")
        except Exception:
            self.db.db.execute("rollback")
            raise

    def relate_realizations(self, experiment_result_id):
        for realization_id in self.realization_ids.values():
            self.db.update_realization_experiment(realization_id, experiment_result_id)
