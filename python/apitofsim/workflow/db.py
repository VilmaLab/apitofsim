# pyright: reportAttributeAccessIssue=false

from collections import namedtuple
from datetime import timedelta

import duckdb
import numpy
import pandas
from ase.db import connect as connect_ase_db
from pint import get_application_registry

import apitofsim.workflow.sql_files as sql_files
from apitofsim import ClusterData

from .db_utils import duckdb_connect_roview_cow

ureg = get_application_registry()
Q_ = ureg.Quantity


def guess_ase_db_filename(cluster_db_filename):
    ase_path = cluster_db_filename
    if ase_path.endswith(".duckdb"):
        ase_path = ase_path[: -len(".duckdb")]
    ase_path += ".ase.sqlite.db"
    return ase_path


class ClusterDatabase:
    TABLES = [sql_files.pathway]

    def __init__(self, filename, *, readonly=False, ase_filename=None):
        self.cleanup = None
        if readonly:
            self.db, self.cleanup = duckdb_connect_roview_cow(
                filename, fallback="connect"
            )
        else:
            self.db = duckdb.connect(filename)
        if ase_filename is not None:
            # TODO: These ClusterDatabase, etc. objects should probably be context managers too
            self.ase_db = connect_ase_db(ase_filename, type="db").__enter__()
        else:
            self.ase_db = None
        self._setup_db()

    def _setup_db(self):
        import os

        self.db.execute("SET preserve_insertion_order=false")
        if "DUCKDB_MEMORY_LIMIT" in os.environ:
            memory_limit = os.environ["DUCKDB_MEMORY_LIMIT"]
            self.db.execute(f"set memory_limit='{memory_limit}';")

    def __del__(self):
        if self.cleanup is not None:
            self.cleanup()
        if self.ase_db is not None:
            self.ase_db.__exit__(None, None, None)

    def create_tables(self):
        sql = "\n".join(self.TABLES)
        self.db.execute(sql)
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
            parent = self.db.db.execute(
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
                pathway_common_names = self.db.db.sql(
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
                        self.db.db.from_arrow(wanted_tbl).set_alias("wanted"),
                        condition=(
                            "wanted.pathway = cluster_common_name "
                            "and ((wanted.product1 = product1_common_name and wanted.product2 = product2_common_name) "
                            "or (wanted.product1 = product2_common_name and wanted.product2 = product1_common_name))"
                        ),
                    )
                    .select("pathway_id")
                    .fetch_arrow_table()["pathway_id"]
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


class SuperClusterDatabase(ClusterDatabase):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
        sql_files.pathway_report,
    ]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

    def refresh_views(self):
        self.db.execute(sql_files.pathway_report)
        self.db.execute(sql_files.experiment_report)


ConfigRow = namedtuple("ConfigRow", ["id", "name", "config"])


class ExperimentDatabase(SuperClusterDatabase):
    TABLES = [
        sql_files.pathway,
        sql_files.histograms,
        sql_files.experiment,
        sql_files.pathway_report,
        sql_files.experiment_report,
    ]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

    def insert_run(self, config_id=None, pathway_at_a_time=False):
        id = self.db.execute(
            "insert into experiment_run values (default, ?, ?, current_timestamp) returning id",
            (config_id, pathway_at_a_time),
        ).fetchone()
        assert id is not None
        return id[0]

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

    def record_result(
        self,
        run_id,
        counters,
        timings,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
    ):
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
                    int(timings.loop / timedelta(microseconds=1)),
                    int(timings.total / timedelta(microseconds=1)),
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
                    int(timings.loop / timedelta(microseconds=1)),
                    int(timings.total / timedelta(microseconds=1)),
                    int(counters.nwarnings),
                    int(counters.n_escaped_total),
                    int(counters.ncoll_total),
                    int(counters.counter_collision_rejections),
                ),
            ).fetchone()
            assert pathway_ids is not None
            assert id is not None
            for pathway_id, fragmented in zip(
                pathway_ids, counters.n_fragmented_total, strict=True
            ):
                self.db.execute(
                    "insert into pathway_fragmentation values (default, ?, ?, ?)",
                    (id[0], pathway_id, int(fragmented)),
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
            "insert into experiment_failure values (default, ?, ?, ?, ?, ?, ?) returning id",
            (run_id, pathway_id, cluster_id, exc_name, msg, overflow_requested),
        ).fetchone()
        assert id is not None
        return id[0]

    def report_df(self, tbl_name):
        return self.db.table(tbl_name).fetchdf()

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
