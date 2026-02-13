# pyright: reportAttributeAccessIssue=false

import numpy
from collections import namedtuple
import pandas
import duckdb
from pint import get_application_registry
from apitofsim import ClusterData
from datetime import timedelta
from typing import Callable

from glob import glob
from os.path import dirname, isfile, basename, expanduser

from .api import (
    ApiTofError,
    ApiTofOverflowError,
    MeshMode,
    MassSpectrometer,
    MassSpecSubstanceInput,
    MassSpecInputFragmentationPathway,
)


def duckdb_connect_roview_cow(filename, *, config=None, fallback="copy"):
    import duckdb
    from reflink import reflink, ReflinkImpossibleError
    from shutil import copy
    from uuid import uuid4
    from os.path import split as psplit, join as pjoin, splitext

    if config is None:
        config = {}
    path, base = psplit(filename)
    base, ext = splitext(base)
    rnd = uuid4().hex
    dest = pjoin(path, f".{base}.rosnap.{rnd}.{ext}")
    try:
        reflink(filename, dest)
    except (NotImplementedError, ReflinkImpossibleError):
        if fallback == "copy":
            copy(filename, dest)
        elif fallback == "connect":
            dest = filename
        elif fallback == "error":
            raise
        else:
            raise ValueError(f"Invalid fallback option: {fallback}")
    return duckdb.connect(dest, read_only = True, config = config)


ureg = get_application_registry()
Q_ = ureg.Quantity

PATHWAY_TABLES = """
create sequence cluster_id_sequence start 1;
create sequence pathway_id_sequence start 1;

create table cluster (
    id integer default nextval('cluster_id_sequence') primary key,
    common_name varchar,
    atomic_mass double,
    charge int,
    electronic_energy double,
    rotational_temperatures double[3],
    vibrational_temperatures double[],
    import_info json
);

create table pathway (
    id integer default nextval('pathway_id_sequence') primary key,
    cluster_id integer not null,
    product1_id integer not null,
    product2_id integer not null,
    foreign key (cluster_id) references cluster (id),
    foreign key (product1_id) references cluster (id),
    foreign key (product2_id) references cluster (id)
);
"""


class ClusterDatabase:
    TABLES = [PATHWAY_TABLES]

    def __init__(self, filename, readonly=False):
        if readonly:
            self.db = duckdb_connect_roview_cow(filename)
        else:
            self.db = duckdb.connect(filename)

    def create_tables(self):
        sql = "\n".join(self.TABLES)
        self.db.execute(sql)

    def clusters_query(self, parent=None, parents_only=False, children_only=False):
        if parents_only and children_only:
            raise ValueError("Cannot set both parents_only and children_only to True")
        query = self.db.table("cluster")
        if parent is None and not (parents_only or children_only):
            # Shortcut for efficiency
            return query
        if parents_only:
            relevant_fragment = "cluster_id"
        elif children_only:
            relevant_fragment = "product1_id, product2_id"
        else:
            relevant_fragment = "cluster_id, product1_id, product2_id"
        pathways_query = self.pathways_query(parent)
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

    def pathways_query(self, parent=None, sort=False):
        pathway = self.db.table("pathway")
        if parent is not None:
            pathway = pathway.filter(
                duckdb.ColumnExpression("cluster_id")
                == duckdb.ConstantExpression(parent)
            )
        if sort:
            pathway = pathway.sort(
                "cluster_id",
                "product1_id",
                "product2_id",
            )
        return pathway

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
        allow_duplicates=False,
    ):
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
            "insert into cluster values (default, ?, ?, ?, ?, ?, ?, ?) returning id",
            (
                name,
                atomic_mass,
                charge,
                electronic_energy,
                rotational_temperatures,
                vibrational_temperatures,
                import_info,
            ),
        ).fetchone()
        assert id is not None
        return True, id[0]

    def insert_pathway(self, parent_id, product1_id, product2_id):
        self.db.execute(
            "insert into pathway values (default, ?, ?, ?)",
            (parent_id, product1_id, product2_id),
        )


def insert_parsed_pathway(db, pathway, *, prefix=None):
    from apitofsim.config import dump_to_raw

    ids = []
    for particle_info in pathway:
        name = particle_info["name"]
        if prefix is not None:
            name = prefix + name
        combined = particle_info["particle"]
        with ureg.context("boltzmann", "spectroscopy"):
            inserted, id = db.insert_cluster(
                name,
                combined["atomic_mass"].to("amu").magnitude,
                combined["charge"],
                combined["electronic_energy"].to("hartree").magnitude,
                combined["rotational_temperatures"].to("K").magnitude
                if combined["rotational_temperatures"] is not None
                else None,
                combined["vibrational_temperatures"].to("K").magnitude
                if combined["vibrational_temperatures"] is not None
                else None,
                dump_to_raw(particle_info).decode("utf-8"),
                allow_duplicates=True,
            )
            ids.append(id)
    db.insert_pathway(*ids)


def ingest_legacy_one(db: ClusterDatabase, filename, clusters, prefix=None):
    from apitofsim.ingest.legacy import parse_legacy_one

    pathway = parse_legacy_one(filename, clusters)
    insert_parsed_pathway(db, pathway, prefix=None)


def ingest_tree(db: ClusterDatabase, pathways):
    if isinstance(pathways, list):
        for pathways_segment in pathways:
            ingest_tree(db, pathways_segment)
        return
    if pathways["type"] == "legacy_glob":
        from apitofsim.ingest.legacy import parse_legacy_tree

        for pathway in parse_legacy_tree(pathways["path"], pathways["clusters"]):
            insert_parsed_pathway(db, pathway, prefix=pathways.get("prefix"))


DERIVED_TABLES = """
create sequence histogram_params_sequence start 1;
create sequence dos_sequence start 1;
create sequence products_dos_sequence start 1;
create sequence k_rate_mesh_sequence start 1;
create sequence k_rate_sequence start 1;

create table histogram_params (
    id integer default nextval('histogram_params_sequence') primary key,
    bin_width double,
    max double
);

create table cluster_dos (
    id integer default nextval('dos_sequence') primary key,
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    cluster_id integer not null,
    foreign key (cluster_id) references cluster (id),
    data double[]
);

create table products_dos (
    id integer default nextval('products_dos_sequence') primary key,
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    cluster1_id integer not null,
    cluster2_id integer not null,
    foreign key (cluster1_id) references cluster (id),
    foreign key (cluster2_id) references cluster (id),
    data double[]
);

create table k_rate_mesh (
    id integer default nextval('k_rate_mesh_sequence') primary key,
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    data double[]
);

create table k_rate (
    id integer default nextval('k_rate_sequence') primary key,
    pathway_id integer not null,
    foreign key (pathway_id) references pathway (id),
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    data double[]
);
"""


PATHWAY_REPORT_VIEW = """
create or replace view pathway_report as
select
    -- Pathway info
    p.id as pathway_id,

    -- Cluster info (the main cluster)
    c.id as cluster_id,
    c.common_name as cluster_common_name,
    c.atomic_mass as cluster_atomic_mass,
    c.electronic_energy as cluster_electronic_energy,
    c.rotational_temperatures as cluster_rotational_temperatures,
    c.vibrational_temperatures as cluster_vibrational_temperatures,

    -- Product 1 info
    p1.id as product1_id,
    p1.common_name as product1_common_name,
    p1.atomic_mass as product1_atomic_mass,
    p1.electronic_energy as product1_electronic_energy,
    p1.rotational_temperatures as product1_rotational_temperatures,
    p1.vibrational_temperatures as product1_vibrational_temperatures,

    -- Product 2 info
    p2.id as product2_id,
    p2.common_name as product2_common_name,
    p2.atomic_mass as product2_atomic_mass,
    p2.electronic_energy as product2_electronic_energy,
    p2.rotational_temperatures as product2_rotational_temperatures,
    p2.vibrational_temperatures as product2_vibrational_temperatures,

from pathway p
inner join cluster c on c.id = p.cluster_id
inner join cluster p1 on p1.id = p.product1_id
inner join cluster p2 on p2.id = p.product2_id;
"""


class SuperClusterDatabase(ClusterDatabase):
    TABLES = [PATHWAY_TABLES, DERIVED_TABLES, PATHWAY_REPORT_VIEW]

    def __init__(self, filename):
        super().__init__(filename)

    def refresh_views(self):
        self.db.execute(PATHWAY_REPORT_VIEW)
        self.db.execute(EXPERIMENT_REPORT_VIEW)


EXPERIMENT_TABLES = """
create sequence experiment_config_sequence start 1;
create sequence experiment_run_sequence start 1;
create sequence experiment_result_sequence start 1;
create sequence pathway_fragmentation_sequence start 1;

create table experiment_config (
    id integer default nextval('experiment_config_sequence') primary key,
    name varchar,
    config json
);

create table experiment_run (
    id integer default nextval('experiment_run_sequence') primary key,
    experiment_config_id integer not null,
    pathway_at_a_time bool default false,
    foreign key (experiment_config_id) references experiment_config (id),
    start_time timestamp
);

create table single_pathway_experiment_result (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    pathway_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    foreign key (pathway_id) references pathway (id),
    loop_us integer,
    total_us integer,
    nwarnings integer,
    n_fragmented_total integer,
    n_escaped_total integer,
    ncoll_total integer,
    counter_collision_rejections integer
);

create table multi_pathway_experiment_result (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    cluster_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    foreign key (cluster_id) references cluster (id),
    loop_us integer,
    total_us integer,
    nwarnings integer,
    n_escaped_total integer,
    ncoll_total integer,
    counter_collision_rejections integer
);

create table pathway_fragmentation (
    id integer default nextval('pathway_fragmentation_sequence') primary key,
    experiment_result_id integer not null,
    foreign key (experiment_result_id) references multi_pathway_experiment_result (id),
    pathway_id integer not null,
    foreign key (pathway_id) references pathway (id),
    count integer
);

create table experiment_failure (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    pathway_id integer,
    foreign key (pathway_id) references pathway (id),
    cluster_id integer,
    foreign key (cluster_id) references cluster (id),
    exc_name varchar,
    msg varchar,
    overflow_requested double
);
"""


EXPERIMENT_REPORT_VIEW = """
create or replace view experiment_report as
select
    -- Experiment run info
    er.id as experiment_run_id,
    er.start_time,

    -- Config info
    conf.name as config_name,
    conf.config as config,

    -- Result/Failure info
    res.id as result_id,
    case when res.msg is not null then 'failure' else 'result' end as outcome_type,

    -- Pathway info
    p.id as pathway_id,

    -- Cluster info (the main cluster)
    c.id as cluster_id,
    c.common_name as cluster_common_name,
    c.atomic_mass as cluster_atomic_mass,
    c.electronic_energy as cluster_electronic_energy,
    c.rotational_temperatures as cluster_rotational_temperatures,
    c.vibrational_temperatures as cluster_vibrational_temperatures,

    -- Product 1 info
    p1.id as product1_id,
    p1.common_name as product1_common_name,
    p1.atomic_mass as product1_atomic_mass,
    p1.electronic_energy as product1_electronic_energy,
    p1.rotational_temperatures as product1_rotational_temperatures,
    p1.vibrational_temperatures as product1_vibrational_temperatures,

    -- Product 2 info
    p2.id as product2_id,
    p2.common_name as product2_common_name,
    p2.atomic_mass as product2_atomic_mass,
    p2.electronic_energy as product2_electronic_energy,
    p2.rotational_temperatures as product2_rotational_temperatures,
    p2.vibrational_temperatures as product2_vibrational_temperatures,

    -- Result/failure fields
    res.msg as failure_msg,
    res.loop_us,
    res.total_us,
    res.nwarnings,
    res.n_fragmented_total,
    res.n_escaped_total,
    res.ncoll_total,
    res.counter_collision_rejections

from experiment_run as er
left join (
    select * from single_pathway_experiment_result
    union by name
    select * from experiment_failure
) as res on res.experiment_run_id = er.id
inner join experiment_config as conf on conf.id = er.experiment_config_id
inner join pathway p on p.id = res.pathway_id
inner join cluster c on c.id = p.cluster_id
inner join cluster p1 on p1.id = p.product1_id
inner join cluster p2 on p2.id = p.product2_id

--where res.id is not null or fail.id is not null;
"""


ConfigRow = namedtuple("ConfigRow", ["id", "name", "config"])


def get_or_insert(db, tbl, **vals):
    query = db.table(tbl)
    for col_name, col_value in vals.items():
        query = query.filter(
            duckdb.ColumnExpression(col_name) == duckdb.ConstantExpression(col_value)
        )
    result = list(query.fetchall())
    if len(result) > 1:
        raise ValueError("Multiple rows found")
    if len(result) == 1:
        return result[0][0]
    expression = (
        "insert into "
        + tbl
        + " ("
        + ",".join(vals.keys())
        + ") "
        + "values ("
        + ",".join("?" for _ in vals)
        + ") returning id"
    )
    id = db.execute(expression, tuple(vals.values())).fetchone()
    return id[0]


class ExperimentDatabase(SuperClusterDatabase):
    TABLES = [
        PATHWAY_TABLES,
        DERIVED_TABLES,
        EXPERIMENT_TABLES,
        PATHWAY_REPORT_VIEW,
        EXPERIMENT_REPORT_VIEW,
    ]

    def __init__(self, filename):
        super().__init__(filename)

    def insert_run(self, config_id=None, pathway_at_a_time=False):
        id = self.db.execute(
            "insert into experiment_run values (default, ?, ?, current_timestamp) returning id",
            (config_id, pathway_at_a_time),
        ).fetchone()
        assert id is not None
        return id[0]

    def insert_config(self, name, config):
        from .config import dump_to_raw

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
        from .config import import_raw_config

        query = self.db.table("experiment_config")
        if name is not None:
            if isinstance(name, list):
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
                    timings.loop / timedelta(microseconds=1),
                    timings.total / timedelta(microseconds=1),
                    int(counters.nwarnings),
                    int(counters.n_fragmented_total),
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
                    timings.loop / timedelta(microseconds=1),
                    timings.total / timedelta(microseconds=1),
                    int(counters.nwarnings),
                    int(counters.n_escaped_total),
                    int(counters.ncoll_total),
                    int(counters.counter_collision_rejections),
                ),
            ).fetchone()
            assert pathway_ids is not None
            for pathway_id, fragmented in zip(pathway_ids, counters.n_fragmented_total):
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

    def export(self, out_path, experiment_id=None):
        if experiment_id:
            where_clause = f" where experiment_run_id = {experiment_id}"
        else:
            where_clause = ""
        self.db.execute(
            f"copy (select * from experiment_report{where_clause}) to '{out_path}' (header, delimiter ',');"
        )

    def experiment_summary_df(self):
        return self.db.execute(
            """
        select
            er.id as experiment_run_id,
            conf.name as config_name,
            er.start_time,
            (
                select count()
                from experiment_result
                where experiment_result.experiment_run_id = er.id
            ) as successes,
            (
                select count()
                from experiment_failure
                where experiment_failure.experiment_run_id = er.id
            ) as failures,
        from experiment_run er
        join experiment_config conf on conf.id = er.experiment_config_id
        """
        ).fetchdf()


def counter_fragmented_total(counters):
    return int(counters.n_fragmented_total.sum())


def get_through_join_else(conn, rel, proj_col, result_dict, **match_cols):
    import pyarrow as pa

    wanted_tbl = pa.table(list(match_cols.values()), names=list(match_cols.keys()))
    data = (
        rel.set_alias("rel")
        .join(
            conn.from_arrow(wanted_tbl).set_alias("wanted"),
            condition=" and ".join(f"wanted.{k} = rel.{k}" for k in match_cols.keys()),
            how="right",
        )
        .select(proj_col)
        .fetch_arrow_table()
    )
    data = data.column(proj_col).chunk(0)
    for match_row, value in zip(zip(*match_cols.values()), data):
        value = value.values
        if value is not None:
            data = value.to_numpy(zero_copy_only=True)
            if len(match_row) == 1:
                match_row = match_row[0]
            result_dict[match_row] = data
        else:
            yield dict(zip(match_cols.keys(), match_row))


class DerivedDataPreparer:
    def __init__(self, db: ExperimentDatabase):
        self.db = db

    def _run_skimmer(self, config):
        from apitofsim import skimmer

        return skimmer(
            T0=config["T"],
            P0=config["pressure_first"],
            rmax=config["lengths"][-1],
            dc=config["dc"],
            alpha_factor=config["alpha_factor"],
            gas=config["gas"],
            N=config["N_iter"],
            M=config["M_iter"],
            resolution=config["resolution"],
            tolerance=config["tolerance"],
        )

    def _run_dos(
        self,
        cluster_indexed,
        histogram_id,
        bin_width,
        energy_max,
        pathway_lookup,
        status_table=None,
    ):
        import pyarrow as pa
        from apitofsim import ProductsCluster, compute_density_of_states_batch

        num_clusters_missed = 0
        cluster_dos_dict = {}
        missed_cluster_ids = []
        wanted_cluster_ids = numpy.array(list(cluster_indexed.keys()))
        density_of_states_inputs = []
        for cluster_id in get_through_join_else(
            self.db.db,
            self.db.db.table("cluster_dos"),
            "data",
            cluster_dos_dict,
            cluster_id=wanted_cluster_ids,
        ):
            density_of_states_inputs.append(cluster_indexed[cluster_id])
            num_clusters_missed += 1

        wanted_p1 = []
        wanted_p2 = []
        for _, product1_id, product2_id in pathway_lookup.values():
            product1_id, product2_id = sorted((product1_id, product2_id))
            wanted_p1.append(product1_id)
            wanted_p2.append(product2_id)

        for p1, p2 in get_through_join_else(
            self.db.db,
            self.db.db.table("products_dos"),
            "data",
            cluster_dos_dict,
            cluster1_id=wanted_p1,
            cluster2_id=wanted_p2,
        ):
            products = ProductsCluster(cluster_indexed[p1], cluster_indexed[p2])
            density_of_states_inputs.append(products)

        if status_table is not None:
            status_table["Description"] = (
                f"{len(cluster_dos_dict)} retrieved; {len(density_of_states_inputs)} to compute"
            )

        if len(density_of_states_inputs) > 0:
            density_of_states = compute_density_of_states_batch(
                density_of_states_inputs,
                energy_max=energy_max,
                bin_width=bin_width,
            )
            assert density_of_states.flags.f_contiguous
            if num_clusters_missed > 0:
                arrow_arr = pa.FixedSizeListArray.from_arrays(
                    density_of_states[:, :num_clusters_missed].flatten("K"),
                    density_of_states.shape[0],
                )
                arrow_table = pa.table(
                    {
                        "histogram_params_id": [histogram_id] * len(missed_cluster_ids),
                        "cluster_id": missed_cluster_ids,
                        "data": arrow_arr,
                    }
                )

                self.db.db.register("arrow_table", arrow_table)
                self.db.db.execute("set preserve_insertion_order=false;")
                self.db.db.execute(
                    "insert into cluster_dos by name select * from arrow_table"
                )

            if len(density_of_states_inputs) > num_clusters_missed:
                arrow_arr = pa.FixedSizeListArray.from_arrays(
                    density_of_states[:, num_clusters_missed:].flatten("K"),
                    density_of_states.shape[0],
                )
                arrow_table = pa.table(
                    {
                        "histogram_params_id": [histogram_id] * len(wanted_p1),
                        "cluster1_id": wanted_p1,
                        "cluster2_id": wanted_p2,
                        "data": arrow_arr,
                    }
                )

                self.db.db.register("arrow_table", arrow_table)
                self.db.db.execute("set preserve_insertion_order=false;")
                self.db.db.execute(
                    "insert into products_dos by name select * from arrow_table"
                )

            for cluster_id, v in zip(
                missed_cluster_ids, density_of_states[:, : len(missed_cluster_ids)].T
            ):
                cluster_dos_dict[cluster_id] = v

            for p1, p2, v in zip(
                wanted_p1, wanted_p2, density_of_states[:, len(missed_cluster_ids) :].T
            ):
                cluster_dos_dict[(p1, p2)] = v
        return cluster_dos_dict

    def _run_k_total(
        self,
        cluster_indexed,
        cluster_dos_dict,
        histogram_id,
        mesh,
        bin_width,
        energy_max,
        energy_max_rate,
        pathway_lookup,
        status_table=None,
    ):
        from apitofsim import (
            KTotalInput,
            compute_k_total_batch,
            FragmentationPathway,
        )
        from apitofsim.api import validate_max_energies
        import pyarrow as pa

        histogram_id = get_or_insert(
            self.db.db,
            "histogram_params",
            bin_width=bin_width.to("K").magnitude,
            max=energy_max_rate.to("K").magnitude,
        )

        k_total_inputs = []
        k_total_keys = []
        k_total_dict = {}

        for pathway_id in get_through_join_else(
            self.db.db,
            self.db.db.table("k_rate").filter(
                duckdb.ColumnExpression("histogram_params_id")
                == duckdb.ConstantExpression(histogram_id)
            ),
            "data",
            k_total_dict,
            pathway_id=pathway_lookup.keys(),
        ):
            cluster_id, product1_id, product2_id = pathway_lookup[pathway_id]
            product1_cpp = cluster_indexed[product1_id].into_cpp()
            product2_cpp = cluster_indexed[product2_id].into_cpp()
            fragmentation_energy = FragmentationPathway(
                cluster_indexed[cluster_id].into_cpp(), product1_cpp, product2_cpp
            ).fragmentation_energy_kelvin()
            validate_max_energies(
                fragmentation_energy=fragmentation_energy,
                bin_width=bin_width,
                energy_max=energy_max,
                energy_max_rate=energy_max_rate,
                quantities_strict=False,
            )
            product_id_tpl = tuple(sorted((product1_id, product2_id)))
            k_total_keys.append(pathway_id)
            assert cluster_dos_dict[cluster_id].flags.c_contiguous
            assert cluster_dos_dict[product_id_tpl].flags.c_contiguous
            k_total_inputs.append(
                KTotalInput(
                    product1_cpp,
                    product2_cpp,
                    fragmentation_energy,
                    # Need to copy here for some reason?
                    numpy.copy(cluster_dos_dict[cluster_id]),
                    numpy.copy(cluster_dos_dict[product_id_tpl]),
                )
            )

        progress_callback: Callable[[int], None] | None = None
        if status_table is not None:
            status_table["Description"] = (
                f"{len(k_total_dict)} retrieved; {len(k_total_inputs)} to compute"
            )

            inner_pbar = status_table(
                len(k_total_inputs),
                position=1,
                description="K total",
                show_throughput=False,
                show_progress=True,
                show_eta=True,
            )

            def _progress_callback(iters_done):
                inner_pbar.set_step(iters_done)

            progress_callback = _progress_callback

        k_rates = compute_k_total_batch(
            k_total_inputs,
            energy_max_rate=energy_max_rate,
            bin_width=bin_width,
            mesh=mesh,
            progress_callback=progress_callback,
        )

        arrow_table = pa.table(
            {
                "histogram_params_id": [histogram_id] * k_rates.shape[1],
                "pathway_id": k_total_keys,
                "data": pa.FixedSizeListArray.from_arrays(
                    k_rates.flatten("K"), k_rates.shape[0]
                ),
            }
        )

        self.db.db.register("arrow_table", arrow_table)
        self.db.db.execute("set preserve_insertion_order=false;")
        self.db.db.execute("insert into k_rate by name select * from arrow_table")

        for key, k_rates in zip(k_total_keys, k_rates.T):
            k_total_dict[key] = k_rates

        return k_total_dict

    def run_preliminaries(self, config, cluster_indexed, pathway_lookup):
        from apitofsim import precompute_mesh
        from timeit import default_timer as timer
        from progress_table import ProgressTable
        from pprint import pprint

        prelim_table = ProgressTable(default_column_alignment="left", refresh_rate=0)
        outer_pbar = prelim_table(
            4,
            description="Preliminary steps",
            show_throughput=False,
            show_progress=True,
            position=2,
        )
        prelim_table.add_column("#", alignment="right", width=3)
        prelim_table.add_column("Step", width=20)
        prelim_table.add_column("Description", width=40)
        prelim_table.add_column("Time (s)", alignment="right")
        prelim_table["#"] = "1/4"
        prelim_table["Step"] = "Skimmer"
        outer_pbar.update()
        start = timer()
        skimmer_np = self._run_skimmer(config)
        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.next_row()

        prelim_table["#"] = "2/4"
        prelim_table["Step"] = "Density of states"
        outer_pbar.update()
        start = timer()

        histogram_id = get_or_insert(
            self.db.db,
            "histogram_params",
            bin_width=config["bin_width"].to("K").magnitude,
            max=config["energy_max"].to("K").magnitude,
        )

        cluster_dos_dict = self._run_dos(
            cluster_indexed,
            histogram_id,
            config["bin_width"],
            config["energy_max"],
            pathway_lookup=pathway_lookup,
            status_table=prelim_table,
        )

        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.next_row()

        prelim_table["#"] = "3/4"
        prelim_table["Step"] = "Computing mesh"
        outer_pbar.update()
        start = timer()

        mesh_points = config["energy_max_rate"] / config["bin_width"]
        prelim_table["Description"] = f"Mesh of {mesh_points} pts"

        mesh = precompute_mesh(
            energy_max_rate=config["energy_max_rate"],
            bin_width=config["bin_width"],
            mesh_mode=MeshMode.compute_mesh_diagonal_multithreaded,
        )

        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.next_row()

        prelim_table["#"] = "4/4"
        prelim_table["Step"] = "K total"
        outer_pbar.update()
        start = timer()

        k_rates = self._run_k_total(
            cluster_indexed,
            cluster_dos_dict,
            histogram_id,
            mesh,
            config["bin_width"],
            config["energy_max"],
            config["energy_max_rate"],
            pathway_lookup=pathway_lookup,
            status_table=prelim_table,
        )

        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.close()

        cluster_dos_by_pathway = {}

        for pathway_id, (
            cluster_id,
            product1_id,
            product2_id,
        ) in pathway_lookup.items():
            product1_id, product2_id = sorted((product1_id, product2_id))
            cluster_dos_by_pathway[pathway_id] = cluster_dos_dict[
                (product1_id, product2_id)
            ]

        return skimmer_np, k_rates, cluster_dos_by_pathway


class ExperimentRunner:
    def __init__(self, db: ExperimentDatabase):
        self.db = db
        self.preparer = DerivedDataPreparer(db)
        self.current_run_id = None

    def _guard_run_started(self):
        if self.current_run_id is None:
            raise RuntimeError("No experiment run started; call start_run() first")

    def start_run(self, config_id=None, **kwargs):
        self.current_run_id = self.db.insert_run(config_id, **kwargs)

    def run_mass_spec(
        self,
        *args,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
        strict=False,
        strict_dos=True,
        **kwargs,
    ):
        self._guard_run_started()
        if pathway_id is None and cluster_id is None:
            raise ValueError("Either pathway_id or cluster_id must be provided")
        from .api import mass_spec

        counters = None
        try:
            counters, timings = mass_spec(
                *args,
                **kwargs,
                named_tuple_counters=True,
                output_timings=True,
                strict=strict_dos,
            )
        except ApiTofError as e:
            if strict:
                raise
            overflow_requested = None
            if isinstance(e, ApiTofOverflowError):
                overflow_requested = e.current
            self.db.record_failure(
                self.current_run_id,
                type(e).__name__,
                str(e),
                overflow_requested,
                pathway_id=pathway_id,
                cluster_id=cluster_id,
            )
        else:
            self.db.record_result(
                self.current_run_id,
                counters,
                timings,
                pathway_id=pathway_id,
                pathway_ids=pathway_ids,
                cluster_id=cluster_id,
            )
        return counters

    def run_from_config(
        self,
        config,
        run_started=False,
        strict_dos=True,
        pathway_at_a_time=False,
        parent_name=None,
    ):
        if not run_started:
            self.db.start_run()

        if parent_name is not None:
            parent_id = self.db.db.execute(
                """
                select id
                from cluster
                where common_name = ?
                """,
                (parent_name,),
            ).fetchone()
            if parent_id is None:
                raise ValueError(f"No cluster found with name {parent_name}")
            parent_id = parent_id[0]
        else:
            parent_id = None
        cluster_indexed, name_lookup = self.db.clusters_objects_indexed(
            include_name_lookup=True, parent=parent_id
        )
        pathway_lookup = {}
        for pathway_id, cluster_id, product1_id, product2_id in self.db.pathways_ids(
            parent=parent_id
        ):
            pathway_lookup[pathway_id] = (cluster_id, product1_id, product2_id)

        skimmer_np, k_rates, cluster_dos = self.preparer.run_preliminaries(
            config, cluster_indexed, pathway_lookup=pathway_lookup
        )

        assert isinstance(skimmer_np, numpy.ndarray)
        mass_spec = MassSpectrometer(
            skimmer_np,
            config["lengths"],
            config["voltages"],
            config["T"],
            Q_(
                numpy.array(
                    [
                        config["pressure_first"].to("pascals").magnitude,
                        config["pressure_second"].to("pascals").magnitude,
                    ]
                ),
                "pascals",
            ),
            quadrupole=config.get("quadrupole"),
        )

        if pathway_at_a_time:
            self._run_pathways_at_a_time(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
            )
        else:
            self._run_cluster_grouped(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
            )

    def _run_pathways_at_a_time(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
        parent=None,
    ):
        from os import environ
        from progress_table import ProgressTable
        from apitofsim.api import Histogram
        from timeit import default_timer as timer

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            pbar_show_throughput=False,
            refresh_rate=0,
        )
        mass_spec_table.add_column("#")
        mass_spec_table.add_column("Cluster", alignment="left")
        mass_spec_table.add_column("Products", width=16, alignment="left")
        mass_spec_table.add_columns(
            "Frags",
            "Intacts",
            "Avg colls",
            "PH rej",
        )
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        for pathway_id, (cluster_id, product1_id, product2_id) in mass_spec_table(
            pathway_lookup.items()
        ):
            density_cluster = cluster_dos[pathway_id]
            rate_const = k_rates[pathway_id]
            cluster = cluster_indexed[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            mass_spec_table["#"] = f"{cluster_seq}/{len(pathway_lookup)}"
            mass_spec_table["Cluster"] = name_lookup[cluster_id]
            mass_spec_table["Products"] = (
                f"{name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
            start = timer()
            density_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max"],
                density_cluster,
            )
            rate_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max_rate"],
                rate_const,
            )
            subs = MassSpecSubstanceInput(
                cluster,
                product1,
                product2,
                config["gas"],
                density_hist,
                rate_hist,
                fragmentation_energy=config.get("fragmentation_energy"),
                cluster_charge_sign=config.get("cluster_charge_sign", 1),
            )
            counters = self.run_mass_spec(
                mass_spec,
                subs,
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"],
                pathway_id=pathway_id,
                sample_mode=2,
                loglevel=0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                realizations = counters.n_fragmented_total + counters.n_escaped_total
                mass_spec_table["Frags"] = int(counters.n_fragmented_total)
                mass_spec_table["Intacts"] = int(counters.n_escaped_total)
                mass_spec_table["Avg colls"] = counters.ncoll_total / realizations
                mass_spec_table["PH rej"] = int(counters.counter_collision_rejections)
                mass_spec_table["Surv prob"] = counters.n_escaped_total / realizations
                mass_spec_table["Warns"] = int(counters.nwarnings)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            cluster_seq += 1

    def _run_cluster_grouped(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
    ):
        from os import environ
        from progress_table import ProgressTable
        from apitofsim.api import Histogram
        from timeit import default_timer as timer

        last_cluster_id = None
        groups = []
        cur_group = None
        for pathway_id, (
            cluster_id,
            product1_id,
            product2_id,
        ) in pathway_lookup.items():
            rate_const = k_rates[pathway_id]
            density_cluster = cluster_dos[pathway_id]
            cluster = cluster_indexed[cluster_id]
            if cluster_id != last_cluster_id:
                if cur_group is not None:
                    groups.append(cur_group)
                density_hist = Histogram.from_mesh(
                    config["bin_width"],
                    config["energy_max"],
                    density_cluster,
                )
                cur_group = {
                    "pathways": [],
                    "cluster": cluster,
                    "cluster_id": cluster_id,
                    "cluster_label": name_lookup[cluster_id],
                    "density_hist": density_hist,
                    "product_labels": [],
                    "pathway_ids": [],
                }
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            rate_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max_rate"],
                rate_const,
            )
            assert cur_group is not None
            cur_group["pathways"].append(
                MassSpecInputFragmentationPathway(
                    cluster, product1, product2, rate_hist
                )
            )
            cur_group["product_labels"].append(
                f"{name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
            cur_group["pathway_ids"].append(pathway_id)
            last_cluster_id = cluster_id
        if cur_group is not None:
            groups.append(cur_group)

        mass_spec_table = ProgressTable(
            default_column_alignment="right", refresh_rate=0
        )
        mass_spec_table.add_column("Cluster", width=16, alignment="left")
        mass_spec_table.add_column("Paths", width=5, alignment="left")
        mass_spec_table.add_column("Frags", width=5)
        mass_spec_table.add_columns(
            "Intacts",
            "Avg colls",
        )
        mass_spec_table.add_column("PH rej", width=6)
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            groups,
            description="Cluster",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )
        for group in outer_pbar:
            mass_spec_table["Cluster"] = group["cluster_label"]
            mass_spec_table["Paths"] = str(len(group["pathways"]))
            start = timer()
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )
            subs = MassSpecSubstanceInput(
                group["cluster"],
                group["pathways"],
                config["gas"],
                group["density_hist"],
                config.get("cluster_charge_sign", 1),
            )
            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            def update_from_counters(counters):
                fragmented_total = counter_fragmented_total(counters)
                realizations = fragmented_total + counters.n_escaped_total
                mass_spec_table["Frags"] = fragmented_total
                mass_spec_table["Intacts"] = int(counters.n_escaped_total)
                mass_spec_table["Avg colls"] = counters.ncoll_total / realizations
                mass_spec_table["PH rej"] = int(counters.counter_collision_rejections)
                mass_spec_table["Surv prob"] = counters.n_escaped_total / realizations
                mass_spec_table["Warns"] = int(counters.nwarnings)
                inner_pbar.set_step(realizations)

            counters = self.run_mass_spec(
                mass_spec,
                subs,
                realizations,
                cluster_id=group["cluster_id"],
                pathway_ids=group["pathway_ids"],
                sample_mode=2,
                loglevel=0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
                result_callback=update_from_counters,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def run_prepared_config(self, name=None, **kwargs):
        from pprint import pprint

        configs = list(self.db.iter_configs(name))
        for idx, row in enumerate(configs):
            print(f"# Running experiment config: {row.name} [{idx + 1}/{len(configs)}]")
            pprint(row.config)
            print()
            self.start_run(
                row.id, pathway_at_a_time=kwargs.get("pathway_at_a_time", False)
            )
            self.run_from_config(row.config, run_started=True, **kwargs)
            print()
