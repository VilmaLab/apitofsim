# pyright: reportAttributeAccessIssue=false

from collections import namedtuple
import pandas
import duckdb
from pint import get_application_registry
from apitofsim import ClusterData
from datetime import timedelta

from glob import glob
from os.path import dirname, isfile, basename, expanduser

from .api import ApiTofError, ApiTofOverflowError, MeshMode

ureg = get_application_registry()
Q_ = ureg.Quantity

PATHWAY_TABLES = """
create sequence cluster_id_sequence start 1;
create sequence pathway_id_sequence start 1;

create table cluster (
    id integer default nextval('cluster_id_sequence') primary key,
    common_name varchar,
    atomic_mass integer,
    electronic_energy double,
    rotational_temperatures double[3],
    vibrational_temperatures double[],
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

    def __init__(self, filename):
        self.db = duckdb.connect(filename)

    def create_tables(self):
        self.db.execute("\n".join(self.TABLES))

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

    def pathways_query(self, parent=None):
        pathway = self.db.table("pathway")
        if parent is not None:
            pathway = pathway.filter(
                duckdb.ColumnExpression("cluster_id ")
                == duckdb.ConstantExpression(parent)
            )
        return pathway

    def pathways_ids(self, parent=None):
        query = self.pathways_query(parent)
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
        electronic_energy,
        rotational_temperatures,
        vibrational_temperatures,
    ):
        existing_id = self.db.execute(
            "select id from cluster where common_name = ?",
            (name,),
        ).fetchone()
        if existing_id is not None:
            return False, existing_id[0]
        id = self.db.execute(
            "insert into cluster values (default, ?, ?, ?, ?, ?) returning id",
            (
                name,
                atomic_mass,
                electronic_energy,
                rotational_temperatures,
                vibrational_temperatures,
            ),
        ).fetchone()
        assert id is not None
        return True, id[0]

    def insert_pathway(self, parent_id, product1_id, product2_id):
        self.db.execute(
            "insert into pathway values (default, ?, ?, ?)",
            (parent_id, product1_id, product2_id),
        )


def backup_search(source, data_file):
    if "backup_search" in source:
        results = glob(
            source["backup_search"] + "/**/" + basename(data_file),
            recursive=True,
        )
        if len(results) == 1:
            return results[0]


def fixup_config(config, particle, backup_dir=None, allow_fail=False):
    for quantity in [
        "vibrational_temperatures",
        "rotational_temperatures",
        "electronic_energy",
    ]:
        config_key = f"file_{quantity}_{particle}"
        data_file = config[config_key]
        if not isfile(data_file):
            particle_failed = True
            if backup_dir is not None:
                result = backup_search(backup_dir, data_file)
                if result is not None:
                    config[config_key] = result
                    particle_failed = False
            if particle_failed:
                msg = f"Could not find {config_key}='{config[config_key]}'"
                if not allow_fail:
                    raise RuntimeError(msg)
                print(msg + "; skipping particle")
                return True
    return False


def ingest_legacy_one(
    db: ClusterDatabase,
    filename,
    backup_dir=None,
    verbose=False,
    allow_fail=False,
    allow_skip=False,
):
    from contextlib import chdir
    from apitofsim.config import (
        parse_config,
        get_particle,
    )

    with chdir(dirname(filename)):
        config = parse_config(filename)
        ids = []
        particle_failed = False
        for particle in ["cluster", "first_product", "second_product"]:
            if fixup_config(config, particle, backup_dir, allow_fail):
                particle_failed = True
                continue
            particle_config = get_particle(config, particle)
            name = particle_config["name"]
            inserted, id = db.insert_cluster(
                name,
                particle_config["atomic_mass"],
                particle_config["electronic_energy"],
                particle_config["rotational_temperatures"],
                particle_config["vibrational_temperatures"],
            )
            ids.append(id)
            if not inserted:
                if not allow_skip:
                    raise RuntimeError(f"Particle already exists: {name}")
                if verbose:
                    print(f"Skipping existing particle: {name}")
        if particle_failed:
            print("Skipping pathway due to missing particles")
            return
        db.insert_pathway(*ids)


def ingest_legacy_tree(db: ClusterDatabase, path, backup_dir=None, verbose=False):
    filenames = glob(expanduser(path), recursive=True)
    for filename in filenames:
        ingest_legacy_one(db, filename, backup_dir, verbose)


EXPERIMENT_TABLES = """
create sequence experiment_config_sequence start 1;
create sequence experiment_run_sequence start 1;
create sequence experiment_result_sequence start 1;

create table experiment_config (
    id integer default nextval('experiment_config_sequence') primary key,
    name varchar,
    config json
);

create table experiment_run (
    id integer default nextval('experiment_run_sequence') primary key,
    experiment_config_id integer not null,
    foreign key (experiment_config_id) references experiment_config (id),
    start_time timestamp
);

create table experiment_result (
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

create table experiment_failure (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    pathway_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    foreign key (pathway_id) references pathway (id),
    exc_name varchar,
    msg varchar,
    overflow_requested double
);
"""


REPORT_VIEW = """
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
    select * from experiment_result
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


class ExperimentDatabase(ClusterDatabase):
    TABLES = [PATHWAY_TABLES, EXPERIMENT_TABLES, REPORT_VIEW]

    def __init__(self, filename):
        super().__init__(filename)

    def refresh_views(self):
        self.db.execute(REPORT_VIEW)

    def insert_run(self, config_id=None):
        id = self.db.execute(
            "insert into experiment_run values (default, ?, current_timestamp) returning id",
            (config_id,),
        ).fetchone()
        assert id is not None
        return id[0]

    def insert_config(self, name, config):
        from .config import dump_to_raw

        if isinstance(config, dict):
            import orjson

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
            query = query.filter(
                duckdb.ColumnExpression("name") == duckdb.ConstantExpression(name)
            )
        for id, name, config in query.fetchall():
            yield ConfigRow(id, name, import_raw_config(orjson.loads(config)))

    def record_result(
        self,
        run_id,
        pathway_id,
        counters,
        timings,
    ):
        id = self.db.execute(
            "insert into experiment_result values (default, ?, ?, ?, ?, ?, ?, ?, ?, ?) returning id",
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
        assert id is not None
        return id[0]

    def record_failure(
        self, run_id, pathway_id, exc_name, msg, overflow_requested=None
    ):
        id = self.db.execute(
            "insert into experiment_failure values (default, ?, ?, ?, ?, ?) returning id",
            (run_id, pathway_id, exc_name, msg, overflow_requested),
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


class ExperimentRunner:
    def __init__(self, db: ExperimentDatabase):
        self.db = db
        self.current_run_id = None

    def _guard_run_started(self):
        if self.current_run_id is None:
            raise RuntimeError("No experiment run started; call start_run() first")

    def start_run(self, config_id=None):
        self.current_run_id = self.db.insert_run(config_id)

    def run_pinhole(
        self,
        *args,
        pathway_id,
        strict=False,
        **kwargs,
    ):
        self._guard_run_started()
        from .api import pinhole

        try:
            counters, timings = pinhole(
                *args,
                **kwargs,
                output_named_tuple=True,
                output_timings=True,
            )
        except ApiTofError as e:
            if strict:
                raise
            overflow_requested = None
            if isinstance(e, ApiTofOverflowError):
                overflow_requested = e.current
            self.db.record_failure(
                self.current_run_id,
                pathway_id,
                type(e).__name__,
                str(e),
                overflow_requested,
            )
        else:
            self.db.record_result(
                self.current_run_id,
                pathway_id,
                counters,
                timings,
            )

    def run_from_config(self, config, run_started=False):
        from os import environ
        from apitofsim import (
            skimmer,
            ProductsCluster,
            KTotalInput,
            compute_density_of_states_batch,
            compute_k_total_batch,
            precompute_mesh,
            FragmentationPathway,
        )
        from apitofsim.api import Histogram
        from timeit import default_timer as timer

        if not run_started:
            self.db.start_run()

        cluster_indexed, name_lookup = self.db.clusters_objects_indexed(
            include_name_lookup=True
        )
        print()

        print("Running skimmer")
        skimmer_np = skimmer(
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
        print("Done")

        num_pathways = 0
        density_of_states_inputs = []
        for _, cluster, _, _ in self.db.pathways_objs(indexed=cluster_indexed):
            density_of_states_inputs.append(cluster)
            num_pathways += 1

        for _, _, product1, product2 in self.db.pathways_objs(indexed=cluster_indexed):
            density_of_states_inputs.append(ProductsCluster(product1, product2))

        print("Computing density of states")
        start = timer()

        density_of_states = compute_density_of_states_batch(
            density_of_states_inputs,
            energy_max=config["energy_max"],
            bin_width=config["bin_width"],
        )
        print(f"Done in {timer() - start}s")

        cluster_dos = density_of_states[:, :num_pathways]
        product_dos = density_of_states[:, num_pathways:]

        k_total_inputs = []
        for idx, (_, cluster, product1, product2) in enumerate(
            self.db.pathways_objs(indexed=cluster_indexed)
        ):
            k_total_inputs.append(
                KTotalInput(
                    product1.into_cpp(),
                    product2.into_cpp(),
                    FragmentationPathway(
                        cluster.into_cpp(), product1.into_cpp(), product2.into_cpp()
                    ).fragmentation_energy_kelvin(),
                    cluster_dos[:, idx],
                    product_dos[:, idx],
                )
            )

        mesh_points = config["energy_max_rate"] / config["bin_width"]
        print(
            f"Compute mesh with {mesh_points} points and mesh_mode=MeshMode.compute_mesh_diagonal_multithreaded"
        )

        start = timer()
        mesh = precompute_mesh(
            energy_max_rate=config["energy_max_rate"],
            bin_width=config["bin_width"],
            mesh_mode=MeshMode.compute_mesh_diagonal_multithreaded,
        )
        print(f"Done in {timer() - start}")

        print(f"Compute k total on {len(k_total_inputs)} inputs")

        start = timer()
        k_rates = compute_k_total_batch(
            k_total_inputs,
            energy_max_rate=config["energy_max_rate"],
            bin_width=config["bin_width"],
            mesh=mesh,
        )
        print(f"Done in {timer() - start}")

        for (
            pathway_id,
            cluster_id,
            product1_id,
            product2_id,
        ), rate_const, density_cluster in zip(
            self.db.pathways_ids(),
            k_rates.T,
            cluster_dos.T,
        ):
            cluster = cluster_indexed[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            print(
                f"{name_lookup[cluster_id]} -> {name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
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
            self.run_pinhole(
                cluster,
                product1,
                product2,
                config["gas"],
                density_hist,
                rate_hist,
                skimmer_np,
                config["lengths"],
                config["voltages"],
                config["T"],
                config["pressure_first"],
                config["pressure_second"],
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"],
                quadrupole=config.get("quadrupole"),
                fragmentation_energy=config.get("fragmentation_energy"),
                cluster_charge_sign=config.get("cluster_charge_sign", -1),
                pathway_id=pathway_id,
                sample_mode=2,
                loglevel=0,
                strict="STRICT" in environ,
            )

    def run_prepared_config(self, name=None):
        for row in self.db.iter_configs(name):
            print("Running experiment config:", row.name)
            print(row.config)
            self.start_run(row.id)
            self.run_from_config(row.config, run_started=True)
