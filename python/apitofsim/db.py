# pyright: reportAttributeAccessIssue=false

import numpy
from collections import namedtuple
import pandas
import duckdb
from pint import get_application_registry
from apitofsim import ClusterData
from datetime import timedelta

from glob import glob
from os.path import dirname, isfile, basename, expanduser

from .api import (
    ApiTofError,
    ApiTofOverflowError,
    MeshMode,
    MassSpectrometer,
    MassSpecSubstanceInput,
)

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

    def pathways_query(self, parent=None, sort=False):
        pathway = self.db.table("pathway")
        if parent is not None:
            pathway = pathway.filter(
                duckdb.ColumnExpression("cluster_id ")
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
        value_names = ["atomic_mass", "charge", "electronic_energy", "rotational_temperatures", "vibrational_temperatures"]
        values = [atomic_mass, charge, electronic_energy, rotational_temperatures, vibrational_temperatures]
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
            for value_name, existing_value, new_value in zip(value_names, existing[1:], values):
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
                        allow_duplicates=True
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
                import_info
            ),
        ).fetchone()
        assert id is not None
        return True, id[0]

    def insert_pathway(self, parent_id, product1_id, product2_id):
        self.db.execute(
            "insert into pathway values (default, ?, ?, ?)",
            (parent_id, product1_id, product2_id),
        )


def ingest_tree(db: ClusterDatabase, pathways):
    if isinstance(pathways, list):
        for pathways_segment in pathways:
            ingest_tree(db, pathways_segment)
        return
    if pathways["type"] == "legacy_glob":
        from apitofsim.ingest.legacy import ingest_legacy_tree
        from apitofsim.config import dump_to_raw
        for pathway in ingest_legacy_tree(pathways["path"], pathways["clusters"]):
            ids = []
            for particle_info in pathway:
                name = particle_info["name"]
                if prefix := pathways.get("prefix"):
                    name = prefix + name
                combined = particle_info["particle"]
                with ureg.context("boltzmann", "spectroscopy"):
                    inserted, id = db.insert_cluster(
                        name,
                        combined["atomic_mass"].to("amu").magnitude,
                        combined["charge"],
                        combined["electronic_energy"].to("hartree").magnitude,
                        combined["rotational_temperatures"].to("K").magnitude,
                        combined["vibrational_temperatures"].to("K").magnitude,
                        dump_to_raw(particle_info).decode("utf-8"),
                        allow_duplicates=True,
                    )
                    ids.append(id)
            db.insert_pathway(*ids)


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

    def run_mass_spec(
        self,
        *args,
        pathway_id,
        strict=False,
        strict_dos=True,
        **kwargs,
    ):
        self._guard_run_started()
        from .api import mass_spec

        counters = None
        try:
            counters, timings = mass_spec(
                *args,
                **kwargs,
                output_named_tuple=True,
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
        return counters

    def run_from_config(self, config, run_started=False, strict_dos=True):
        from os import environ
        from apitofsim import (
            ProductsCluster,
            KTotalInput,
            compute_density_of_states_batch,
            compute_k_total_batch,
            precompute_mesh,
            FragmentationPathway,
        )
        from apitofsim.api import Histogram, validate_max_energies
        from timeit import default_timer as timer
        from progress_table import ProgressTable

        if not run_started:
            self.db.start_run()

        cluster_indexed, name_lookup = self.db.clusters_objects_indexed(
            include_name_lookup=True
        )

        prelim_table = ProgressTable(default_column_alignment="left")
        prelim_table.add_column("#", alignment="right", width=3)
        prelim_table.add_column("Step", width=20)
        prelim_table.add_column("Description", width=40)
        prelim_table.add_column("Time (s)", alignment="right")
        prelim_table["#"] = "1/4"
        prelim_table["Step"] = "Skimmer"
        start = timer()
        skimmer_np = self._run_skimmer(config)
        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.next_row()

        prelim_table["#"] = "2/4"
        prelim_table["Step"] = "Density of states"
        start = timer()
        num_pathways = 0
        density_of_states_inputs = []
        for _, cluster, _, _ in self.db.pathways_objs(indexed=cluster_indexed):
            density_of_states_inputs.append(cluster)
            num_pathways += 1

        for _, _, product1, product2 in self.db.pathways_objs(indexed=cluster_indexed):
            density_of_states_inputs.append(ProductsCluster(product1, product2))

        prelim_table["Description"] = f"{len(density_of_states_inputs)} inputs"

        density_of_states = compute_density_of_states_batch(
            density_of_states_inputs,
            energy_max=config["energy_max"],
            bin_width=config["bin_width"],
        )
        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.next_row()

        prelim_table["#"] = "3/4"
        prelim_table["Step"] = "Computing mesh"
        start = timer()
        cluster_dos = density_of_states[:, :num_pathways]
        product_dos = density_of_states[:, num_pathways:]

        k_total_inputs = []
        for idx, (_, cluster, product1, product2) in enumerate(
            self.db.pathways_objs(indexed=cluster_indexed)
        ):
            fragmentation_energy = FragmentationPathway(
                cluster.into_cpp(), product1.into_cpp(), product2.into_cpp()
            ).fragmentation_energy_kelvin()
            validate_max_energies(
                fragmentation_energy=fragmentation_energy,
                energy_max_rate=config["energy_max_rate"],
                energy_max=config["energy_max"],
                bin_width=config["bin_width"],
                quantities_strict=False,
            )

            k_total_inputs.append(
                KTotalInput(
                    product1.into_cpp(),
                    product2.into_cpp(),
                    fragmentation_energy,
                    cluster_dos[:, idx],
                    product_dos[:, idx],
                )
            )

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
        prelim_table["Description"] = f"{len(k_total_inputs)} inputs"

        start = timer()
        k_rates = compute_k_total_batch(
            k_total_inputs,
            energy_max_rate=config["energy_max_rate"],
            bin_width=config["bin_width"],
            mesh=mesh,
        )
        prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
        prelim_table.close()
        del prelim_table

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

        mass_spec_table = ProgressTable(default_column_alignment="right")
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
            "Surv. prob.",
            "Time (s)",
        )
        pathway_ids = list(self.db.pathways_ids(sort=True))
        cluster_seq = 1
        for (
            pathway_id,
            cluster_id,
            product1_id,
            product2_id,
        ), rate_const, density_cluster in zip(
            pathway_ids,
            k_rates.T,
            cluster_dos.T,
        ):
            cluster = cluster_indexed[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            mass_spec_table["#"] = f"{cluster_seq}/{len(pathway_ids)}"
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
                cluster_charge_sign=config.get("cluster_charge_sign", -1),
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
            mass_spec_table.next_row()
            cluster_seq += 1

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

    def run_prepared_config(self, name=None, **kwargs):
        from pprint import pprint

        configs = list(self.db.iter_configs(name))
        for idx, row in enumerate(configs):
            print(f"# Running experiment config: {row.name} [{idx + 1}/{len(configs)}]")
            pprint(row.config)
            print()
            self.start_run(row.id)
            self.run_from_config(row.config, run_started=True, **kwargs)
            print()
