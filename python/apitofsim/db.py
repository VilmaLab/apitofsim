# type: ignore

import pandas
import duckdb
from pint import get_application_registry
from apitofsim import ClusterData

from glob import glob
from os.path import dirname, isfile, basename, expanduser

ureg = get_application_registry()
Q_ = ureg.Quantity


class ClusterDatabase:
    TABLES = """
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
        cluster_id integer,
        product1_id integer,
        product2_id integer,
        foreign key (cluster_id) references cluster (id),
        foreign key (product1_id) references cluster (id),
        foreign key (product2_id) references cluster (id)
    );
    """

    def __init__(self, filename):
        self.db = duckdb.connect(filename)
        self.cluster_cache = {}

    def create_tables(self):
        self.db.execute(self.TABLES)

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
        return (
            query.join(
                self.db.from_df(relevant_cluster_ids).set_alias("relevant"),
                condition="relevant.relevant_cluster_id = cluster.id",
            )
            .fetchdf()
            .replace({pandas.NA: None})
        )

    def clusters_df(self, *args, **kwargs):
        return self.clusters_query(*args, **kwargs).fetchdf()

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
                pathway.cluster_id,
                pathway.product1_id,
                pathway.product2_id,
            )

    def pathways_objs(self, *args, indexed=None, **kwargs):
        if indexed is None:
            indexed = self.clusters_objects_indexed()
        for cluster_id, product1_id, product2_id in self.pathways_ids(*args, **kwargs):
            yield (
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


def fixup_config(config, particle, backup_dir=None):
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
                print(f"Could not find {config[config_key]}; skipping particle")
                return True
    return False


def ingest_legacy(db: ClusterDatabase, path, backup_dir=None):
    from contextlib import chdir
    from pprint import pprint
    from apitofsim.config import (
        parse_config,
        get_particle,
    )

    filenames = glob(expanduser(path), recursive=True)
    for filename in filenames:
        print("Reading", filename)
        with chdir(dirname(filename)):
            config = parse_config(filename)
            ids = []
            particle_failed = False
            for particle in ["cluster", "first_product", "second_product"]:
                if fixup_config(config, particle, backup_dir):
                    particle_failed = True
                    continue
                pprint(config)
                particle_config = get_particle(config, particle)
                inserted, id = db.insert_cluster(
                    particle_config["name"],
                    particle_config["atomic_mass"],
                    particle_config["electronic_energy"],
                    particle_config["rotational_temperatures"],
                    particle_config["vibrational_temperatures"],
                )
                ids.append(id)
                if not inserted:
                    print("Skipping existing particle", particle_config["name"])
            if particle_failed:
                print("Skipping pathway due to missing particles")
                continue
            db.insert_pathway(*ids)
