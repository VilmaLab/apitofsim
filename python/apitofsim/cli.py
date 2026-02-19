import argparse
import sys
import os
import numpy
from apitofsim.api import (
    skimmer,
    densityandrate,
    mass_spec,
    defaults,
)
from apitofsim.config import (
    read_histogram,
    read_skimmer,
    parse_config_with_particles,
    ConfigFile,
    get_clusters,
)
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "command",
    type=click.Choice(["skimmer", "densityandrate", "mass_spec"], case_sensitive=False),
    required=False,
)
@click.argument("config", required=True)
@click.option(
    "-C", "--chdir", default=1, help="Change the working directory before executing"
)
def legacy(command, config, chdir):
    """
    Run COMMAND with configuration at path CONFIG.
    """
    if chdir:
        try:
            os.chdir(chdir)
        except FileNotFoundError:
            print(f"Error: The directory {chdir} does not exist.", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(
                f"Error: Permission denied to change to directory {chdir}.",
                file=sys.stderr,
            )
            sys.exit(1)

    config = ConfigFile(filename=config)

    if command == "skimmer" or command is None:
        skimmer_df = skimmer(
            config.get("T", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("pressure_first", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("Lsk", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("dc", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("alpha_factor", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("gas", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("N_iter", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("M_iter", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("resolution", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("tolerance", by="short_name"),  # pyright: ignore [reportArgumentType]
            output_pandas=True,
        )
        print(skimmer_df)
    if command == "densityandrate" or command is None:
        clusters = get_clusters(parse_config_with_particles(config))
        rhos, k_rate = densityandrate(
            *clusters,
            config.get("energy_max", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("energy_max_rate", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("bin_width", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("bonding_energy", by="short_name"),  # pyright: ignore [reportArgumentType]
        )
        numpy.set_printoptions(threshold=sys.maxsize)
    if command == "mass_spec" or command is None:
        config_dict = parse_config_with_particles(config)
        clusters = get_clusters(config_dict)
        density_cluster = read_histogram(
            config_dict["config"]["output_file_density_cluster"]
        )
        rate_constant = read_histogram(
            config_dict["config"]["output_file_rate_constant"]
        )
        skimmer_data = read_skimmer(config_dict["config"]["Output_file_skimmer"])
        if skimmer_data is None:
            raise ValueError("Skimmer file is empty")
        skimmer_data, mesh_skimmer = skimmer_data

        def log_callback(type, message):
            # print(type, message, end="")
            pass

        def result_callback(counters):
            print(counters)

        counters = mass_spec(
            *clusters,
            config.get("gas"),  # pyright: ignore [reportArgumentType]
            density_cluster,  # pyright: ignore [reportArgumentType]
            rate_constant,  # pyright: ignore [reportArgumentType]
            skimmer_data,
            config.get("lengths"),  # pyright: ignore [reportArgumentType]
            config.get("voltages"),  # pyright: ignore [reportArgumentType]
            config.get("T"),  # pyright: ignore [reportArgumentType]
            config.get("pressure_first"),  # pyright: ignore [reportArgumentType]
            config.get("pressure_second"),  # pyright: ignore [reportArgumentType]
            config.get("N"),  # pyright: ignore [reportArgumentType]
            mesh_skimmer=mesh_skimmer,
            quadrupole=config.get("quadrupole"),  # pyright: ignore [reportArgumentType]
            fragmentation_energy=config.get("bonding_energy") or None,  # pyright: ignore [reportArgumentType]
            cluster_charge_sign=config.get("cluster_charge_sign") or defaults.cluster_charge_sign,  # pyright: ignore [reportArgumentType]
            seed=42,
            log_callback=None,
            result_callback=result_callback,
        )
        print("Final counters:", counters)


@cli.group()
def db():
    """
    Work with the database-backed workflow
    """
    pass


@db.command()
@click.argument("config", required=True)
@click.argument("database", required=True)
@click.option(
    "-t",
    "--db-type",
    type=click.Choice(["experiment", "cluster", "super"]),
    default="experiment",
    help="The type of database to create: this will determine which tables are created",
)
@click.option("-w", "--warm", is_flag=True, help="Warm the database with histogrammed data")
def prepare(config, database, db_type, warm):
    """
    Prepare the database at path DATABASE using the json configuration file at path CONFIG.
    """
    import os
    from os import unlink
    import orjson
    from pprint import pprint

    from apitofsim.db import ingest_tree, ClusterDatabase, ExperimentDatabase, SuperClusterDatabase, DerivedDataPreparer
    from apitofsim.config import import_raw_config

    def iter_raw_configs(json):
        for config in json.get("configs", []):
            yield config["name"], {**json.get("default_config", {}), **config}

    if os.path.exists(database):
        unlink(database)
    if db_type == "experiment":
        db = ExperimentDatabase(database)
    elif db_type == "cluster":
        db = ClusterDatabase(database)
        if warm:
            raise click.UsageError("Warm option is not supported for cluster database")
    elif db_type == "super":
        db = SuperClusterDatabase(database)
    else:
        assert False

    db.create_tables()

    with open(config, "rb") as f:
        source = orjson.loads(f.read())

    ingest_tree(db, source["pathways"])

    if warm and "densityandrate_configs" in source:
        assert isinstance(db, SuperClusterDatabase)
        preparer = DerivedDataPreparer(db)
        for config in source["densityandrate_configs"]:
            print("Warming up density and rate for config:")
            pprint(config)
            cluster_indexed, _, pathway_lookup = db.get_all_lookups()
            preparer.run_densityandrate(import_raw_config(config), cluster_indexed, pathway_lookup)

    if db_type == "experiment":
        assert isinstance(db, ExperimentDatabase)
        for name, config in iter_raw_configs(source):
            db.insert_config(name, config)
            if warm:
                print("Warming up density and rate for config:", name)
                pprint(config)
                preparer = DerivedDataPreparer(db)
                cluster_indexed, _, pathway_lookup = db.get_all_lookups()
                preparer.run_densityandrate(import_raw_config(config), cluster_indexed, pathway_lookup)


@db.command()
@click.argument("database", required=True)
@click.option("--strict-dos/--no-strict-dos", default=False)
@click.option("--filter-parent", default=None)
@click.option("--filter-pathway", multiple=True, default=None)
@click.option("--filter-config", multiple=True, default=None)
@click.option("--pathway-at-a-time", default=False, is_flag=True)
@click.option("--verbose", default=False, is_flag=True)
def run(database, strict_dos, filter_parent, filter_pathway, filter_config, pathway_at_a_time, verbose):
    """
    Run simulation according to the configurations in DATABASE.
    """
    from apitofsim.db import ExperimentDatabase, ExperimentRunner

    db = ExperimentDatabase(database)
    num_configs = db.db.sql("select count(*) from duckdb_tables() where table_name = 'experiment_config'").fetchone()
    assert num_configs is not None
    if num_configs[0] == 0:
        raise click.UsageError("The specified database does not contain experiment_config. Did you create it as an experiment database?")
    runner = ExperimentRunner(db)
    runner.run_prepared_config(
        name=filter_config or None,
        strict_dos=strict_dos,
        pathway_at_a_time=pathway_at_a_time,
        parent=filter_parent,
        pathways=(pathway.split(",") for pathway in filter_pathway) if filter_pathway else None,
        verbose=verbose,
    )


if __name__ == "__main__":
    cli()
