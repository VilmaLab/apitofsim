import os
import pathlib
import sys

import click
import numpy

from apitofsim.api import (
    defaults,
    densityandrate,
    mass_spec,
    skimmer,
)
from apitofsim.config import (
    ConfigFile,
    get_clusters,
    parse_config_with_particles,
    read_histogram,
    read_skimmer,
)


@click.group()
def cli():
    """
    The command line interface to apitofsim
    """
    pass


@cli.command(short_help="Partial re-implementation of legacy commmands (please ignore)")
@click.argument(
    "command",
    type=click.Choice(["skimmer", "densityandrate", "mass_spec"], case_sensitive=False),
    required=False,
)
@click.argument("config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-C",
    "--chdir",
    help="Change the working directory before executing",
    type=click.Path(exists=True, file_okay=False),
)
def legacy(command, config, chdir):
    """
    This is a partial re-implementation of the old CLI, however it does not currently implement the full functionality of the old CLI.
    For now, stick to `apitofsim-skimmer`, `apitofsim-densityandrate`, and `apitofsim-mass-spec` for the old CLI functionality.

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
            cluster_charge_sign=config.get("cluster_charge_sign")
            or defaults.cluster_charge_sign,  # pyright: ignore [reportArgumentType]
            seed=42,
            log_callback=None,
            result_callback=result_callback,
        )
        print("Final counters:", counters)


@cli.group(short_help="Commands to work with the database-backed workflow")
def db():
    """
    Commands to work with the database-backed workflow.

    Typically you will `prepare` your dataset into a database, then `run` it, and then use `report` and `plot` to analyze the results.
    """
    pass


@db.command(short_help="Prepare the database for use")
@click.argument("mode", required=True, type=click.Choice(["create", "append"]))
@click.argument("config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("database", required=True, type=click.Path(dir_okay=False))
@click.option(
    "-t",
    "--db-type",
    type=click.Choice(["experiment", "cluster", "super"]),
    default="experiment",
    help="The type of database to create: this will determine which tables are created",
)
@click.option(
    "-w", "--warm", is_flag=True, help="Warm the database with histogrammed data"
)
def prepare(mode, config, database, append, replace_by_name, db_type, warm):
    """
    Prepare the database according to MODE at path DATABASE using the json configuration file at path CONFIG.

    Typically, you will be creating an database in order to run simulation experiments,
    and so --db-type should be kept as --db-type=experiment.

    MODE can be create or append, depending on whether you want to create a new database or append to an existing one.

    You may choose to `--warm` your database, so that histogramming is done now, in advance of the simulation itself.

    The CONFIG file specifies both where to import the information about the cluster and pathways from, and the parameters to use for the simulation.

    **Pathways**

    For pathways, currently only the "legacy_glob" type is supported, which imports pathways from the legacy .in files matching a glob pattern.
    The per-cluster data is then taken either from .dat files specified in the .in file or ORCA or Guassian outputs files next to the .in file.

    ```toml
    [[pathways]]
    type = "legacy_glob"
    # An optional prefix to add to the common names of all clusters imported here
    prefix = ""
    # A glob pattern to use to find .in files. The wildcard * matches any number of characters, while ** can span directories.
    path = "/path/to/**/*.in"
    # Paths specified in the .in files are relative to the .in file directory.
    cwd = "."

    [pathways.clusters]
    # By default, all 
    default_source = "gaussian"
    electronic_energy = "orca.final_single_point_energy + gaussian.zero_point_energy"

    # Here the sources for cluster information are specified.
    [pathways.clusters.sources.dat]

    [pathways.clusters.sources.orca]
    append_to_common_prefix = ".out"

    [pathways.clusters.sources.gaussian]
    append_to_common_prefix = ".log"
    ```

    **Parameters**

    For the simulation parameters, you can specify one or more `[[configs]]` sections, each with a `name` field and the values of all parameters.
    You can put common parameters in the `default_config` section, so that each `[[configs]]` section inherits these as overridable defaults.

    ```toml
    [default_config]
    M_iter = 1_000
    N = 1_000
    N_iter = 1_000
    T = "300.0 kelvin"
    alpha_factor = "0.25 halfturn"
    bin_width = "1.0 kelvin"
    dc = "0.0005 meter"
    energy_max = "2.0e5 kelvin"
    energy_max_rate = "1.0e5 kelvin"
    lengths = [ [ 0.001, 0.00244, 0.101, 0.00448, 0.0005 ], "meter" ]
    pressure_first = "194.0 pascal"
    pressure_second = "3.88 pascal"
    resolution = 1_000
    tolerance = 1e-8
    voltages = [ [ -19, -9, -7, -6, 11 ], "volt" ]

    [default_config.gas]
    radius = "1.84e-10 meter"
    mass = "4.65e-26 kilogram"
    adiabatic_index = 1.4

    [[configs]]
    name = "simple"

    [[configs]]
    name = "with-quadrupole-and-pinhole"
    radius_pinhole = "1 mm"

    [configs.quadrupole]
    dc_field = "0.0 volt"
    ac_field = "200.0 volt"
    radiofrequency = "1.3e6 Hz"
    r_quadrupole = "6.0e-3 meter"
    ```
    """
    import os
    from pprint import pprint

    import tomllib

    from apitofsim.config import import_raw_config
    from apitofsim.workflow import (
        ClusterDatabase,
        DerivedDataPreparer,
        ExperimentDatabase,
        SuperClusterDatabase,
        ingest_tree,
    )

    def iter_raw_configs(json):
        for config in json.get("configs", []):
            yield config["name"], {**json.get("default_config", {}), **config}

    if os.path.exists(database) and mode == "create":
        raise click.UsageError(f"Database file {database} already exists, will not overwrite (delete it yourself first if you want)")
    if not os.path.exists(database) and mode != "create":
        raise click.UsageError(
            f"Database file {database} does not exist, cannot append"
        )

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

    if mode == "create":
        db.create_tables()

    with open(config, "rb") as f:
        source = tomllib.load(f)

    ingest_tree(db, source["pathways"])

    if warm and "densityandrate_configs" in source:
        assert isinstance(db, SuperClusterDatabase)
        preparer = DerivedDataPreparer(db)
        for config in source["densityandrate_configs"]:
            print("Warming up density and rate for config:")
            pprint(config)
            cluster_indexed, _, pathway_lookup = db.get_all_lookups()
            preparer.run_densityandrate(
                import_raw_config(config), cluster_indexed, pathway_lookup
            )

    if db_type == "experiment":
        assert isinstance(db, ExperimentDatabase)
        for name, config in iter_raw_configs(source):
            db.insert_config(name, config)
            if warm:
                print("Warming up density and rate for config:", name)
                pprint(config)
                preparer = DerivedDataPreparer(db)
                cluster_indexed, _, pathway_lookup = db.get_all_lookups()
                preparer.run_densityandrate(
                    import_raw_config(config), cluster_indexed, pathway_lookup
                )


@db.command(short_help="Run the simulations according to the prepared database")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--strict-dos/--no-strict-dos", default=False, help="Whether to fail early when particle energy go above the max energy the DOS is histogrammed for")
@click.option("--filter-parent", default=None, help="Only run pathways with a specified common name for the parent cluster")
@click.option("--filter-pathway", multiple=True, default=None, help="Only run the pathway specified using common names as 'PARENT,CHILD,CHILD'")
@click.option("--filter-config", multiple=True, default=None, help="Only run the experiment the parameters in the named configuration")
@click.option("--pathway-at-a-time", default=False, is_flag=True, help="Run one pathway at a time")
@click.option("--verbose", default=False, is_flag=True)
def run(
    database,
    strict_dos,
    filter_parent,
    filter_pathway,
    filter_config,
    pathway_at_a_time,
    verbose,
):
    """
    Run simulation according to the configurations in DATABASE.
    """
    from apitofsim.workflow import ExperimentDatabase, ExperimentRunner

    db = ExperimentDatabase(database)
    num_configs = db.db.sql(
        "select count(*) from duckdb_tables() where table_name = 'experiment_config'"
    ).fetchone()
    assert num_configs is not None
    if num_configs[0] == 0:
        raise click.UsageError(
            "The specified database does not contain experiment_config. Did you create it as an experiment database?"
        )
    runner = ExperimentRunner(db)
    runner.run_prepared_config(
        name=filter_config or None,
        strict_dos=strict_dos,
        pathway_at_a_time=pathway_at_a_time,
        parent=filter_parent,
        pathways=(pathway.split(",") for pathway in filter_pathway)
        if filter_pathway
        else None,
        verbose=verbose,
    )


@db.group(help="Commands for plotting results")
def plot():
    """
    Commands for plotting results.
    """
    pass


def select_experiment(db):
    import questionary

    df = db.report_df("experiment_summary")
    choices = []
    for row in df.itertuples():
        pathway_desc = "single pathway" if row.is_single_pathway else "multi-pathway"
        choices.append(
            questionary.Choice(
                f"#{row.experiment_run_id} {row.config_name} run at {row.start_time} "
                f"({pathway_desc}, success rate: {row.successes}/{row.successes + row.failures})",
                value=row.experiment_run_id,
            )
        )
    return questionary.prompt(
        {
            "type": "select",
            "name": "experiment",
            "message": "Select a experiment to plot survival rates from:",
            "choices": choices,
        },
        use_jk_keys=False,
        use_search_filter=True,
    )["experiment"]


def select_cluster_result(db):
    import questionary

    df = db.report_df("experiment_cluster_report")
    print(df)
    choices = []
    for row in df.itertuples():
        pathway_desc = "single pathway" if row.is_single_pathway else "multi-pathway"
        choices.append(
            questionary.Choice(
                f"#{row.experiment_run_id} {row.config_name} run at {row.start_time}"
                f" with cluster {row.cluster_common_name} mass {row.cluster_atomic_mass} electronic energy {row.cluster_electronic_energy}"
                f" ({pathway_desc})",
                value=(row.experiment_run_id, row.cluster_id, row.is_single_pathway),
            )
        )
    return questionary.prompt(
        {
            "type": "select",
            "name": "cluster_result",
            "message": "Select a experiment to plot survival rates from:",
            "choices": choices,
        },
        use_jk_keys=False,
        use_search_filter=True,
    )["cluster_result"]


def get_joint_survivals(db, er_id):
    from functools import reduce
    from operator import mul

    import duckdb

    joint_survivals = {}

    for cluster in db.clusters_df(parents_only=True).itertuples():
        print("#", cluster.common_name)
        df = (
            db.db.table("experiment_report")
            .filter(
                (
                    duckdb.ColumnExpression("experiment_run_id")
                    == duckdb.ConstantExpression(er_id)
                )
                & (
                    duckdb.ColumnExpression("cluster_id")
                    == duckdb.ConstantExpression(cluster.id)
                )
            )
            .select(
                duckdb.SQLExpression(
                    "format('{} -> {} + {}', cluster_common_name, product1_common_name, product2_common_name)"
                ).alias("pathway_name"),
                *(
                    duckdb.ColumnExpression(col)
                    for col in [
                        "outcome_type",
                        "failure_msg",
                        "nwarnings",
                        "n_fragmented_total",
                        "n_escaped_total",
                        "ncoll_total",
                        "counter_collision_rejections",
                    ]
                ),
                duckdb.SQLExpression(
                    "n_escaped_total / (n_escaped_total + n_fragmented_total)"
                ).alias("survival_rate"),
            )
        ).fetchdf()
        print(df)
        print()
        if len(df) == 0:
            print("No results for", cluster.common_name)
            continue
        # This seems a bit naughty
        survival_rate = df["survival_rate"][df["survival_rate"] > 0]
        joint_survivals[cluster.common_name] = reduce(mul, survival_rate, 1.0)
    return joint_survivals


def make_survival_plot(outf, cluster_names, values):
    import matplotlib.pyplot as plt
    import numpy as np

    # Bar positions
    x = np.arange(len(cluster_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    ax.bar(
        x,
        values,
        width,
        edgecolor="none",
    )

    # Customize axes
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xticklabels(cluster_names)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Add horizontal gridlines only
    ax.yaxis.grid(True, linestyle="-", alpha=0.7, color="gray")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Style spines
    ax.spines["left"].set_color("gray")
    ax.spines["right"].set_color("gray")
    ax.spines["top"].set_color("gray")
    ax.spines["bottom"].set_color("gray")

    # Legend in upper right
    fig.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")

    plt.tight_layout()
    plt.savefig(outf, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")


def get_intensities_multipathway(db, er_id, cluster_id=None):
    intensities_sql = """
    with
        pathway_products as (
            select id as pathway_id, product1_id as product_id from pathway
            union
            select id as pathway_id, product2_id as product_id from pathway
        ),
        cluster_counts as (
            select
                multi_pathway_experiment_result.experiment_run_id as experiment_run_id,
                multi_pathway_experiment_result.cluster_id as parent_id,
                multi_pathway_experiment_result.id as experiment_result_id,
                pathway_products.product_id as product_id,
                pathway_fragmentation.count as count
            from
                multi_pathway_experiment_result
            inner join
                pathway_fragmentation on multi_pathway_experiment_result.id = pathway_fragmentation.experiment_result_id
            inner join
                pathway_products on pathway_products.pathway_id = pathway_fragmentation.pathway_id
            union
            select
                multi_pathway_experiment_result.experiment_run_id as experiment_run_id,
                multi_pathway_experiment_result.cluster_id as parent_id,
                multi_pathway_experiment_result.id as experiment_result_id,
                multi_pathway_experiment_result.cluster_id as product_id,
                multi_pathway_experiment_result.n_escaped_total as count
            from
                multi_pathway_experiment_result
        ),
        experiment_counts as (
            select
                cluster_counts.parent_id as parent_id,
                cluster_counts.experiment_result_id as experiment_result_id,
                sum(cluster_counts.count) as count
            from
                cluster_counts
            group by
                parent_id,
                experiment_result_id
        )
    select
        parent_cluster.common_name as parent_name,
        product_cluster.common_name as product_name,
        product_cluster.atomic_mass,
        cluster_counts.count / experiment_counts.count as relative_count,
        abs(relative_count * product_cluster.charge) as intensity
    from
        cluster_counts
    inner join
        cluster as parent_cluster
        on parent_cluster.id = cluster_counts.parent_id
    inner join
        cluster as product_cluster
        on product_cluster.id = cluster_counts.product_id
    inner join
        experiment_counts
        on experiment_counts.parent_id = cluster_counts.parent_id
        and experiment_counts.experiment_result_id = cluster_counts.experiment_result_id
    """
    if cluster_id is None:
        return db.db.execute(
            intensities_sql
            + """
        where
            cluster_counts.experiment_run_id = ?
        order by
            cluster_counts.parent_id
        """,
            (er_id,),
        ).fetchdf()
    else:
        return db.db.execute(
            intensities_sql
            + """
        where
            cluster_counts.experiment_run_id = ? and
            cluster_counts.parent_id = ?
        """,
            (er_id, cluster_id),
        ).fetchdf()


@plot.command(short_help="Plot survival rates for each parent cluster in an experiment")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("pngout", type=click.Path(dir_okay=False))
def survival(database, pngout):
    """
    Output to PNGOUT a bar chart of the survival rate for each cluster in the database at path DATABASE.
    """
    from pprint import pprint

    from apitofsim.workflow import ExperimentDatabase

    db = ExperimentDatabase(database, readonly=True)
    experiment_id = select_experiment(db)
    joint_survivals = get_joint_survivals(db, experiment_id)
    pprint(joint_survivals)
    make_survival_plot(pngout, joint_survivals.keys(), joint_survivals.values())


def plot_spectrogram(outf, df):
    import holoviews

    holoviews.extension("matplotlib")  # type: ignore

    spectrogram = holoviews.Spikes(
        (df["atomic_mass"], df["intensity"]),
        holoviews.Dimension("m/z", soft_range=(0, 1)),
        "Intensity",
    ).opts(fig_inches=(6, 3), aspect=2)
    matplotlib_fig = holoviews.render(spectrogram)
    matplotlib_fig.savefig(outf, dpi=300)


@plot.command(short_help="Plot a spectrogram for a single cluster in an experiment")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("pngout", type=click.Path(dir_okay=False))
def spectrogram(database, pngout):
    """
    Output to PNGOUT a spectrogram of the results for single cluster / experiment using the database at path DATABASE.
    """
    from apitofsim.workflow import ExperimentDatabase

    db = ExperimentDatabase(database, readonly=True)
    experiment_id, cluster_id, is_single_pathway = select_cluster_result(db)
    if is_single_pathway:
        raise click.UsageError("TODO: Single pathway experiments not supported yet")
    df = get_intensities_multipathway(db, experiment_id, cluster_id)
    plot_spectrogram(pngout, df)


@plot.command(short_help="Plot a spectrogram for each cluster in an experiment")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("dirout", type=click.Path(file_okay=False, path_type=pathlib.Path))
def spectrogram_many(database, dirout):
    """
    Output to DIROUT a spectrogram per cluster using the results from single experiments using the database at path DATABASE.
    """
    from apitofsim.workflow import ExperimentDatabase

    db = ExperimentDatabase(database, readonly=True)
    experiment_id = select_experiment(db)
    df = get_intensities_multipathway(db, experiment_id)
    dirout.mkdir(exist_ok=True)
    for parent_name, cluster_df in df.groupby("parent_name"):
        pngout = dirout / f"{parent_name}.png"
        plot_spectrogram(pngout, cluster_df)


@db.command(short_help="Produce an Excel-friendly CSV report from the database")
@click.argument(
    "report_type",
    type=click.Choice(
        ["pathway-report", "experiment-report", "experiment-summary"],
        case_sensitive=False,
    ),
    required=False,
)
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("csvout", type=click.Path(dir_okay=False))
def report(report_type, database, csvout):
    """
    Produce a report REPORT_TYPE from the database at path DATABASE and write it to CSV at path CSVOUT.

    * The pathway_report contains the input pathways giving one row per pathway, with no information about results.
    * The experiment_pathway_report contains one row per pathway / experiment run, and includes the outcome of that run for that pathway.
    * The experiment_cluster_report per parent cluster / experiment run, and includes the summarises information results from its pathways.
    * The experiment_summary contains one row per experiment run, and summarizes the outcomes across all pathways for that run.
    """
    from apitofsim.workflow import ExperimentDatabase

    db = ExperimentDatabase(database, readonly=True)
    db.db.table(report_type.replace("-", "_")).to_csv(csvout)


@db.command(help="Refresh views in the database at path DATABASE (please ignore)")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
def refresh_views(database):
    """
    Refresh views in the database at path DATABASE.

    End-users shouldn't typically need to run this command.
    """
    from apitofsim.workflow import ExperimentDatabase

    db = ExperimentDatabase(database)
    db.refresh_views()


if __name__ == "__main__":
    cli()
