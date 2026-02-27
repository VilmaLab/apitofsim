import pathlib
from collections import Counter
from itertools import combinations
from typing import List, Tuple

import click
from ase import Atoms


@click.group()
def cli():
    """
    The command line interface to apitofsim
    """
    pass


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
def prepare(mode, config, database, db_type, warm):
    """
    Prepare the database according to MODE at path DATABASE using the json configuration file at path CONFIG.

    Typically, you will be creating an database in order to run simulation experiments,
    and so --db-type should be kept as --db-type=experiment.

    MODE can be create or append, depending on whether you want to create a new database or append to an existing one.

    You may choose to `--warm` your database, so that histogramming is done now, in advance of the simulation itself.

    The CONFIG file specifies a TOML formatted file containing both where to import the information about the cluster and pathways from, and the parameters to use for the simulation.

    **Pathways**
    For pathways, you can specify the `type` as "legacy_glob" or "csv".

    The "legacy_glob" type imports pathways from the legacy .in files matching a glob pattern.
    The per-cluster data is then taken either from .dat files specified in the .in file or ORCA or Gaussian outputs files next to the .in file.

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
    # By default, all attributes are taken from the guassian source
    default_source = "gaussian"
    # Individual attributes can be taken from different sources and combined using simple expressions. Here, the electronic energy is taken as the sum of the final single point energy from orca and the zero point energy from gaussian.
    electronic_energy = "orca.final_single_point_energy + gaussian.zero_point_energy"

    # Here the sources for cluster information are specified.
    # The dat source is actually unused in this example
    [pathways.clusters.sources.dat]

    # The ORCA source is only used for part of the electronic energy
    [pathways.clusters.sources.orca]
    # append_to_common_prefix means that the common name of the cluster will be taken as the .in file name with .out appended
    append_to_common_prefix = ".out"

    # Finally, the Guassian source, where most attributes are taken from is specified
    [pathways.clusters.sources.gaussian]
    append_to_common_prefix = ".log"
    ```

    For the "csv" type, you specify the pathways and clusters in separate CSV files, with, as above, the information about how to combine sources specified in the toml config file.

    ```toml
    [[pathways]]
    type = "csv"
    pathways_path = "pathways.csv"
    clusters_path = "clusters.csv"

    # These are the same as legacy_glob, see above
    [pathways.clusters]
    default_source = "gaussian"
    electronic_energy = "orca.final_single_point_energy + gaussian.zero_point_energy"

    [pathways.clusters.sources.orca]
    append_to_common_prefix = ".out"

    [pathways.clusters.sources.gaussian]
    append_to_common_prefix = ".log"
    ```

    Then `clusters.csv` associates filename prefixes with common names for clusters:
    ```csv
    name,prefix
    1A_1SA_negative,negative/1A_1SA
    1A_2SA_negative,negative/1A_2SA
    ```

    While `pathways.csv` relates the parent clusters to their products for each pathway:
    ```csv
    parent,product1,product2
    2A_2SA_negative,1A_1SA_negative,1A_1SA_neutral
    2A_3SA_negative,1A_1SA_negative,1A_2SA_neutral
    ```

    Note that `apitofsim generate pathways` can be used to generate these CSV files from a directory of QC output files.
    They can then be edited, e.g. with Excel, to remove unwanted clusters and pathways.

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
    from copy import deepcopy
    from pprint import pprint

    from tomlkit_extras import TOMLDocumentDescriptor, load_toml_file

    from apitofsim.config import import_raw_config
    from apitofsim.ingest.common import CombineError
    from apitofsim.workflow import (
        ClusterDatabase,
        DerivedDataPreparer,
        ExperimentDatabase,
        SuperClusterDatabase,
        ingest_tree,
    )

    def iter_raw_configs(config):
        for config in config.get("configs", []):
            yield config["name"], {**config.get("default_config", {}), **config}

    if os.path.exists(database) and mode == "create":
        raise click.ClickException(
            f"Database file {database} already exists, will not overwrite (delete it yourself first if you want)"
        )
    if not os.path.exists(database) and mode != "create":
        raise click.ClickException(
            f"Database file {database} does not exist, cannot append"
        )

    if db_type == "experiment":
        db = ExperimentDatabase(database)
    elif db_type == "cluster":
        db = ClusterDatabase(database)
        if warm:
            raise click.ClickException(
                "Warm option is not supported for cluster database"
            )
    elif db_type == "super":
        db = SuperClusterDatabase(database)
    else:
        assert False

    if mode == "create":
        db.create_tables()

    source = load_toml_file(config)
    path_base = pathlib.Path(config).parent
    source_safe = deepcopy(source)
    if "configs" in source_safe:
        # It looks like tomlkit-extras can't handle arrays and stuff put in [...]
        del source_safe["configs"]
    try:
        ingest_tree(
            db, source["pathways"], path_base, TOMLDocumentDescriptor(source_safe)
        )
    except CombineError as e:
        line_no = e.info.get("line_no")
        path = e.info.get("path")
        source_name = e.info.get("source_name")
        raise click.ClickException(
            f"Problem in configuration file {config} at {line_no}\n"
            + f'[[{path}]] = "{source_name}"\n'
            + str(e.info["exception"])
            + "\n"
            "Available source quantities were:\n"
            + "\n".join(
                f"- {quantity}"
                for quantity in e.info.get("available_source_quantities", [])
            )
        )

    if warm and "densityandrate_configs" in source:
        assert isinstance(db, SuperClusterDatabase)
        preparer = DerivedDataPreparer(db)
        for config_dict in source["densityandrate_configs"].unwrap():
            print("Warming up density and rate for config:")
            pprint(config_dict)
            cluster_indexed, _, pathway_lookup = db.get_all_lookups()
            preparer.run_densityandrate(
                import_raw_config(config_dict), cluster_indexed, pathway_lookup
            )

    if db_type == "experiment":
        assert isinstance(db, ExperimentDatabase)
        for name, config_dict in iter_raw_configs(source.unwrap()):
            db.insert_config(name, config_dict)
            if warm:
                print("Warming up density and rate for config:", name)
                pprint(config_dict)
                preparer = DerivedDataPreparer(db)
                cluster_indexed, _, pathway_lookup = db.get_all_lookups()
                preparer.run_densityandrate(
                    import_raw_config(config_dict), cluster_indexed, pathway_lookup
                )


@db.command(short_help="Run the simulations according to the prepared database")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--strict-dos/--no-strict-dos",
    default=False,
    help="Whether to fail early when particle energy go above the max energy the DOS is histogrammed for",
)
@click.option(
    "--filter-parent",
    default=None,
    help="Only run pathways with a specified common name for the parent cluster",
)
@click.option(
    "--filter-pathway",
    multiple=True,
    default=None,
    help="Only run the pathway specified using common names as 'PARENT,CHILD,CHILD'",
)
@click.option(
    "--filter-config",
    multiple=True,
    default=None,
    help="Only run the experiment the parameters in the named configuration",
)
@click.option(
    "--pathway-at-a-time", default=False, is_flag=True, help="Run one pathway at a time"
)
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
        raise click.ClickException(
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
    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")
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
    try:
        import holoviews  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")

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
        raise NotImplementedError("TODO: Single pathway experiments not supported yet")
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


@cli.group(short_help="Commands to convert between different file formats")
def convert():
    pass


def remove_none(obj):
    if isinstance(obj, dict):
        return {k: remove_none(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_none(v) for v in obj if v is not None]
    else:
        return obj


@convert.command(help="Convert .in configuration files to .toml")
@click.argument(
    "config_in", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.argument("config_out", required=True, type=click.File("w"))
def config(config_in, config_out):
    """
    Convert an legacy .in configuration file at path CONFIG_IN to a new .toml configuration file at CONFIG_OUT.
    """
    import orjson
    from tomlkit import dump

    from apitofsim.config import ConfigFile, dump_to_raw

    config = ConfigFile(filename=config_in)
    obj = config.into_json_config()
    # TOOD: Get rid of this; need reimplement orjson dumping numpy conversion/default behaviour
    obj_roundtripped = orjson.loads(dump_to_raw(obj))
    obj_no_none = remove_none(obj_roundtripped)
    assert isinstance(obj_no_none, dict)
    exported_config = {"name": "converted_config", **obj_no_none}
    dump({"configs": [exported_config]}, config_out)


@cli.group(short_help="Commands to inspect different files")
def inspect():
    pass


@inspect.command(
    short_help="Inspect a log file from a QC processing program such as Gaussian or ORCA"
)
@click.argument(
    "format",
    required=True,
    type=click.Choice(["gaussian", "orca"], case_sensitive=False),
)
@click.argument("log_in", required=True, type=click.Path(exists=True, dir_okay=False))
def qc_log(format, log_in):
    from pprint import pprint

    if format == "orca":
        from apitofsim.ingest.orca import parse_orca

        with open(log_in) as f:
            orca_result = parse_orca(f)
            pprint(orca_result)
    if format == "gaussian":
        from apitofsim.ingest.gaussian import parse_gaussian

        with open(log_in) as f:
            gaussian_result = parse_gaussian(f)
            pprint(gaussian_result)


def atoms_to_counter(atoms: Atoms) -> Counter[str]:
    """Convert an Atoms object to a Counter of element symbols."""
    return Counter(atoms.get_chemical_symbols())


def find_combination_triples(
    counters: List[Counter[str]],
) -> List[Tuple[int, int, int]]:
    """
    Given a list of Atoms objects, find all triples (i, j, k) of indices such
    that atoms_list[i] + atoms_list[j] has exactly the same atoms as
    atoms_list[k] (i.e. i and j are reactants that combine to form product k).

    Returns a list of (reactant_a_index, reactant_b_index, product_index)
    tuples. Each unordered pair {i, j} appears at most once per product k,
    with i < j.
    """
    # Build a lookup from a frozen counter (i.e. a composition signature)
    # to the list of indices that share that composition.
    # This lets us do O(1) product lookups instead of scanning the whole list.
    from collections import defaultdict

    composition_to_indices: dict[frozenset[tuple[str, int]], list[int]] = defaultdict(
        list
    )
    for idx, c in enumerate(counters):
        key = frozenset(c.items())
        composition_to_indices[key].append(idx)

    results: List[Tuple[int, int, int]] = []

    # Enumerate all pairs of potential reactants
    for i, j in combinations(range(len(counters)), 2):
        # The combined composition is the element-wise sum of both counters
        combined = counters[i] + counters[j]
        combined_key = frozenset(combined.items())

        # Check whether any Atoms object in our list matches the combined
        # composition (those would be the products)
        for k in composition_to_indices.get(combined_key, []):
            results.append((i, j, k))

    return results


def generate_common_names(paths):
    result = []
    level = 0
    while 1:
        for path in paths:
            name = path.stem
            for i in range(level):
                try:
                    name += "_" + path.parents[i].stem
                except IndexError:
                    raise ValueError(
                        f"Failed to generate unique common name for {path}, ran out of disambiguating parent directories"
                    )
            if name in result:
                result.clear()
                level += 1
                break
            result.append(name)
        else:
            break
    return result


@cli.group(short_help="Commands to inspect different files")
def generate():
    pass


@generate.command(short_help="")
@click.argument(
    "format",
    required=True,
    type=click.Choice(["gaussian", "orca", "xyz"], case_sensitive=False),
)
@click.argument(
    "pathways_out",
    required=True,
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "clusters_out",
    required=True,
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "files",
    required=True,
    nargs=-1,
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.option("-g", "--guess-prefix", is_flag=True, help="")
def pathways(format, pathways_out, clusters_out, files, guess_prefix):
    from ase.io import read as ase_read

    counters = []
    for file in files:
        if format == "orca":
            atoms = ase_read(file, format="orca-output")[0]
        elif format == "gaussian":
            atoms = ase_read(file, format="gaussian-out")[0]
        else:
            assert format == "xyz"
            atoms = ase_read(file, format="xyz")
        assert isinstance(atoms, Atoms)
        counters.append(atoms_to_counter(atoms))
    common_names = generate_common_names(files)
    output_paths = []
    for file in files:
        output_path = file.relative_to(clusters_out.parent)
        if guess_prefix:
            output_path = output_path.parent / output_path.stem
        output_paths.append(str(output_path))
    triples = find_combination_triples(counters)
    with open(pathways_out, "w") as out:
        out.write("parent,product1,product2\n")
        for i, j, k in triples:
            out.write(
                ",".join([common_names[k], common_names[i], common_names[j]]) + "\n"
            )
    with open(clusters_out, "w") as out:
        if guess_prefix:
            attr = "prefix"
        else:
            attr = format
        out.write(f"name,{attr}\n")
        for name, path in zip(common_names, output_paths):
            out.write(f"{name},{path}\n")


if __name__ == "__main__":
    cli()
