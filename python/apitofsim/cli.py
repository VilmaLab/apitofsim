import pathlib
from collections import Counter
from itertools import combinations
from typing import List, Tuple

import click
import pandas
from ase import Atoms

from apitofsim.workflow.db import connection_scope, guess_ase_db_filename


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
    type=click.Choice(["experiment", "cluster", "super", "realization"]),
    default="experiment",
    help="The type of database to create: this will determine which tables are created",
)
@click.option("-a", "--ase", is_flag=True, help="Create a linked ASE database")
@click.option(
    "-w", "--warm", is_flag=True, help="Warm the database with histogrammed data"
)
def prepare(mode, config, database, db_type, ase, warm):
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

    \b
    ```toml
    [[pathways]]
    type = "legacy_glob"
    # An optional prefix to add to the common names of all clusters imported here
    prefix = ""
    # A glob pattern to use to find .in files.
    # The wildcard * matches any number of characters, while ** can span directories.
    path = "/path/to/**/*.in"
    # Paths specified in the .in files are relative to the .in file directory.
    cwd = "."
    \b
    [pathways.clusters]
    # By default, all attributes are taken from the gaussian source
    default_source = "gaussian"
    # Individual attributes can be taken from different sources and combined
    # using simple expressions. Here, the electronic energy is taken as the
    # sum of the final single point energy from orca and the zero point energy
    # from gaussian.
    electronic_energy = "orca.final_single_point_energy + gaussian.zero_point_energy"
    \b
    # Here the sources for cluster information are specified.
    # The dat source is actually unused in this example
    [pathways.clusters.sources.dat]
    \b
    # The ORCA source is only used for part of the electronic energy
    [pathways.clusters.sources.orca]
    # append_to_common_prefix means that the common name of the cluster will
    # be taken as the .in file name with .out appended
    append_to_common_prefix = ".out"
    \b
    # Finally, the Gaussian source, where most attributes are taken from is specified
    [pathways.clusters.sources.gaussian]
    append_to_common_prefix = ".log"
    ```

    For the "csv" type, you specify the pathways and clusters in separate CSV files, with, as above, the information about how to combine sources specified in the toml config file.

    \b
    ```toml
    [[pathways]]
    type = "csv"
    pathways_path = "pathways.csv"
    clusters_path = "clusters.csv"
    \b
    # These are the same as legacy_glob, see above
    [pathways.clusters]
    default_source = "gaussian"
    electronic_energy = "orca.final_single_point_energy + gaussian.zero_point_energy"
    \b
    [pathways.clusters.sources.orca]
    append_to_common_prefix = ".out"
    \b
    [pathways.clusters.sources.gaussian]
    append_to_common_prefix = ".log"
    ```

    \b
    Then `clusters.csv` associates filename prefixes with common names for clusters:
    ```csv
    name,prefix
    1A_1SA_negative,negative/1A_1SA
    1A_2SA_negative,negative/1A_2SA
    ```

    \b
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

    \b
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
    pressures = [ [ 194.0, 3.88 ], "pascal"]
    resolution = 1_000
    tolerance = 1e-8
    voltages = [ [ -19, -9, -7, -6, 11 ], "volt" ]
    \b
    [default_config.gas]
    radius = "1.84e-10 meter"
    mass = "4.65e-26 kilogram"
    adiabatic_index = 1.4
    \b
    [[configs]]
    name = "simple"
    \b
    [[configs]]
    name = "with-quadrupole-and-pinhole"
    radius_pinhole = "1 mm"
    \b
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
        RealizationDatabase,
        SuperClusterDatabase,
        ingest_tree,
    )

    def iter_raw_configs(config):
        for config in config.get("configs", []):
            yield config["name"], {**config.get("default_config", {}), **config}

    if ase:
        ase_path = guess_ase_db_filename(database)
    else:
        ase_path = None

    for path in (database, *((ase_path,) if ase_path else ())):
        if os.path.exists(path) and mode == "create":
            raise click.ClickException(
                f"Database file {path} already exists, will not overwrite (delete it yourself first if you want)"
            )
        if not os.path.exists(path) and mode != "create":
            raise click.ClickException(
                f"Database file {path} does not exist, cannot append"
            )

    if db_type == "experiment":
        db_cls = ExperimentDatabase
    elif db_type == "cluster":
        if warm:
            raise click.ClickException(
                "Warm option is not supported for cluster database"
            )
        db_cls = ClusterDatabase
    elif db_type == "super":
        db_cls = SuperClusterDatabase
    elif db_type == "realization":
        db_cls = RealizationDatabase
    else:
        assert False

    with connection_scope(db_cls, database, ase_filename=ase_path) as db:
        if mode == "create":
            db.create_tables()

        source = load_toml_file(config)
        path_base = pathlib.Path(config).parent
        source_safe = deepcopy(source)
        # It looks like tomlkit-extras can't handle arrays and stuff put in [...]
        if "configs" in source_safe:
            del source_safe["configs"]
        if "default_config" in source_safe:
            del source_safe["default_config"]
        try:
            ingest_tree(
                db,
                source["pathways"],
                path_base,
                TOMLDocumentDescriptor(source_safe),
                ingest_ase=ase,
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
    help="Only run the experiment using the parameters in the named configuration",
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

    with connection_scope(ExperimentDatabase, database) as db:
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


def select_experiment_choices(db, filter_config):
    df = db.report_df("experiment_summary")
    for row in df.itertuples():
        pathway_desc = "single pathway" if row.is_single_pathway else "multi-pathway"
        yield (
            f"#{row.experiment_run_id} {row.config_name} run at {row.start_time} "
            f"({pathway_desc}, success rate: {row.successes}/{row.successes + row.failures})",
            (row.experiment_run_id, row.is_single_pathway),
        )


def select_experiment(db, filter_config=None):
    import questionary

    choices = []
    for label, value in select_experiment_choices(db, filter_config):
        choices.append(questionary.Choice(label, value=value))
    if len(choices) == 0:
        raise click.ClickException("No experiments found in the database")
    if len(choices) == 1:
        return choices[0].value
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


def select_cluster_result(db, filter_parent=None, filter_config=None):
    import questionary

    tbl_orig = tbl = db.db.table("experiment_cluster_report")
    if filter_parent:
        tbl = tbl.filter(f"(cluster_common_name ~~~ '{filter_parent}')")
    matches = tbl.count("*").fetchone()[0]
    if matches == 0:
        raise click.ClickException(
            f"No cluster results found in the database with parent cluster name matching glob '{filter_parent}'"
        )
    if filter_config:
        print("filter_config", filter_config)
        tbl = tbl.filter(f"(config_name ~~~ '{filter_config}')")
    matches = tbl.count("*").fetchone()[0]
    if matches == 0:
        tbl_test = tbl_orig.filter(f"(config_name ~~~ '{filter_config}')")
        matches_test = tbl_test.count("*").fetchone()[0]
        if matches_test == 0:
            raise click.ClickException(
                f"No cluster results found in the database with config name matching glob '{filter_config}'"
            )
        else:
            raise click.ClickException(
                f"No cluster results found in the database with config name matching glob '{filter_config}' "
                f"and config name matching glob '{filter_parent}' (but both were found individually)"
            )
    df = tbl.fetchdf()
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
    if len(choices) == 1:
        return choices[0].value
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


@plot.command(short_help="Plot survival rates for each parent cluster in an experiment")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("pngout", type=click.Path(dir_okay=False))
def survival(database, pngout):
    """
    Output to PNGOUT a bar chart of the survival rate for each cluster in the database at path DATABASE.
    """
    from pprint import pprint

    from apitofsim.plotting import get_joint_survivals, make_survival_plot
    from apitofsim.workflow import ExperimentDatabase

    with connection_scope(ExperimentDatabase, database, readonly=True) as db:
        experiment_id, _ = select_experiment(db)
        joint_survivals = get_joint_survivals(db, experiment_id)
    pprint(joint_survivals)
    make_survival_plot(pngout, joint_survivals.keys(), joint_survivals.values())


def transform_intensity(df, model_transmission):
    from apitofsim.transmission import (
        new_transmission_neg,
        new_transmsision_pos,
        old_transmission,
    )

    if model_transmission is not None:
        model_transmission_func = {
            "old": old_transmission,
            "new_neg": new_transmission_neg,
            "new_pos": new_transmsision_pos,
        }[model_transmission]
        df["intensity"] = model_transmission_func(df["atomic_mass"]) * df["intensity"]


@plot.command(short_help="Plot a spectrogram for a single cluster in an experiment")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("pngout", type=click.Path(dir_okay=False))
@click.option(
    "--filter-parent",
    default=None,
    help="Only run pathways with a specified common name for the parent cluster",
)
@click.option(
    "--filter-config",
    default=None,
    help="Only run the experiment using the parameters in the named configuration",
)
@click.option(
    "--model-transmission",
    type=click.Choice(["old", "new_neg", "new_pos"]),
    default=None,
)
@click.option(
    "--label",
    type=click.Choice(
        [
            "all",
            "nonzero",
            "threshold",
            "none",
        ],
        case_sensitive=False,
    ),
    default="none",
    help="Add labels indicating the cluster",
)
@click.option(
    "--label-threshold",
    type=float,
    default=0.1,
    help="specified threshold for labeling clusters when --label=threshold",
)
def spectrogram(
    database,
    pngout,
    filter_parent,
    filter_config,
    model_transmission,
    label,
    label_threshold,
):
    """
    Output to PNGOUT a spectrogram of the results for single cluster / experiment using the database at path DATABASE.
    """
    from apitofsim.plotting import (
        get_intensities,
        plot_spectrogram_to_file,
    )
    from apitofsim.workflow import ExperimentDatabase

    with connection_scope(ExperimentDatabase, database, readonly=True) as db:
        experiment_id, cluster_id, is_single_pathway = select_cluster_result(
            db, filter_parent, filter_config
        )
        df = get_intensities(db, experiment_id, cluster_id, is_single_pathway)
    transform_intensity(df, model_transmission)
    plot_spectrogram_to_file(
        pngout, df, scale="max", label=label, label_threshold=label_threshold
    )


@plot.command(short_help="Plot a spectrogram for each cluster in an experiment")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("dirout", type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option(
    "--filter-config",
    default=None,
    help="Only run the experiment using the parameters in the named configuration",
)
@click.option(
    "--model-transmission",
    type=click.Choice(["old", "new_neg", "new_pos"]),
    default=None,
)
@click.option(
    "--label",
    type=click.Choice(
        [
            "all",
            "nonzero",
            "threshold",
            "none",
        ],
        case_sensitive=False,
    ),
    default="none",
    help="Add labels indicating the cluster",
)
@click.option(
    "--label-threshold",
    type=float,
    default=0.1,
    help="specified threshold for labeling clusters when --label=threshold",
)
def spectrogram_many(
    database, dirout, filter_config, model_transmission, label, label_threshold
):
    """
    Output to DIROUT a spectrogram per cluster using the results from single experiments using the database at path DATABASE.
    """
    from apitofsim.plotting import (
        get_intensities,
        plot_spectrogram_to_file,
    )
    from apitofsim.workflow import ExperimentDatabase

    with connection_scope(ExperimentDatabase, database, readonly=True) as db:
        experiment_id, is_single_pathway = select_experiment(
            db, filter_config=filter_config
        )
        df = get_intensities(db, experiment_id, is_single_pathway)
    transform_intensity(df, model_transmission)
    dirout.mkdir(exist_ok=True, parents=True)
    max_x = df["atomic_mass"].max() * 1.1
    for parent_name, cluster_df in df.groupby("parent_name"):
        pngout = dirout / f"{parent_name}.png"
        plot_spectrogram_to_file(
            pngout,
            cluster_df,
            scale="max",
            max_x=max_x,
            label=label,
            label_threshold=label_threshold,
        )


@db.command(short_help="Produce an Excel-friendly CSV report from the database")
@click.argument(
    "report_type",
    type=click.Choice(
        [
            "cluster-report",
            "pathway-report",
            "experiment-pathway-report",
            "experiment-cluster-report",
            "experiment-summary",
            "spectrogram",
        ],
        case_sensitive=False,
    ),
    required=False,
)
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("csvout", type=click.Path(dir_okay=False))
def report(report_type, database, csvout):
    """
    Produce a report REPORT_TYPE from the database at path DATABASE and write it to CSV at path CSVOUT.

    All databases have the following reports
    * The cluster-report contains one row per parent cluster.
    * The pathway-report contains the input pathways giving one row per pathway, with no information about results.

    Databases created as --db-type=experiment additionally have the following reports:
    * The experiment-pathway-report contains one row per pathway / experiment run, and includes the outcome of that run for that pathway.
    * The experiment-cluster-report per parent cluster / experiment run, and includes the summarises information results from its pathways.
    * The experiment-summary contains one row per experiment run, and summarizes the outcomes across all pathways for that run.
    * The spectrogram report contains the same data used to plot spectograms.
    """
    from apitofsim.workflow import ExperimentDatabase

    with connection_scope(ExperimentDatabase, database, readonly=True) as db:
        if not db.is_experiment_db() and report_type in {
            "experiment-pathway-report",
            "experiment-cluster-report",
            "experiment-summary",
            "spectrogram",
        }:
            raise click.ClickException(
                f"Report type {report_type} is only available for experiment databases"
            )

        if report_type == "spectrogram":
            from apitofsim.plotting import get_intensities

            dataframes: List[pandas.DataFrame] = []
            for row in db.report_df("experiment_summary").itertuples():
                dataframes.append(
                    get_intensities(
                        db,
                        row.experiment_run_id,
                        is_single_pathway=row.is_single_pathway,
                    )
                )
            df = pandas.concat(dataframes)
            df.to_csv(csvout)
        else:
            db.db.table(report_type.replace("-", "_")).to_csv(csvout)


@db.command(help="Refresh views in the database at path DATABASE (please ignore)")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
def refresh_views(database):
    """
    Refresh views in the database at path DATABASE.

    End-users shouldn't typically need to run this command.
    """
    from apitofsim.workflow import ExperimentDatabase

    with connection_scope(ExperimentDatabase, database) as db:
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
            atoms = ase_read(file, format="orca-output")
        elif format == "gaussian":
            atoms = ase_read(file, format="gaussian-out")
        else:
            assert format == "xyz"
            atoms = ase_read(file, format="xyz")
        if isinstance(atoms, list):
            atoms = atoms[-1]
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
