import pathlib

import click
from ase import Atoms

from apitofsim.workflow import SimulationMode
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

    from apitofsim.config import check_for_deprecated_keys, import_raw_config
    from apitofsim.ingest.common import CombineError
    from apitofsim.workflow import (
        ClusterDatabase,
        DerivedDataPreparer,
        ExperimentDatabase,
        RealizationDatabase,
        SuperClusterDatabase,
        ingest_tree,
    )

    def iter_raw_configs(config_root):
        for config in config_root.get("configs", []):
            yield config["name"], {**config_root.get("default_config", {}), **config}

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
                cluster_indexed, name_lookup, pathway_lookup = db.get_all_lookups()
                preparer.run_densityandrate(
                    import_raw_config(config_dict),
                    cluster_indexed,
                    pathway_lookup,
                    name_lookup=name_lookup,
                )

        if db_type in ("experiment", "realization"):
            assert isinstance(db, ExperimentDatabase)
            for name, config_dict in iter_raw_configs(source.unwrap()):
                check_for_deprecated_keys(config_dict)
                db.insert_config(name, config_dict)
                if warm:
                    print("Warming up density and rate for config:", name)
                    pprint(config_dict)
                    preparer = DerivedDataPreparer(db)
                    cluster_indexed, name_lookup, pathway_lookup = db.get_all_lookups()
                    preparer.run_densityandrate(
                        import_raw_config(config_dict),
                        cluster_indexed,
                        pathway_lookup,
                        name_lookup=name_lookup,
                    )


@db.command(
    short_help="Run the simulations according to the prepared database",
    context_settings={"token_normalize_func": lambda x: x.replace("_", "-").lower()},
)
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
    "--simulation-mode",
    default=SimulationMode.SINGLE_CLUSTER,
    type=click.Choice(SimulationMode, case_sensitive=False),
    help="Simulation mode",
)
@click.option("--verbose", default=False, is_flag=True)
def run(
    database,
    strict_dos,
    filter_parent,
    filter_pathway,
    filter_config,
    simulation_mode,
    verbose,
):
    """
    Run simulation according to the configurations in DATABASE.
    """
    from apitofsim.workflow import (
        ExperimentDatabase,
        ExperimentRunner,
        RealizationDatabase,
        auto_db_type,
    )

    with connection_scope(auto_db_type, database) as db:
        if not isinstance(db, (ExperimentDatabase, RealizationDatabase)):
            raise click.ClickException(
                "Database must be created as an experiment database or a realization database"
            )
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
            mode=simulation_mode,
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
        df = get_intensities(
            db,
            experiment_id=experiment_id,
            cluster_id=cluster_id,
            is_single_pathway=is_single_pathway,
        )
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
@click.option("--verbose", default=False, is_flag=True)
def spectrogram_many(
    database, dirout, filter_config, model_transmission, label, label_threshold, verbose
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
        df = get_intensities(
            db, experiment_id=experiment_id, is_single_pathway=is_single_pathway
        )
    transform_intensity(df, model_transmission)
    dirout.mkdir(exist_ok=True, parents=True)
    max_x = df["atomic_mass"].max() * 1.1
    if verbose:
        print("Intensities:")
        print(df)
    for parent_name, cluster_df in df.groupby("parent_name"):
        pngout = dirout / f"{parent_name}.png"
        if verbose:
            print(f"Writing {pngout}")
        plot_spectrogram_to_file(
            pngout,
            cluster_df,
            scale="max",
            max_x=max_x,
            label=label,
            label_threshold=label_threshold,
        )


@plot.command(
    short_help="Plot a diagram of events (collisions/fragmentations/escapes) for a whole run"
)
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("dirout", type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option(
    "-t",
    "--plot-type",
    type=click.Choice(
        [
            "off-center",
            "off-center-facet",
            "beeswarm",
            "beeswarm-facet",
            "stripplot",
            "stripplot-facet",
            "violinplot",
            "violinplot-facet",
        ]
    ),
    default="experiment",
    help="The plot type",
)
@click.option(
    "-r", "--rescale", type=click.Choice(["none", "equal", "schematic"]), default="none"
)
def plot_events(database, dirout, plot_type, rescale):
    """
    Output to DIROUT a diagram of events (collisions/fragmentations/escapes) for a whole run using the database at path DATABASE.
    """

    import numpy as np
    from seaborn import (
        FacetGrid,
        relplot,
        scatterplot,
        stripplot,
        swarmplot,
        violinplot,
    )

    from apitofsim.config import import_raw_config
    from apitofsim.workflow import RealizationDatabase

    def get_geometery(db, experiment_run_id):
        import orjson

        res = db.db.execute(
            """
            select
                config
            from
                experiment_config
            inner join
                experiment_run
                on experiment_run.experiment_config_id = experiment_config.id
            where
                experiment_run.id = ?
            """,
            (experiment_run_id,),
        ).fetchone()
        assert res is not None
        config = res[0]
        config = import_raw_config(orjson.loads(config))
        return config["lengths"]

    def get_events(db, experiment_run_id, is_single_pathway):
        assert not is_single_pathway
        return db.db.execute(
            "select * from event_report where experiment_result_id = ?",
            (experiment_run_id,),
        ).fetchdf()
        # return db.db.table("realization_events").filter(f"experiment_run_id = {experiment_id}").fetchdf()

    with connection_scope(RealizationDatabase, database, readonly=True) as db:
        experiment_id, is_single_pathway = select_experiment(db)
        if is_single_pathway:
            raise click.ClickException(
                "Event plotting is not supported for single pathway experiments"
            )

        lengths = get_geometery(db, experiment_id)

        first_chamber_end = lengths[0]
        sk_end = first_chamber_end + lengths[-1]
        quadrupole_start = sk_end + lengths[1]
        quadrupole_end = quadrupole_start + lengths[2]
        second_chamber_end = quadrupole_end + lengths[3]
        # total_length = second_chamber_end
        cumulative_lengths = [
            first_chamber_end,
            sk_end,
            quadrupole_start,
            quadrupole_end,
            second_chamber_end,
        ]
        cumulative_lengths = [
            length.to("meters").magnitude for length in cumulative_lengths
        ]
        cumulative_lengths.insert(0, 0.0)
        cumulative_lengths = np.array(cumulative_lengths)

        df = get_events(db, experiment_id, is_single_pathway)
        dirout.mkdir(exist_ok=True, parents=True)
        df["d"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        df["event_type"] = df["event_type"].astype("category")
        if rescale == "equal":
            insertion_points = np.searchsorted(cumulative_lengths, df["z"])
            insertion_points = np.clip(insertion_points, 0, len(cumulative_lengths) - 1)
            new_z = []
            for idx, z in zip(insertion_points, df["z"]):
                lo = cumulative_lengths[idx - 1]
                hi = cumulative_lengths[idx]
                new_z.append(
                    (z - lo) / (hi - lo) + (idx - 1) / (len(cumulative_lengths) - 1)
                )
            df["z"] = np.array(new_z)
        elif rescale == "schematic":
            # insertion_points = np.searchsorted(cumulative_lengths, df["z"])
            raise NotImplementedError("schematic rescaling is not implemented yet")
        for parent_name, cluster_df in df.groupby("parent_name"):
            cluster_df = cluster_df.sort_values(by=["event_type"])
            if plot_type == "off-center":
                ax = scatterplot(
                    cluster_df,
                    s=5,
                    linewidths=0.25,
                    marker="x",
                    x="z",
                    y="d",
                    hue="event_type",
                )
            elif plot_type == "off-center-facet":
                ax = relplot(
                    cluster_df,
                    s=5,
                    linewidths=0.25,
                    marker="x",
                    x="z",
                    y="d",
                    col="event_type",
                    kind="scatter",
                )
            elif plot_type == "beeswarm":
                ax = swarmplot(cluster_df, x="z", hue="event_type")
            elif plot_type == "beeswarm-facet":
                ax = swarmplot(cluster_df, x="z", col="event_type")
            elif plot_type == "stripplot":
                ax = stripplot(
                    cluster_df, s=5, linewidth=0.25, marker="x", x="z", hue="event_type"
                )
            elif plot_type == "stripplot-facet":
                ax = stripplot(
                    cluster_df, s=5, linewidth=0.25, marker="x", x="z", y="event_type"
                )
            elif plot_type == "violinplot":
                ax = violinplot(cluster_df, x="z", split=True, cut=0)
            elif plot_type == "violinplot-facet":
                ax = violinplot(cluster_df, x="z", y="event_type", split=True, cut=0)
            else:
                raise click.ClickException(f"Unknown plot type {plot_type}")
            if rescale == "equal":
                boundaries = np.linspace(0, 1, len(cumulative_lengths))
            elif rescale == "schematic":
                raise NotImplementedError("schematic rescaling is not implemented yet")
            else:
                assert rescale == "none"
                boundaries = cumulative_lengths
            if plot_type in ("off-center", "off-center-facet"):
                if isinstance(ax, FacetGrid):
                    for ax in ax.axes:
                        ax.vlines(boundaries, 0.0, cluster_df["d"].max(), color="black")
                else:
                    ax.vlines(boundaries, 0.0, cluster_df["d"].max(), color="black")
            else:
                for x in boundaries:
                    if isinstance(ax, FacetGrid):
                        for ax in ax.axes:
                            ax.axvline(x, color="black")
                    else:
                        ax.axvline(x, color="black")
            pngout = dirout / f"{parent_name}.png"
            if isinstance(ax, FacetGrid):
                fig = ax.figure
                fig.savefig(
                    str(pngout),
                    dpi=150,
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                )
            else:
                fig = ax.get_figure(root=True)
                assert fig is not None
                fig.savefig(
                    str(pngout),
                    dpi=150,
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
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
            "event-report",
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
    from apitofsim.plotting import UnknownReportTypeError, get_report
    from apitofsim.workflow import auto_db_type

    with connection_scope(auto_db_type, database, readonly=True) as db:
        try:
            df = get_report(db, report_type)
        except UnknownReportTypeError as e:
            raise click.ClickException(e.message)
        df.to_csv(csvout)


@db.command(help="Refresh views in the database at path DATABASE (please ignore)")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
def refresh_views(database):
    """
    Refresh views in the database at path DATABASE.

    End-users shouldn't typically need to run this command.
    """
    from apitofsim.workflow import auto_db_type

    with connection_scope(auto_db_type, database) as db:
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


def is_array(val):
    import numpy as np
    from pint import Quantity

    return (
        isinstance(val, np.ndarray)
        or isinstance(val, Quantity)
        and isinstance(val.magnitude, np.ndarray)
    )


def is_energy(val):
    from pint import Quantity

    if not isinstance(val, Quantity):
        return False
    if val.dimensionality == {"[energy]": 1}:
        return True
    return val.units == "hartree" or val.units == "eV"


def get_pathways_iter(config):
    from tomlkit_extras import load_toml_file

    from apitofsim.ingest.csv import parse_csv_tree
    from apitofsim.ingest.legacy import parse_legacy_tree

    source = load_toml_file(config).unwrap()
    path_base = pathlib.Path(config).parent
    pathways = source["pathways"]
    if isinstance(pathways, list):
        pathways = pathways[0]
    if pathways["type"] == "legacy_glob":
        return parse_legacy_tree(pathways["path"], pathways["clusters"], path_base)
    elif pathways["type"] == "csv":
        clusters_path = path_base / pathways["clusters_path"]
        current_path_base = clusters_path.parent

        return parse_csv_tree(
            path_base / pathways["pathways_path"],
            clusters_path,
            pathways["clusters"],
            current_path_base,
        )
    else:
        raise click.ClickException(f"Unknown pathway type {pathways['type']}")


@inspect.command(
    short_help="Inspect the quantities that can be ingested from the chemistry data files"
)
@click.argument("config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("outf", required=True, type=click.File("w"))
@click.option("--skip-metadata", is_flag=True)
@click.option("--skip-arrays", is_flag=True)
@click.option("--focus-energies", is_flag=True)
def ingest(config, outf, skip_metadata, skip_arrays, focus_energies):
    from pint import Quantity

    pathways_iter = get_pathways_iter(config)

    header_printed = False
    header_names = []
    units = []

    def dotted_items(tpl):
        result = {}
        for particle_role, particle in zip(("parent", "prod1", "prod2"), tpl):
            result[f"{particle_role}.name"] = particle["name"]
            for attribute, val in particle["particle"].items():
                if skip_arrays and is_array(val):
                    continue
                if focus_energies and not is_energy(val):
                    continue
                result[f"{particle_role}.particle.{attribute}"] = val
            for source, source_dict in particle["sources"].items():
                for attribute, val in source_dict.items():
                    if skip_arrays and is_array(val):
                        continue
                    if focus_energies and not is_energy(val):
                        continue
                    if skip_metadata and attribute in {
                        "input",
                        "software_version",
                        "citation",
                        "version_and_date",
                    }:
                        continue
                    result[f"{particle_role}.{source}.{attribute}"] = val
        return result

    for tpl in pathways_iter:
        pathway_dict = dotted_items(tpl)
        if not header_printed:
            for name, val in pathway_dict.items():
                header_names.append(name)
                if isinstance(val, Quantity):
                    print(f"{name} ({val.units})", end=",", file=outf)
                    units.append(val.units)
                else:
                    print(name, end=",", file=outf)
                    units.append(None)
            print(file=outf)
        header_printed = True
        for name, unit in zip(header_names, units):
            val = pathway_dict.get(name, "MISSING")
            if isinstance(val, Quantity):
                val = val.to(unit).magnitude
            print(str(val).replace(",", ";").replace("\n", " / "), end=",", file=outf)
        print(file=outf)


@inspect.command(short_help="")
@click.argument("config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument(
    "outdir",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
)
def ingest_tree(config, outdir):
    import graphviz

    graph_builder = graphviz.Digraph(
        "ingestrecords",
        filename="ingestrecords.gv",
        node_attr={"shape": "record"},
        graph_attr={"rankdir": "LR"},
    )
    added_nodes = set()

    def add_node(builder, node, identifier=None):
        name = node["name"]
        if identifier is None:
            identifier = name
        number_of_atoms = node["particle"]["number_of_atoms"]
        energy = node["particle"]["electronic_energy"]
        builder.node(
            identifier, label=f"{name}|{number_of_atoms} atoms|{energy:.2f~#P}"
        )

    def add_pathway(
        builder,
        pathway_name,
        parent,
        prod1,
        prod2,
        parent_id=None,
        prod1_id=None,
        prod2_id=None,
    ):
        if parent_id is None:
            parent_id = parent["name"]
        if prod1_id is None:
            prod1_id = prod1["name"]
        if prod2_id is None:
            prod2_id = prod2["name"]
        fragmentation_energy = (
            parent["particle"]["electronic_energy"]
            - prod1["particle"]["electronic_energy"]
            - prod2["particle"]["electronic_energy"]
        )
        builder.node(
            pathway_name, label=f"{fragmentation_energy:.2f~#P}", shape="diamond"
        )
        builder.edge(parent_id, pathway_name)
        builder.edge(pathway_name, prod1_id)
        builder.edge(pathway_name, prod2_id)

    def ensure_node(node):
        name = node["name"]
        if name in added_nodes:
            return
        add_node(graph_builder, node)
        added_nodes.add(name)

    lookup = {}
    pathway_idx = 0
    pathways_iter = get_pathways_iter(config)
    for parent, prod1, prod2 in pathways_iter:
        ensure_node(parent)
        ensure_node(prod1)
        ensure_node(prod2)
        pathway_name = f"pathway{pathway_idx}"
        pathway_idx += 1
        if parent["name"] not in lookup:
            lookup[parent["name"]] = (parent, [])
        add_pathway(graph_builder, pathway_name, parent, prod1, prod2)
        lookup[parent["name"]][1].append((prod1, prod2))

    print("Writing graph")
    outdir.mkdir(exist_ok=True, parents=True)
    with (outdir / "graph.svg").open("wb") as outf:
        outf.write(graph_builder.pipe(format="svg"))
    print("Done writing graph")

    pathway_idx = 0

    def walk(builder, node, path=""):
        nonlocal pathway_idx
        name = node["name"]
        next_path = path + "_" + name
        add_node(builder, node, identifier=next_path)
        if name in lookup:
            node, pathways = lookup[name]
            for prod1, prod2 in pathways:
                pathway_name = f"pathway{pathway_idx}"
                pathway_idx += 1
                add_pathway(
                    builder,
                    pathway_name,
                    node,
                    prod1,
                    prod2,
                    next_path,
                    next_path + "_" + prod1["name"],
                    next_path + "_" + prod2["name"],
                )
                walk(builder, prod1, path=next_path)
                walk(builder, prod2, path=next_path)

    for root in lookup:
        print("Writing tree for root", root)
        tree_name = f"tree_{root}"
        tree_builder = graphviz.Digraph(
            tree_name,
            filename=f"{tree_name}.gv",
            node_attr={"shape": "record"},
            graph_attr={"rankdir": "LR"},
        )
        pathway_idx = 0
        walk(tree_builder, lookup[root][0])
        with (outdir / f"{tree_name}.dot").open("w") as outf:
            print(tree_builder.source, file=outf)
        with (outdir / f"{tree_name}.svg").open("wb") as outf:
            outf.write(tree_builder.pipe(format="svg"))
        print("Done writing tree for root", root)


@inspect.command(short_help="")
@click.argument("database", required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument(
    "outdir",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
)
def db_tree(database, outdir):
    from apitofsim.api import MassSpecSubstanceTreeInput
    from apitofsim.plotting import mk_tree_input_graph
    from apitofsim.workflow.db import ExperimentDatabase
    from apitofsim.workflow.runners import ExperimentRunner

    db = ExperimentDatabase(database)
    runner = ExperimentRunner(db)
    configs = list(db.iter_configs())
    config = configs[0][2]
    (mass_spec, cluster_indexed, name_lookup, pathway_lookup, k_rates, cluster_dos) = (
        runner._prepare_from_config(config)
    )
    roots = runner._prepare_cluster_tree(
        config, cluster_indexed, name_lookup, pathway_lookup, k_rates, cluster_dos
    )
    outdir.mkdir(exist_ok=True, parents=True)
    for pathway_ids, product_ids, tree, root in roots:
        label = root["cluster_label"]
        subs = MassSpecSubstanceTreeInput(config["gas"], tree)
        builder = mk_tree_input_graph(subs)
        with (outdir / f"{label}.svg").open("wb") as outf:
            outf.write(builder.pipe(format="svg"))


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


@cli.group(short_help="Commands to inspect different files")
def generate():
    pass


def filter_blacklist(common_names, files, all_atoms, charges, blacklist_file):
    import fnmatch
    import re

    regex_bits = []
    for line in blacklist_file:
        line = line.strip()
        if line and not line.startswith("#"):
            regex = fnmatch.translate(line)
            regex_bits.append(f"({regex})")
    regex = re.compile("|".join(regex_bits))
    keep_indices = [i for i, name in enumerate(common_names) if not regex.match(name)]
    common_names = [common_names[i] for i in keep_indices]
    files = [files[i] for i in keep_indices]
    all_atoms = [all_atoms[i] for i in keep_indices]
    if charges is not None:
        charges = [charges[i] for i in keep_indices]
    return common_names, files, all_atoms, charges


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
@click.option("--blacklist", type=click.File("r"), help="")
@click.option("-g", "--guess-prefix", is_flag=True, help="")
@click.option(
    "--ignore-charge",
    is_flag=True,
    help="Ignore charge information, overgenerating pathways. Implies --allow-neutral-parents.",
)
@click.option(
    "--allow-neutral-parents",
    is_flag=True,
    help="Generate pathways with a neutral parent. These will be useless for simulation, so only generate these for insepcting your data.",
)
def pathways(
    format,
    pathways_out,
    clusters_out,
    files,
    blacklist,
    guess_prefix,
    ignore_charge,
    allow_neutral_parents,
):
    from ase.io import read as ase_read

    from apitofsim.generate import generate_common_names, viable_fragmentations

    all_atoms = []
    if ignore_charge:
        charges = None
    else:
        charges = []
    for file in files:
        if format == "orca":
            from apitofsim.ingest.orca import parse_orca

            atoms = ase_read(file, format="orca-output")
            if not ignore_charge:
                assert charges is not None
                info = parse_orca(file.open())
                if len(info) >= 1 and "charge" in info[0]:
                    charge = info[0]["charge"]
                    charges.append(charge)
                else:
                    raise ValueError(
                        f"Could not find charge in ORCA output {file}. You can use --ignore-charge to ignore this error and proceed without charge information, but this will overgenerate."
                    )
        elif format == "gaussian":
            from apitofsim.ingest.gaussian import parse_gaussian

            atoms = ase_read(file, format="gaussian-out")
            if not ignore_charge:
                assert charges is not None
                info = parse_gaussian(file.open())
                if "charge" in info:
                    charge = info["charge"]
                    charges.append(charge)
                else:
                    raise ValueError(
                        f"Could not find charge in Gaussian output {file}. You can use --ignore-charge to ignore this error and proceed without charge information, but this will overgenerate."
                    )
        else:
            assert format == "xyz"
            atoms = ase_read(file, format="xyz")
            if not ignore_charge:
                assert charges is not None
                charge = {"positive": 1, "neutral": 0, "negative": -1}.get(
                    file.parent.name.lower()
                )
                if charge is not None:
                    charges.append(charge)
                else:
                    raise ValueError(
                        f"Could not infer charge from parent directory name {file.parent.name} (should be positive, neutral or negative) for xyz file {file}. You can use --ignore-charge to ignore this error and proceed without charge information, but this will overgenerate."
                    )
        if isinstance(atoms, list):
            atoms = atoms[-1]
        assert isinstance(atoms, Atoms)
        all_atoms.append(atoms)
    common_names = generate_common_names(files)
    files = list(common_names.values())
    common_names = list(common_names.keys())
    if blacklist:
        common_names, files, all_atoms, charges = filter_blacklist(
            common_names, files, all_atoms, charges, blacklist
        )
    triples = viable_fragmentations(
        all_atoms, charges, allow_neutral_parents=allow_neutral_parents
    )
    output_paths = []
    for file in files:
        output_path = file.relative_to(clusters_out.parent)
        if guess_prefix:
            output_path = output_path.parent / output_path.stem
        output_paths.append(str(output_path))
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
