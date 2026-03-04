import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison of simulation results with real data

    This notebook compares survival rates and spectograms produced by the simulation and real data. It uses data from this paper:

    Alfaouri, D., Passananti, M., Zanca, T., Ahonen, L.R., Kangasluoma, J., Kubečka, J., Myllys, N., & Vehkamäki, H. (2022). A study on the fragmentation of sulfuric acid and dimethylamine clusters inside an atmospheric pressure interface time-of-flight mass spectrometer. Atmospheric Measurement Techniques. [[doi]](https://doi.org/10.5194/amt-15-11-2022)

    See also the `README.md` file in this directory.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    paths = mo.md('''
        Give the full paths to the data files below:

        {database}

        {original_results}
    ''').batch(
        database=mo.ui.text(value="", label="Database path"),
        original_results=mo.ui.text(value="", label="Original results CSV"),
    )
    paths
    return (paths,)


@app.cell(hide_code=True)
def _():
    import pandas as pd
    from apitofsim.workflow import ExperimentDatabase
    from apitofsim.plotting import get_joint_survivals, plot_spectrogram, get_intensities_multipathway, get_intensities_singlepathway
    import holoviews as hv
    import numpy as np

    return (
        ExperimentDatabase,
        get_intensities_multipathway,
        get_intensities_singlepathway,
        get_joint_survivals,
        hv,
        np,
        pd,
        plot_spectrogram,
    )


@app.cell(hide_code=True)
def _():
    TRANSMISSIONS = {
        "1B": 0.01691,
        "1S1B": 0.01881,
        "2S1B": 0.01961,
        "1D2S1B": 0.0196,
        "1D3S1B": 0.01876,
        "2D1S1B": 0.01959,
        "2D2S1B": 0.01935,
        "2D3S1B": 0.01802,
        "3D3S1B": 0.01703,
        "3D4S1B": 0.01406,
        "4D4S1B": 0.01236,
        "1D1S1B": 0.01933,
    }
    return (TRANSMISSIONS,)


@app.cell(hide_code=True)
def _():
    def mk_name_converter(order, lookup):
        def converter(cluster):
            from itertools import batched
            bits = batched(cluster, 2)
            bits = [bit[0] + lookup.get(bit[1], bit[1]) for bit in bits]
            return "".join(sorted(bits, key=lambda x: order.index(x[1])))
        return converter

    paper2dataset = mk_name_converter(["B", "A", "D"], {"S": "A"})
    dataset2paper = mk_name_converter(["D", "S", "B"], {"A": "S"})

    IMPORT_CONFIGS = ["nodat", "pragdat", "alldat"]
    return IMPORT_CONFIGS, dataset2paper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison of survival rates

    Here we look at the survival rates as produced by the simulation you have run as well as the simulation and experimental results from the paper.

    The final plot produced is comparable with Figure 5 from https://doi.org/10.5194/amt-15-11-2022
    """)
    return


@app.cell(hide_code=True)
def _(ExperimentDatabase, IMPORT_CONFIGS, mo, paths):
    mo.stop(paths.value is None or any((v == "" for v in paths.value.values())), mo.md("**Enter all paths above to continue**"))

    def pathway_type_lbl(is_single_pathway):
        return "single pathway" if is_single_pathway else "multi-pathway"

    def get_experiment_choices():
        experiment_choices = {}
        df = db.report_df("experiment_summary")
        for row in df.itertuples():
            pathway_desc = pathway_type_lbl(row.is_single_pathway)
            label = (
                f"#{row.experiment_run_id} {row.config_name} run at {row.start_time} "
                f"({pathway_desc}, success rate: {row.successes}/{row.successes + row.failures})"
            )
            value = (row.experiment_run_id, row.config_name, row.is_single_pathway)
            experiment_choices[label] = value
        return experiment_choices

    db = ExperimentDatabase(paths.value["database"], readonly=True)
    experiment_multiselect = mo.ui.multiselect(
        options=get_experiment_choices(), label="Select experiments"
    )
    import_configs_multiselect = mo.ui.multiselect(
        options=IMPORT_CONFIGS, label="Select import configs"
    )

    selections = mo.md('''
        Select which experiments and import configs to plot:

        {experiment_multiselect}

        {import_configs_multiselect}
    ''').batch(
        experiment_multiselect=experiment_multiselect,
        import_configs_multiselect=import_configs_multiselect,
    )
    selections
    return db, get_experiment_choices, pathway_type_lbl, selections


@app.cell(hide_code=True)
def _(dataset2paper, db, np, paths, pd):
    mass_lookup = {}
    df = db.db.execute("select common_name, atomic_mass from cluster;").fetchdf()
    for cluster, mass in df.itertuples(index=False):
        if cluster.startswith("nodat_"):
            cluster = cluster[len("nodat_") :]
        else:
            continue
        mass_lookup[dataset2paper(cluster)] = mass

    original_results = pd.read_csv(paths.value["original_results"])
    original_results.ffill(inplace=True)
    original_results["survived"] = original_results["Main Cluster"] == original_results["Cluster + Fragment"]
    original_results.rename(columns={
        "Main Cluster": "parent",
        "Cluster + Fragment": "product",
        "Survival Probability – Experiment": "survival_raw_experiment",
        "Overall Survival Probability – Experiment": "survival_overall_experiment",
        "Overall Survival Probability - Model for single pathway": "survival_raw_model",
        "Overall Survival Probability - Model recalculated for multiple pathways": "survival_overall_model",
        "Signal": "signal",
        "Transmission Corrected Signal": "signal_corrected"
    }, inplace=True, errors="raise")
    original_results.replace('-', np.nan, inplace=True)
    original_results = original_results.astype({
        "survival_raw_experiment": "float64",
        "survival_overall_experiment": "float64",
        "survival_raw_model": "float64",
        "survival_overall_model": "float64",
    })
    original_results["product_mass"] = [mass_lookup[name] for name in original_results["product"]]
    return (original_results,)


@app.cell(hide_code=True)
def _(mo, original_results):
    original_clusters = original_results[original_results["survived"]]

    original_results_multiselect = mo.ui.multiselect(
        options=[
            "survival_raw_experiment",
            "survival_overall_experiment",
            "survival_raw_model",
            "survival_overall_model"
        ],
        label="Select results from paper to plot"
    )

    original_results_multiselect
    return original_clusters, original_results_multiselect


@app.cell(hide_code=True)
def _(
    dataset2paper,
    db,
    get_joint_survivals,
    original_clusters,
    original_results_multiselect,
    pathway_type_lbl,
    pd,
    selections,
):
    def build_survivals():
        survivals_df = {"name": [], "cluster": [], "survival": []}
        for experiment_id, config_name, is_single_pathway in selections["experiment_multiselect"].value:
            for k, v in get_joint_survivals(db, experiment_id).items():
                import_config, cluster_name = k.split("_", 1)
                if import_config not in selections["import_configs_multiselect"].value:
                    continue
                if "__" in cluster_name:
                    cluster_name = cluster_name.rsplit("__", 1)[0]
                survivals_df["name"].append(f"{import_config} #{experiment_id} {config_name} {pathway_type_lbl(is_single_pathway)}")
                survivals_df["cluster"].append(dataset2paper(cluster_name))
                survivals_df["survival"].append(v)
        for _, row in original_clusters.iterrows():
            for name in original_results_multiselect.value:
                survivals_df["name"].append(name)
                survivals_df["cluster"].append(row["parent"])
                survivals_df["survival"].append(row[name])
        survivals_df = pd.DataFrame(survivals_df)
        survivals_df.sort_values(["cluster", "name"], inplace=True)
        return survivals_df

    survivals_df = build_survivals()
    return (survivals_df,)


@app.cell(hide_code=True)
def _(hv, mo, original_results_multiselect, selections, survivals_df):
    mo.stop(not (selections["experiment_multiselect"].value and selections["import_configs_multiselect"].value) and not original_results_multiselect.value, mo.md("**Pick configs above to see the survival plot**"))
    bars = hv.Bars(survivals_df, kdims=['cluster', 'name'])
    bars.opts(width=1000, height=1000, multi_level=False, legend_position='bottom', legend_cols=2)
    bars
    return


@app.cell(hide_code=True)
def _(IMPORT_CONFIGS, get_experiment_choices, mo, survivals_df):
    cluster_selection = mo.md('''
        Make your selections for viewing spectrograms:

        {cluster_dropdown}

        {experiment_dropdown}

        {transmission_checkbox}
    ''').batch(
        cluster_dropdown=mo.ui.dropdown(
            options=survivals_df["cluster"], label="Cluster"
        ),
        experiment_dropdown= mo.ui.multiselect(
            {
                f"{config} {experiment}": (*experiment_info, config, f"{config} {experiment}")
                for config in IMPORT_CONFIGS
                for experiment, experiment_info in get_experiment_choices().items()
            },
            label="Experiment / Import config"
        ),
        transmission_checkbox=mo.ui.checkbox(label="Consider transmission", value=True),
    )
    cluster_selection
    return (cluster_selection,)


@app.cell(hide_code=True)
def _(
    TRANSMISSIONS,
    cluster_selection,
    dataset2paper,
    db,
    get_intensities_multipathway,
    get_intensities_singlepathway,
    hv,
    mo,
    original_results,
    pd,
    plot_spectrogram,
):
    mo.stop(len(cluster_selection.value["experiment_dropdown"]) == 0, mo.md("**Select an experiment above to continue**"))

    max_x = original_results["product_mass"].max() * 1.1

    @mo.cache
    def get_real_spectrogram(cluster, use_transmission):
        if use_transmission:
            signal_col = "signal"
        else:
            signal_col = "signal_corrected"
        current_cluster_original_df = original_results[original_results["parent"] == cluster]
        spectrogram_df = pd.DataFrame(
            {
                "parent": current_cluster_original_df["parent"],
                "cluster": current_cluster_original_df["product"],
                "intensity": current_cluster_original_df[signal_col],
                "atomic_mass": current_cluster_original_df["product_mass"],
            }
        )
        return plot_spectrogram(spectrogram_df, scale="max", max_x=max_x)

    @mo.cache
    def get_model_spectrogram(cluster, experiment_id, is_single_pathway, import_config, use_transmission):
        if is_single_pathway:
            intensities_df = get_intensities_singlepathway(db, experiment_id)
        else:
            intensities_df = get_intensities_multipathway(db, experiment_id)
        intensities_df = intensities_df[intensities_df["parent_name"].map(lambda x: x.startswith(import_config + "_"))]
        for col in ("parent_name", "product_name"):
            intensities_df[col] = intensities_df[col].map(lambda k: dataset2paper(k.split("_", 1)[1]))
        intensities_df = intensities_df[intensities_df["parent_name"] == cluster]
        if use_transmission:
            intensities_df["intensity"] *= intensities_df["product_name"].map(TRANSMISSIONS.get)
        return plot_spectrogram(intensities_df, scale="max", max_x=max_x)

    spectrograms = [get_real_spectrogram(cluster_selection["cluster_dropdown"].value, cluster_selection["transmission_checkbox"].value)]
    spectrogram_config_labels = []
    for experiment_id, _, is_single_pathway, import_config, label in cluster_selection.value["experiment_dropdown"]:
        model_spectogram = get_model_spectrogram(cluster_selection["cluster_dropdown"].value, experiment_id, is_single_pathway, import_config, cluster_selection["transmission_checkbox"].value)
        spectrograms.append(model_spectogram)
        spectrogram_config_labels.append(label)

    def spectogram_grid(spectrograms):
        grid = hv.Layout([spectrograms[0].relabel('Experimental')])
        for label, spectogram in zip(spectrogram_config_labels, spectrograms[1:]):
            grid += spectogram.options(yaxis=None).relabel(label)
        for spectogram in spectrograms[1:]:
            grid += spectogram
            for _ in range(len(spectrograms) - 1):
                grid += hv.Empty()
        return grid.cols(len(spectrograms)).opts(shared_axes=True)

    spectogram_grid(spectrograms)
    return


if __name__ == "__main__":
    app.run()
