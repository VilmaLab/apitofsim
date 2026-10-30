import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Besel 2020: survival rates by simulation mode

    Compare `single-cluster`, `pathway-at-a-time`, and `cluster-tree` runs
    from the same experiment database. Run `./run.sh` in this directory first.

    Survival rates come from `get_joint_survivals`, as in the validation
    notebook: pathway-at-a-time rates are combined as a product of pathway
    survival probabilities. The other modes use the fraction of clusters
    escaping without fragmentation.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from apitofsim.plotting.survivals import get_joint_survivals
    from apitofsim.workflow import ExperimentDatabase

    return ExperimentDatabase, Path, get_joint_survivals, mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    database_path = mo.ui.text(
        value=str(mo.notebook_dir() / "database.duckdb"),
        label="Database path",
        full_width=True,
    )
    database_path
    return (database_path,)


@app.cell(hide_code=True)
def _(ExperimentDatabase, Path, database_path, get_joint_survivals, mo, pd):
    mo.stop(
        not Path(database_path.value).is_file(),
        mo.md("**Enter the path to an existing experiment database.**"),
    )
    _db = ExperimentDatabase(database_path.value, readonly=True)
    try:
        runs = _db.db.execute("""
            select summary.* exclude (simulation_mode),
                replace(lower(summary.simulation_mode::varchar), '_', '-') as simulation_mode
            from experiment_summary as summary
            where summary.simulation_mode in
                ('SINGLE_CLUSTER', 'PATHWAY_AT_A_TIME', 'CLUSTER_TREE')
            order by summary.start_time, summary.experiment_run_id
        """).fetchdf()
        survivals = pd.DataFrame(
            [
                (row.experiment_run_id, cluster, survival)
                for row in runs.itertuples()
                for cluster, survival in get_joint_survivals(
                    _db, row.experiment_run_id
                ).items()
            ],
            columns=["experiment_run_id", "cluster", "survival"],
        )
    finally:
        _db.close()
    return runs, survivals


@app.cell(hide_code=True)
def _(mo, runs):
    mo.stop(runs.empty, mo.md("**No runs for the three simulation modes were found.**"))
    _choices = {
        f"#{row.experiment_run_id} {row.config_name} · {row.simulation_mode} · {row.start_time} "
        f"({row.successes} successes, {row.failures} failures)": row.experiment_run_id
        for row in runs.itertuples()
    }
    _latest_ids = set(
        runs[runs.successes > 0]
        .drop_duplicates(["config_name", "simulation_mode"], keep="last")
        .experiment_run_id
    )
    experiments = mo.ui.multiselect(
        options=_choices,
        value=[label for label, run_id in _choices.items() if run_id in _latest_ids],
        label="Runs (defaults to the latest with results per config and mode)",
        full_width=True,
    )
    experiments
    return (experiments,)


@app.cell(hide_code=True)
def _(experiments, mo, runs, survivals):
    mo.stop(not experiments.value, mo.md("**Select runs above to compare survival rates.**"))
    _selected = runs[runs.experiment_run_id.isin(experiments.value)].copy()
    _selected["run"] = [
        f"{row.simulation_mode} · #{row.experiment_run_id} {row.config_name}"
        for row in _selected.itertuples()
    ]
    comparison = (
        survivals.merge(_selected[["experiment_run_id", "run"]], on="experiment_run_id")
        .pivot(index="cluster", columns="run", values="survival")
        .sort_index()
    )
    mo.stop(comparison.empty, mo.md("**The selected runs have no survival results.**"))
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison, plt):
    _fig, _ax = plt.subplots(figsize=(max(12, len(comparison) * 0.6), 6))
    comparison.plot.bar(ax=_ax, width=0.8)
    _ax.set(xlabel="Cluster", ylabel="Survival probability", ylim=(0, 1))
    _ax.set_axisbelow(True)
    _ax.yaxis.grid(True, alpha=0.3)
    _ax.legend(title="Simulation mode / run", loc="upper left", bbox_to_anchor=(1, 1))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(comparison, mo):
    mo.vstack([
        mo.md("Missing results remain blank; they are not treated as zero survival."),
        mo.ui.table(comparison.reset_index(), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
