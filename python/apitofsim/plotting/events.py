def get_geometery(db, experiment_run_id):
    import orjson

    from apitofsim.config import import_raw_config

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


def lengths_to_cumulative_lengths(lengths):
    import numpy as np

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
    return np.array(cumulative_lengths)


def get_events(db, experiment_run_id, is_single_pathway, cluster_id=None):
    assert not is_single_pathway
    if cluster_id is not None:
        return db.db.execute(
            "select * from event_report where experiment_result_id = ? and cluster_id = ?",
            (experiment_run_id, cluster_id),
        ).fetchdf()
    else:
        return db.db.execute(
            "select * from event_report where experiment_result_id = ?",
            (experiment_run_id,),
        ).fetchdf()


def prepare_df(df, cumulative_lengths, rescale):
    import numpy as np

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


def plot_events_from_df(cluster_df, cumulative_lengths, rescale, plot_type):
    import numpy as np
    from seaborn import (
        FacetGrid,
        relplot,
        scatterplot,
        stripplot,
        swarmplot,
        violinplot,
    )

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
        raise ValueError(f"Unknown plot type {plot_type}")
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
    if isinstance(ax, FacetGrid):
        fig = ax.figure
    else:
        fig = ax.get_figure(root=True)
        assert fig is not None
    return fig


def plot_events_experiment(db, experiment_id, is_single_pathway, rescale, plot_type):
    cumulative_lengths = lengths_to_cumulative_lengths(get_geometery(db, experiment_id))

    df = get_events(db, experiment_id, is_single_pathway)
    prepare_df(df, cumulative_lengths, rescale)
    for parent_name, cluster_df in df.groupby("parent_name"):
        fig = plot_events_from_df(cluster_df, cumulative_lengths, rescale, plot_type)
        yield parent_name, fig


def plot_events_cluster(
    db, experiment_id, is_single_pathway, cluster_id, rescale, plot_type
):
    cumulative_lengths = lengths_to_cumulative_lengths(get_geometery(db, experiment_id))
    df = get_events(db, experiment_id, is_single_pathway, cluster_id=cluster_id)
    prepare_df(df, cumulative_lengths, rescale)
    return plot_events_from_df(df, cumulative_lengths, rescale, plot_type)
