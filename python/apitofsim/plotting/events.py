"""Matplotlib cumulative outcomes, regional bars and instrument schematic."""

PANELS = ("schematic", "cumulative", "bars")
REGIONS = ("First chamber", "Skimmer", "C2 pre", "Quadrupole", "C2 post")
COLORS = {"fragmentation": "#d62728", "escape": "#2ca02c"}


def get_geometry(db, experiment_run_id):
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


def get_events(db, experiment_run_id, is_single_pathway=False, cluster_id=None):
    if is_single_pathway:
        raise ValueError(
            "Event plotting is not supported for single pathway experiments"
        )
    events = db.events_query(run_id=experiment_run_id, parent=cluster_id).fetchdf()
    names = (
        db.clusters_query()
        .select("id as cluster_id, common_name as parent_name")
        .fetchdf()
    )
    return events.merge(names, on="cluster_id", how="left")


def _terminal_events(df, boundaries, realization_ids):
    """Accept one terminal event per history, only if it is the last event."""
    import numpy as np

    events = df.sort_values(["realization_id", "t", "event_id"])
    if realization_ids is None:
        realization_ids = events["realization_id"].unique()
    events = events[events["realization_id"].isin(realization_ids)]
    terminal = events["event_type"].isin(COLORS)
    counts = terminal.groupby(events["realization_id"]).sum()
    last = events.groupby("realization_id", sort=False).tail(1).copy()
    last = last[
        last["realization_id"].map(counts).eq(1) & last["event_type"].isin(COLORS)
    ].copy()
    last["z"] = last["z"].clip(lower=0)
    region = np.searchsorted(boundaries, last["z"], side="right") - 1
    region = np.where(last["z"] == boundaries[-1], len(boundaries) - 2, region)
    last["region"] = region
    valid_fragmentation = last["pathway_id"].notna() & last["z"].between(
        boundaries[0], boundaries[-1]
    )
    return last[last["event_type"].eq("escape") | valid_fragmentation], len(
        set(realization_ids)
    )


def _draw_schematic(ax, boundaries, escape_end):
    from matplotlib.patches import Rectangle

    def box(lo, hi, label, color, bottom=0.12, height=0.76):
        ax.add_patch(
            Rectangle(
                (lo, bottom), hi - lo, height, facecolor=color, edgecolor="#64748b"
            )
        )
        ax.text(
            (lo + hi) / 2,
            bottom + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

    box(boundaries[0], boundaries[1], "First\nchamber", "#dbeafe")
    box(boundaries[1], boundaries[2], "Skimmer", "#e2e8f0")
    box(boundaries[2], boundaries[-1], "", "#dbeafe")
    for i in (2, 4):
        ax.text(
            (boundaries[i] + boundaries[i + 1]) / 2,
            0.5,
            REGIONS[i].replace(" ", "\n"),
            ha="center",
            va="center",
            fontsize=9,
        )
    box(boundaries[3], boundaries[4], "Quad", "#bfdbfe", 0.196, 0.608)
    box(boundaries[-1], escape_end, "Escaped", "#f8fafc")
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def plot_events_from_df(
    cluster_df,
    cumulative_lengths,
    rescale="schematic",
    panels=PANELS,
    *,
    realization_ids=None,
):
    """Plot any 1–3 panels in the given top-to-bottom order.

    ``panels`` contains distinct names from ``schematic``, ``cumulative``,
    and ``bars``. Coordinates are physical (``none``, in mm), equal-width
    regions (``equal``), or explorer chamber weights (``schematic``).
    Fractions use all recorded realizations, including incomplete/ambiguous
    histories. Supply ``realization_ids`` to include histories with no events.
    Cumulative outcomes are stacked; escapes occur at the instrument endpoint.
    Bars show fragmentation per region and escapes in a separate end slot.
    The input frame is never modified.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    panels = tuple(panels)
    if (
        not 1 <= len(panels) <= 3
        or len(set(panels)) != len(panels)
        or any(panel not in PANELS for panel in panels)
    ):
        raise ValueError(f"panels must contain 1–3 distinct names from {PANELS}")
    physical = np.asarray(cumulative_lengths, dtype=float)
    if (
        physical.shape != (6,)
        or not np.isfinite(physical).all()
        or np.any(np.diff(physical) <= 0)
    ):
        raise ValueError("Expected six finite, strictly increasing region boundaries")
    if rescale == "none":
        boundaries = physical * 1000
    elif rescale in ("equal", "schematic"):
        weights = [1] * 5 if rescale == "equal" else [3, 3, 1, 1, 1]
        boundaries = np.r_[0, np.cumsum(weights)].astype(float)
    else:
        raise ValueError(f"Unknown rescale mode {rescale}")
    terminals, n = _terminal_events(cluster_df, physical, realization_ids)
    terminals = terminals.copy()
    terminals["position"] = np.interp(terminals["z"], physical, boundaries)
    terminals.loc[terminals["event_type"] == "escape", "position"] = boundaries[-1]
    escape_end = boundaries[-1] + max(
        np.diff(boundaries)[-1] * 0.65, 0.5 if rescale != "none" else 0
    )
    edges = np.r_[boundaries, escape_end]
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        squeeze=False,
        figsize=(10, sum(0.9 if p == "schematic" else 2.6 for p in panels)),
        height_ratios=[0.9 if p == "schematic" else 2.6 for p in panels],
        layout="constrained",
    )
    for panel, ax in zip(panels, axes[:, 0], strict=True):
        ax.set_label(panel)
        if panel == "schematic":
            _draw_schematic(ax, boundaries, escape_end)
            continue
        if panel == "cumulative":
            positions = np.sort(terminals["position"].unique())
            x = np.r_[boundaries[0], np.repeat(positions, 2), escape_end]
            bottom = np.zeros(len(x))
            for kind, color in COLORS.items():
                counts = (
                    terminals.loc[terminals["event_type"] == kind]
                    .groupby("position")
                    .size()
                    .reindex(positions, fill_value=0)
                    .to_numpy()
                )
                cumulative = np.r_[0, counts.cumsum()]
                fraction = np.r_[np.repeat(cumulative, 2)] / max(n, 1)
                ax.fill_between(
                    x,
                    bottom,
                    bottom + fraction,
                    color=color,
                    alpha=0.85,
                    label=kind.capitalize(),
                )
                bottom += fraction
            ax.set_ylim(0, 1)
        else:
            fragments = terminals[terminals["event_type"] == "fragmentation"]
            counts = fragments.groupby("region").size().reindex(range(5), fill_value=0)
            values = np.r_[
                counts.to_numpy(), (terminals["event_type"] == "escape").sum()
            ]
            ax.bar(
                edges[:-1],
                values / max(n, 1),
                width=np.diff(edges),
                align="edge",
                color=[COLORS["fragmentation"]] * 5 + [COLORS["escape"]],
                alpha=0.65,
                edgecolor="white",
            )
            ax.set_ylim(bottom=0, top=max(values.max() / max(n, 1) * 1.12, 0.05))
        ax.set_ylabel("Fraction")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for boundary in boundaries:
            ax.axvline(boundary, color="gray", linewidth=0.6, alpha=0.4)
        ax.set_xticks((edges[:-1] + edges[1:]) / 2, [*REGIONS, "Escaped"])
        ax.tick_params(axis="x", labelbottom=True, labelsize=9)
        if panel == "cumulative":
            ax.legend(loc="upper left")
    axes[0, 0].set_xlim(edges[0], edges[-1])
    if rescale == "none":
        # Keep zone labels on the panels and expose physical distances separately.
        ax = next((a for p, a in zip(panels, axes[:, 0]) if p != "schematic"), None)
        if ax is not None:
            physical_axis = ax.secondary_xaxis("top")
            physical_axis.set_xlabel("Axial distance (mm)")
    return fig


def plot_events_experiment(
    db,
    experiment_id,
    is_single_pathway=False,
    rescale="schematic",
    panels=PANELS,
):
    """Yield a (parent name, figure) pair for each recorded cluster in a run."""
    cumulative_lengths = lengths_to_cumulative_lengths(get_geometry(db, experiment_id))
    events = get_events(db, experiment_id, is_single_pathway)
    realizations = db.realizations_query(run_id=experiment_id).fetchdf()
    names = dict(db.clusters_query().select("id, common_name").fetchall())
    for cluster_id, members in realizations.groupby("cluster_id"):
        yield (
            names[cluster_id],
            plot_events_from_df(
                events[events["cluster_id"] == cluster_id],
                cumulative_lengths,
                rescale=rescale,
                panels=panels,
                realization_ids=members["realization_id"],
            ),
        )


def plot_events_cluster(
    db,
    experiment_id,
    is_single_pathway,
    cluster_id,
    rescale="schematic",
    panels=PANELS,
):
    """Plot one recorded cluster cohort."""
    cumulative_lengths = lengths_to_cumulative_lengths(get_geometry(db, experiment_id))
    events = get_events(db, experiment_id, is_single_pathway, cluster_id=cluster_id)
    members = db.realizations_query(run_id=experiment_id, parent=cluster_id).fetchdf()
    return plot_events_from_df(
        events,
        cumulative_lengths,
        rescale=rescale,
        panels=panels,
        realization_ids=members["realization_id"],
    )
