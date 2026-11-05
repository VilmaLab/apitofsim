from itertools import permutations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apitofsim.plotting.events import (
    PANELS,
    plot_events_from_df,
)
from apitofsim.workflow.db import ExperimentDatabase, connection_scope


@pytest.fixture
def events():
    # Includes tied positions, an internal boundary and the final endpoint.
    rows = [
        (1, "fragmentation", 0.001, 1),
        (2, "fragmentation", 0.001, 1),
        (3, "fragmentation", 0.005, 1),
        (4, "escape", 0.009, None),
        (5, "fragmentation", 0.002, 1),
        (5, "escape", 0.005, None),  # Ambiguous.
        (6, "fragmentation", 0.002, 1),
        (6, "collision", 0.003, None),  # Terminal event wasn't last.
        (7, "collision", 0.002, None),  # Incomplete.
        (8, "fragmentation", 0.002, None),  # Missing pathway.
    ]
    df = pd.DataFrame(rows, columns=["realization_id", "event_type", "z", "pathway_id"])
    df["event_id"] = np.arange(len(df))
    df["t"] = np.arange(len(df))
    return df


@pytest.mark.parametrize("rescale", ["none", "equal", "schematic"])
def test_outcome_fractions(events, rescale):
    before = events.copy(deep=True)
    fig = plot_events_from_df(
        events, np.arange(6) * 0.001, rescale=rescale, realization_ids=range(1, 11)
    )
    schematic, cumulative, bars = fig.axes
    assert [bar.get_height() for bar in bars.patches] == pytest.approx(
        [0, 0.2, 0, 0, 0.1, 0.1]
    )
    areas = [area.get_paths()[0].vertices for area in cumulative.collections]
    assert areas[0][:, 1].max() == pytest.approx(0.3)
    assert areas[1][:, 1].max() == pytest.approx(0.4)
    endpoint = bars.patches[-1].get_x()
    assert areas[1][areas[1][:, 0] < endpoint, 1].max() == pytest.approx(0.2)
    assert schematic.get_xlim() == cumulative.get_xlim() == bars.get_xlim()
    pd.testing.assert_frame_equal(events, before)
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize(
    "panels", [p for n in (1, 2, 3) for p in permutations(PANELS, n)]
)
def test_panel_selection(events, panels):
    fig = plot_events_from_df(events, np.arange(6) * 0.001, panels=panels)
    assert tuple(ax.get_label() for ax in fig.axes) == panels
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize(
    "panels", [(), ("bars", "bars"), ("unknown",), PANELS + ("bars",)]
)
def test_invalid_panels(events, panels):
    with pytest.raises(ValueError, match="panels"):
        plot_events_from_df(events, np.arange(6) * 0.001, panels=panels)


def test_empty_cohort(events):
    fig = plot_events_from_df(events.iloc[:0], np.arange(6) * 0.001)
    assert all(bar.get_height() == 0 for bar in fig.axes[-1].patches)
    fig.canvas.draw()
    plt.close(fig)


def test_run_scope_and_missing_histories(monkeypatch, tmp_path):
    from archive_fixtures import publish_events

    with connection_scope(ExperimentDatabase, tmp_path / "plot.duckdb") as db:
        db.create_tables()
        db.insert_cluster("parent", 100, -1, 0, [1, 1, 1], [])
        config_id = db.insert_config("events", "{}")
        run_id = db.insert_run(config_id)
        result = db.begin_execution(run_id, 1, None, [], recorded=True)
        publish_events(
            db,
            [
                dict(
                    result_id=result,
                    realization_index=0,
                    event_index=0,
                    event_kind=5,
                    z=0.005,
                    t=1,
                )
            ],
        )
        assert db.events_query(run_id=run_id).count("*").fetchone() == (1,)
        assert db.events_query(run_id=999).count("*").fetchone() == (0,)
