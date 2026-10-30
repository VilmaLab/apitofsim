"""Persisted execution identity and read-only histories (no legacy inference)."""

from types import SimpleNamespace

import duckdb
import pytest
from apitofsim.workflow import (
    ExperimentRunner,
    RealizationDatabase,
    sql_files,
    sql_files_nofk,
)


def populate(db):
    db.execute(
        "insert into cluster (id, common_name) values (10, 'root'), (20, 'left'), (30, 'right'), (40, 'shared')"
    )
    db.execute(
        "insert into pathway values (1, 10, 20, 30), (2, 10, 30, 20), (3, 20, 40, 30), (4, 30, 40, 20), (5, 40, 20, 30), (6, 10, 40, 20)"
    )


def tree():
    # Two routes to species 40 retain separate occurrences, 3 and 4.
    return SimpleNamespace(
        tree_nodes=[
            SimpleNamespace(payload_idx=p, pathway_indices=indices)
            for p, indices in [(0, [0, 1]), (1, [2]), (2, [3]), (3, [4]), (3, [5])]
        ],
        tree_pathways=[
            SimpleNamespace(product_idx=i) for i in [1, 2, 3, 4, None, None]
        ],
    )


def start(db):
    return db.start_tree_result(
        1, 10, tree(), {0: 10, 1: 20, 2: 30, 3: 40}, [1, 2, 3, 4, 5, 5]
    )


def event(db, realization, kind, node, transition=None, destination=None, pathway=None):
    event_id = db.execute(
        """insert into event_info values (default, ?, ?,
        {x:0,y:0,z:0,t:0}, {x:0,y:0,z:0}, {x:0,y:0,z:0}, 0, 0, ?) returning id""",
        [realization, kind, node],
    ).fetchone()[0]
    if kind == "fragmentation" and transition is not None:
        db.execute(
            "insert into fragmentation_event values (?, ?, ?, ?)",
            [event_id, transition, destination, pathway],
        )
    return event_id


@pytest.mark.parametrize("attached", [False, True])
def test_tree_history_reader(tmp_path, attached):
    filename = str(tmp_path / "results.duckdb")
    if attached:
        clusters = str(tmp_path / "clusters.duckdb")
        with duckdb.connect(clusters) as conn:
            conn.execute(sql_files.pathway)
            populate(conn)
        conn = duckdb.connect()
        conn.execute(f"attach '{clusters}' as molecules (read_only)")
        conn.execute(f"attach '{filename}' as output")
        conn.execute("use output; set search_path = 'output,molecules'")
        conn.execute(sql_files_nofk.experiment)
        conn.execute(sql_files_nofk.realizations)
        db = RealizationDatabase((conn, None))
    else:
        db = RealizationDatabase(filename)
        db.create_tables()
        populate(db.db)
    conn = db.db
    db.refresh_views()
    conn.execute("insert into experiment_config (id) values (1)")
    conn.execute(
        "insert into experiment_run (id, experiment_config_id, simulation_mode) values (1, 1, 'CLUSTER_TREE'), (2, 1, 'CLUSTER_TREE')"
    )
    first, second = start(db), start(db)
    conn.execute(
        "update cluster_tree_experiment_result set experiment_run_id = 2 where id = ?",
        [second],
    )
    conn.execute(
        "insert into realization values (1, ?), (2, ?), (3, ?)", [first, first, second]
    )
    for realization, nodes, transitions, paths in [
        (1, [0, 1, 3], [0, 2, 4], [1, 3, 5]),
        (2, [0, 2, 4], [1, 3, 5], [2, 4, 5]),
    ]:
        event(conn, realization, "init", 0)
        for node, transition, path, destination in zip(
            nodes, transitions, paths, [*nodes[1:], -1], strict=True
        ):
            event(conn, realization, "collision", node)
            event(conn, realization, "postcollision", node)
            event(
                conn, realization, "fragmentation", node, transition, destination, path
            )
    event(conn, 3, "init", 0)
    event(conn, 3, "fragmentation", 0, 1, 2, 2)
    event(conn, 3, "escape", 2)
    # A complete split at a continuing node is still an interrupted history.
    conn.execute("insert into realization values (4, ?)", [second])
    event(conn, 4, "init", 0)
    event(conn, 4, "fragmentation", 0, 0, 1, 1)

    def check(database):
        histories = database.fragmentation_histories_query(result_id=first)
        assert histories.select(
            "realization_id, particle_index, current_cluster_id, next_particle_index, next_cluster_id"
        ).fetchall() == [
            (1, 0, 10, 1, 20),
            (1, 1, 20, 3, 40),
            (1, 3, 40, -1, None),
            (2, 0, 10, 2, 30),
            (2, 2, 30, 4, 40),
            (2, 4, 40, -1, None),
        ]
        events = database.tree_events_query(result_id=first)
        assert events.aggregate("count(*), count(distinct event_id)").fetchone() == (
            20,
            20,
        )
        assert (
            events.filter(
                "provenance != 'persisted' or not continuity_valid"
            ).fetchall()
            == []
        )
        assert database.tree_events_query(run_id=1, result_id=second).fetchall() == []
        assert database.tree_events_query(parent=20).fetchall() == []
        assert database.tree_events_query(run_id=2).aggregate(
            "count(*)"
        ).fetchone() == (5,)
        assert database.tree_events_query(result_id=second).filter(
            "event_type = 'escape'"
        ).select("current_cluster_id").fetchone() == (30,)
        assert database.fragmentation_histories_query(result_id=second).select(
            "next_particle_index"
        ).fetchall() == [(2,), (1,)]
        # Enabled subset is persisted, including the entirely unobserved route in second.
        assert database.db.execute(
            "select count(*) from cluster_tree_transition where experiment_result_id = ?",
            [second],
        ).fetchone() == (6,)
        assert database.db.execute(
            "select count(*) from cluster_tree_transition where pathway_id = 6"
        ).fetchone() == (0,)
        assert database.experiment_overview_query(1).project(
            "successes"
        ).fetchone() == (0,)

    check(db)
    db.close()
    before = (tmp_path / "results.duckdb").read_bytes()
    if attached:
        conn = duckdb.connect()
        conn.execute(f"attach '{clusters}' as molecules (read_only)")
        conn.execute(f"attach '{filename}' as output (read_only)")
        conn.execute("use output; set search_path = 'output,molecules'")
    else:
        conn = duckdb.connect(filename, read_only=True)
    readonly = RealizationDatabase((conn, None))
    check(readonly)
    readonly.close()
    assert (tmp_path / "results.duckdb").read_bytes() == before


def test_missing_and_inconsistent_tree_provenance():
    db = RealizationDatabase(":memory:")
    db.create_tables()
    populate(db.db)
    db.refresh_views()
    conn = db.db
    conn.execute("insert into experiment_config (id) values (1)")
    conn.execute(
        "insert into experiment_run (id, experiment_config_id, simulation_mode) values (1, 1, 'CLUSTER_TREE')"
    )
    result = start(db)
    conn.execute("insert into realization values (1, ?)", [result])
    event(conn, 1, "init", 0)
    event(conn, 1, "fragmentation", 0)
    event(conn, 1, "fragmentation", 0, 0, 1, None)
    event(conn, 1, "fragmentation", 0, 99, 1, 1)
    event(conn, 1, "fragmentation", 0, 2, 3, 3)
    event(conn, 1, "fragmentation", 0, 0, None, 1)
    event(conn, 1, "fragmentation", 0, 0, -1, 1)
    event(conn, 1, "escape", 99)
    assert db.tree_events_query().select("provenance").fetchall() == [
        (p,)
        for p in [
            "persisted",
            "missing_subtype",
            "missing_pathway",
            "missing_transition",
            "inconsistent_transition",
            "missing_destination",
            "inconsistent_destination",
            "missing_node",
        ]
    ]
    assert db.tree_events_query().select("continuity_valid").fetchall()[-1] == (False,)
    db.close()


def test_tree_failure_keeps_early_identity(monkeypatch):
    from apitofsim import api

    db = RealizationDatabase(":memory:")
    db.create_tables()
    populate(db.db)
    db.refresh_views()
    db.db.execute("insert into experiment_config (id) values (1)")
    runner = ExperimentRunner(db)
    runner.start_run(1)

    def fail(*args, event_callback, **kwargs):
        assert event_callback.result_id is not None
        db.db.execute(
            "insert into realization values (1, ?)", [event_callback.result_id]
        )
        event(db.db, 1, "init", 0)
        raise RuntimeError("interrupted execution")

    monkeypatch.setattr(api, "mass_spec", fail)
    with pytest.raises(RuntimeError, match="interrupted execution"):
        runner.run_mass_spec(
            None,
            tree(),
            cluster_id=10,
            pathway_ids=[1, 2, 3, 4, 5, 5],
            tree_cluster_ids={0: 10, 1: 20, 2: 30, 3: 40},
        )
    assert db.tree_events_query().select(
        "current_cluster_id, provenance"
    ).fetchall() == [(10, "persisted")]
    assert db.db.execute(
        "select total_us, failure_id is not null from cluster_tree_experiment_result"
    ).fetchone() == (None, True)
    assert db.experiment_overview_query().project("successes, failures").fetchone() == (
        0,
        1,
    )
    db.close()


def test_tree_completion_and_distinct_roots():
    db = RealizationDatabase(":memory:")
    db.create_tables()
    populate(db.db)
    db.refresh_views()
    conn = db.db
    conn.execute("insert into experiment_config (id) values (1)")
    conn.execute(
        "insert into experiment_run (id, experiment_config_id, simulation_mode) values (1, 1, 'CLUSTER_TREE')"
    )
    first = start(db)
    # A different root also uses local node 0, without sharing its identity.
    other = db.start_tree_result(
        1,
        20,
        SimpleNamespace(
            tree_nodes=[SimpleNamespace(payload_idx=0, pathway_indices=[])],
            tree_pathways=[],
        ),
        {0: 20},
        [],
    )
    conn.execute("insert into realization values (1, ?), (2, ?)", [first, other])
    event(conn, 1, "init", 0)
    event(conn, 2, "init", 0)
    event(conn, 2, "escape", 0)
    counters = SimpleNamespace(
        nwarnings=0,
        n_escaped_total=10,
        n_fragmented_total=[0] * 6,
        ncoll_total=20,
        counter_collision_rejections=0,
    )
    assert (
        db.record_result(
            1,
            counters,
            cluster_id=10,
            pathway_ids=[1, 2, 3, 4, 5, 5],
            tree_result_id=first,
        )
        == first
    )
    assert db.tree_events_query().select(
        "realization_id, current_cluster_id"
    ).fetchall() == [(1, 10), (2, 20), (2, 20)]
    assert db.tree_events_query(parent=20).aggregate("count(*)").fetchone() == (2,)
    assert conn.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (0,)
    assert conn.execute(
        "select count(*), sum(count) from pathway_fragmentation where experiment_result_id = ?",
        [first],
    ).fetchone() == (5, 0)
    assert db.experiment_overview_query().project("successes, failures").fetchone() == (
        1,
        0,
    )
    assert db.report_df("experiment_cluster_report").survival_rate.tolist() == [1.0]
    assert len(db.report_df("experiment_pathway_report")) == 5
    db.close()
