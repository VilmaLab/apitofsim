"""Persisted execution identity and read-only histories (no legacy inference)."""

from types import SimpleNamespace

import duckdb
import pytest
from apitofsim import apitofsimraw as raw
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


def tree(nodes=None, destinations=None):
    # Two routes to species 40 retain separate occurrences, 3 and 4.
    if nodes is None:
        nodes = [(0, [0, 1]), (1, [2]), (2, [3]), (3, [4]), (3, [5])]
    if destinations is None:
        destinations = [1, 2, 3, 4, None, None]
    return raw.MassSpecSubstanceTreeInput(
        -1,
        raw.Gas(1e-10, 1e-26, 1.4),
        [],
        [],
        [raw.MSSubstanceTreeNode(p, indices) for p, indices in nodes],
        [raw.MSSubstanceTreePathway(0, i) for i in destinations],
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
            if destination >= 0:
                event(conn, realization, "postfragmentation", destination)
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
            24,
            24,
        )
        assert (
            events.filter(
                "provenance != 'persisted' or not continuity_valid"
            ).fetchall()
            == []
        )
        assert events.filter("event_type = 'postfragmentation'").select(
            "realization_id, particle_index, current_cluster_id"
        ).fetchall() == [(1, 1, 20), (1, 3, 40), (2, 2, 30), (2, 4, 40)]
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

    class FailedStream:
        failed = False

        def read_batch(self, max_items=512):
            if self.failed:
                raise StopIteration
            self.failed = True
            row = db.db.execute(
                "select id from cluster_tree_experiment_result"
            ).fetchone()
            assert row is not None
            db.db.execute("insert into realization values (1, ?)", [row[0]])
            event(db.db, 1, "init", 0)
            raise RuntimeError("interrupted execution")

        def request_cancel(self):
            pass

        def join_if_joinable(self):
            pass

    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: FailedStream())
    with pytest.raises(RuntimeError, match="interrupted execution"):
        runner.run_mass_spec(
            None,
            tree(),
            1,
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
        tree(nodes=[(0, [])], destinations=[]),
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


@pytest.mark.parametrize("batch_size", [1, 100])
def test_native_postfragmentation_persistence(batch_size):
    import numpy as np
    from apitofsim import api
    from apitofsim.workflow.db import EventRecorder, connection_scope

    raw = api.apitofsimraw
    energies = np.arange(0.5, 32) * 1e-21
    density = raw.Histogram(energies, np.ones(32))
    rate = raw.Histogram(energies, np.full(32, 1e30))
    subs = raw.MassSpecSubstanceTreeInput(
        -1,
        raw.Gas(2e-10, 4.8e-26, 1.4),
        [
            raw.MSSubstanceTreeCluster(mass, 5e-10, density)
            for mass in [3e-25, 2e-25, 1e-25]
        ],
        [raw.MassSpecInputFragmentationPathway(rate, -1e-21)],
        [raw.MSSubstanceTreeNode(i, [path]) for i, path in enumerate([2, 0, 1])],
        [raw.MSSubstanceTreePathway(0, destination) for destination in [2, None, 1]],
    )
    ms = raw.MassSpectrometer(
        np.array([[100, 300, 100], [100, 300, 100]], dtype=float),
        0.0005,
        np.full(5, 0.001),
        np.zeros(5),
        300,
        np.array([100, 2], dtype=float),
        None,
        None,
    )
    with api.MassSpecIterator(ms, subs, 1, logconf=(0, True)) as stream:
        events = [item for item in stream if hasattr(item, "state")]
    assert [type(e) for e in events] == [
        api.InitEvent,
        api.FragmentationEvent,
        api.PostFragmentationEvent,
        api.FragmentationEvent,
        api.PostFragmentationEvent,
        api.FragmentationEvent,
    ]
    for before, after in [(events[1], events[2]), (events[3], events[4])]:
        np.testing.assert_array_equal(before.state.postime, after.state.postime)
        assert before.state.realization == after.state.realization
        assert after.state.particle_index == before.next_particle_index
        np.testing.assert_array_equal(after.state.velocity, np.zeros(3))
        np.testing.assert_array_equal(after.state.omega, np.zeros(3))
        assert after.state.rot_energy == after.state.vibrational_energy == 0
    assert events[-1].next_particle_index == -1

    with connection_scope(RealizationDatabase, ":memory:") as db:
        db.create_tables()
        populate(db.db)
        db.db.execute("insert into experiment_config (id) values (1)")
        db.db.execute(
            "insert into experiment_run (id, experiment_config_id, simulation_mode) values (1, 1, 'CLUSTER_TREE')"
        )
        paths = [3, 5, 1]
        result_id = db.start_tree_result(1, 10, subs, {0: 10, 1: 20, 2: 40}, paths)
        recorder = EventRecorder(db, paths, result_id=result_id)
        for offset in range(0, len(events), batch_size):
            recorder.record_batch(events[offset : offset + batch_size])
        rows = db.tree_events_query(result_id=result_id)
        assert rows.aggregate("count(*), count(distinct event_id)").fetchone() == (6, 6)
        assert (
            rows.filter("not continuity_valid or provenance != 'persisted'").fetchall()
            == []
        )
        after_rows = rows.filter("event_type = 'postfragmentation'")
        assert after_rows.select("particle_index, current_cluster_id").fetchall() == [
            (1, 20),
            (2, 40),
        ]
        for row, native in zip(
            after_rows.fetchall(), [events[2], events[4]], strict=True
        ):
            values = dict(zip(after_rows.columns, row, strict=True))
            state = native.state
            np.testing.assert_array_equal(
                list(values["postime"].values()), state.postime
            )
            np.testing.assert_array_equal(
                list(values["velocity"].values()), state.velocity
            )
            np.testing.assert_array_equal(list(values["omega"].values()), state.omega)
            assert values["rot_energy"] == state.rot_energy
            assert values["vibrational_energy"] == state.vibrational_energy
            assert (
                values["fragmentation_event_id"] is values["collision_event_id"] is None
            )


def test_tree_persistence_rolls_back_bulk_insert_failure():
    db = RealizationDatabase(":memory:")
    db.create_tables()
    populate(db.db)
    db.db.execute("insert into experiment_config (id) values (1)")
    db.db.execute(
        "insert into experiment_run (id, experiment_config_id, simulation_mode) values (1, 1, 'CLUSTER_TREE')"
    )
    # Node rows are inserted first; the transition FK fails afterward.
    with pytest.raises(duckdb.ConstraintException):
        db.start_tree_result(1, 10, tree(), {0: 10, 1: 20, 2: 30, 3: 40}, [999] * 6)
    for table in [
        "cluster_tree_experiment_result",
        "cluster_tree_node",
        "cluster_tree_transition",
    ]:
        assert db.db.table(table).count("*").fetchone() == (0,)
    # A subsequent complete run still records the exact tree.
    start(db)
    assert db.db.table("cluster_tree_transition").count("*").fetchone() == (6,)
    db.close()


def test_tree_persistence_empty_and_caller_transaction():
    db = RealizationDatabase(":memory:")
    db.create_tables()
    populate(db.db)
    db.db.execute("insert into experiment_config (id) values (1)")
    db.db.execute(
        "insert into experiment_run (id, experiment_config_id, simulation_mode) values (1, 1, 'CLUSTER_TREE')"
    )
    db.db.execute("begin transaction")
    db.start_tree_result(1, 10, tree(nodes=[], destinations=[]), {}, [])
    start(db)
    db.db.execute("rollback")
    assert db.db.table("cluster_tree_experiment_result").count("*").fetchone() == (0,)
    assert db.db.table("cluster_tree_node").count("*").fetchone() == (0,)
    db.close()
