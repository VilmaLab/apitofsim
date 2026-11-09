"""Composite event identity and persisted tree mappings across shared IPC roots."""

import re

import pytest
from apitofsim import apitofsimraw as raw
from apitofsim import convert_events, recording
from apitofsim.workflow import ExperimentDatabase
from archive_fixtures import publish_events


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


@pytest.mark.parametrize("parquet", [False, True])
def test_shared_archive_tree_identity_and_order(tmp_path, monkeypatch, parquet):
    path = tmp_path / "trees.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        populate(db.db)
        db.db.execute("insert into experiment_config(id) values(1)")
        db.db.execute(
            "insert into experiment_run(id,experiment_config_id,simulation_mode) values (1,1,'CLUSTER_TREE'),(2,1,'CLUSTER_TREE')"
        )
        first, second = start(db), start(db)
        db.db.execute(
            "update cluster_tree_experiment_result set experiment_run_id=2 where id=?",
            [second],
        )
        for result, run in [(first, 1), (second, 2)]:
            db.begin_execution(
                run, 10, None, [1, 2, 3, 4, 5, 5], result_id=result, recorded=True
            )
        rows = []
        for result, realization, nodes, transitions in [
            (first, 0, [0, 1, 3], [0, 2, 4]),
            (first, 1, [0, 2, 4], [1, 3, 5]),
            (second, 0, [0, 1, 3], [0, 2, 4]),
        ]:

            def emit(kind, node, **details):
                count = sum(
                    r["result_id"] == result and r["realization_index"] == realization
                    for r in rows
                )
                rows.append(
                    dict(
                        result_id=result,
                        realization_index=realization,
                        event_index=count,
                        event_kind=kind,
                        particle_index=node,
                        **details,
                    )
                )

            emit(0, 0)
            for node, transition, destination in zip(
                nodes, transitions, [*nodes[1:], -1], strict=True
            ):
                emit(1, node, accepted=1)
                emit(2, node)
                emit(
                    3,
                    node,
                    transition_index=transition,
                    next_particle_index=destination,
                )
                if destination >= 0:
                    emit(4, destination)
        publish_events(
            db, rows, batch_size=8, statuses={first: "cancelled", second: "failed"}
        )
        histories = db.tree_events_query().fetchall()
        assert len(histories) == len(rows)
        assert db.realizations_query().count("*").fetchone() == (3,)
        assert len(db.tree_events_query(run_id=1).fetchall()) == 2 * len(
            db.tree_events_query(run_id=2).fetchall()
        )
        assert (
            db.tree_events_query()
            .filter("not continuity_valid or provenance != 'persisted'")
            .fetchall()
            == []
        )
        splits = db.fragmentation_histories_query(result_id=first).fetchdf()
        assert splits["pathway_id"].tolist() == [1, 3, 5, 2, 4, 5]
        assert splits["current_cluster_id"].tolist() == [10, 20, 40, 10, 30, 40]
        assert splits["event_index"].tolist() == [3, 7, 11, 3, 7, 11]
        expected = db.events_query(sort=True).fetchall()
    # Both the database and its sidecar remain relocatable as a pair.
    renamed = tmp_path / "renamed.duckdb"
    path.rename(renamed)
    (tmp_path / "trees.events.arrow").rename(tmp_path / "renamed.events.arrow")
    if parquet:
        convert_events(renamed)
    with ExperimentDatabase(renamed, readonly=True) as reopened:
        assert reopened.events_query(sort=True).fetchall() == expected
        reads = []
        open_file = recording.pa.ipc.open_file

        class TracedReader:
            def __init__(self, source, options):
                self.reader = open_file(source, options=options)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.reader.__exit__(*exc)

            def get_batch(self, index):
                batch = self.reader.get_batch(index)
                reads.append((index, batch.schema.names))
                return batch

        if not parquet:
            monkeypatch.setattr(recording.pa.ipc, "open_file", TracedReader)
        scope = dict(run_id=2, parent=10, result_id=second)
        scoped_events = reopened.events_query(**scope, sort=True)
        assert [index for index, _ in reads] == ([] if parquet else [3, 4])
        reads.clear()
        scoped_tree = reopened.tree_events_query(**scope)
        assert [index for index, _ in reads] == ([] if parquet else [3, 4])
        reads.clear()
        scoped_realizations = reopened.realizations_query(**scope)
        assert reads == (
            []
            if parquet
            else [(index, ["realization_index", "result_id"]) for index in (3, 4)]
        )
        reads.clear()

        expected_events = [row for row in expected if row[2] == second]
        assert scoped_events.fetchall() == expected_events
        assert scoped_tree.fetchall() == [row for row in histories if row[2] == second]
        assert scoped_realizations.fetchall() == [
            (f"{second}:0", second, 2, 10, None, False)
        ]
        assert scoped_events.fetchall() == expected_events
        assert reads == []
        if not parquet:
            assert len(re.findall(r"│\s+ARROW_SCAN\s+│", scoped_tree.explain())) == 1
        assert reopened.recorded_output_query(**scope).fetchone() == (
            1,
            12,
            0,
            3,
            3,
            3,
            3,
            1,
        )

        assert reopened.tree_events_query(parent=10).fetchall() == histories
        reads.clear()
        empty = dict(run_id=1, parent=10, result_id=second)
        assert reopened.events_query(**empty).fetchall() == []
        assert reopened.tree_events_query(**empty).fetchall() == []
        assert reopened.realizations_query(**empty).fetchall() == []
        assert reads == []


def test_reject_legacy_event_database(tmp_path):
    import duckdb

    path = tmp_path / "old.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("create table event_info(id integer)")
    with pytest.raises(ValueError, match="Legacy relational-event.*migration"):
        ExperimentDatabase(path)
