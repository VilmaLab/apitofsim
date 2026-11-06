"""In-place publication and dual-format queries for database recordings."""

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest
from apitofsim import convert_events, event_conversion, read_events
from apitofsim.cli import cli
from apitofsim.recording import events_filename, recording_lock
from apitofsim.workflow import ExperimentDatabase
from apitofsim.workflow.db import auto_db_type
from archive_fixtures import publish_events
from click.testing import CliRunner
from portalocker.exceptions import LockException


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "experiment.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        db.db.execute("insert into cluster (id,common_name) values (10,'parent')")
        db.db.execute("insert into pathway values (1,10,10,10)")
        db.db.execute("insert into experiment_config (id) values (1)")
        db.db.execute("""
            insert into experiment_run (id,experiment_config_id,simulation_mode)
            values (1,1,'PATHWAY_AT_A_TIME'),(2,1,'PATHWAY_AT_A_TIME')
        """)
        for result, run in [(1, 1), (2, 2), (3, 1)]:
            db.begin_execution(run, 10, 1, [1], result_id=result, recorded=True)
        publish_events(
            db,
            [
                dict(
                    result_id=2,
                    realization_index=5,
                    event_index=0,
                    event_kind=0,
                    x=-0.0,
                ),
                dict(
                    result_id=1,
                    realization_index=2,
                    event_index=1,
                    event_kind=1,
                    x=1.0000000000000002,
                ),
                dict(result_id=1, realization_index=2, event_index=0, event_kind=0),
                dict(result_id=1, realization_index=0, event_index=0, event_kind=0),
            ],
            batch_size=1,
            statuses={1: "cancelled", 2: "failed"},
        )
        db.finish_execution(3, "complete")
    return path


def snapshot(db):
    results = []
    for scope in (
        {},
        {"run_id": 1, "parent": 10},
        {"run_id": 2, "result_id": 1},
        {"result_id": 3},
        {"result_id": 2},
    ):
        for query in (
            db.events_query(**scope, sort=True),
            db.realizations_query(**scope).order("realization_id"),
            db.tree_events_query(**scope),
            db.recorded_output_query(**scope),
            db.results_query(**scope).order("experiment_result_id"),
        ):
            results.append((query.columns, query.types, query.fetchall()))
    for view, order in (
        ("event_details", "id"),
        ("event_info", "id"),
        ("realization", "id"),
        ("event_report", "event_id"),
    ):
        query = db.db.table(view).order(order)
        results.append((query.columns, query.types, query.fetchall()))
    return results


def test_convert_and_discover_or_relocate(recording):
    ipc = events_filename(recording)
    with ExperimentDatabase(recording, readonly=True) as db:
        before = snapshot(db)
        identity, generation = db.db.execute(
            "select archive_id,generation from event_recording_state"
        ).fetchone()
    result = CliRunner().invoke(
        cli, ["convert-events", str(recording), "--to", "parquet"]
    )
    assert result.exit_code == 0, result.output
    assert "Validating Parquet" in result.output
    destination = events_filename(recording, parquet=True)
    assert not ipc.exists()
    with ExperimentDatabase(recording) as db:
        assert db.events_path == destination
        assert snapshot(db) == before
        assert db.db.table("event_recording_state").fetchall() == [
            (2, identity, generation + 1, destination.stat().st_size)
        ]
        assert db.db.sql("select sum(batch_count) from event_execution").fetchone() == (
            0,
        )
        with pytest.raises(ValueError, match="separate database"):
            with db.recording_transaction():
                pytest.fail("Append opened")
        query = db.events_query(result_id=1).filter("realization_index=2").select("x")
        plan = query.explain()
        assert "PARQUET_SCAN" in plan or "READ_PARQUET" in plan
        assert "result_id=1" in plan
        assert "realization_index=2" in plan
        assert query.order("x").fetchall() == [(0.0,), (1.0000000000000002,)]
    with auto_db_type(recording, readonly=True) as db:
        assert snapshot(db) == before
    assert convert_events(recording) == destination
    assert [
        (e.realization_index, e.event_index)
        for e in read_events(destination, events="individual", result_id=1)
    ] == [(0, 0), (2, 0), (2, 1)]
    assert np.signbit(pq.read_table(destination)["x"][-1].as_py())

    relocated = destination.with_name("relocated archive's data")
    destination.rename(relocated)
    with pytest.raises(FileNotFoundError, match=r"matching .events.parquet"):
        ExperimentDatabase(recording, readonly=True)
    with ExperimentDatabase(recording, readonly=True, events_filename=relocated) as db:
        assert snapshot(db) == before
    altered = relocated.read_bytes()
    destination.write_bytes(altered[:-1])
    with pytest.raises(ValueError, match="identity/size"):
        ExperimentDatabase(recording, readonly=True)


def test_convert_empty_archive(tmp_path):
    path = tmp_path / "empty.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        publish_events(db, [])
        before = snapshot(db)
    destination = convert_events(path)
    assert pq.read_metadata(destination).num_rows == 0
    with ExperimentDatabase(path, readonly=True) as db:
        assert snapshot(db) == before


def test_reject_missing_source_and_wrong_generation(recording):
    source = events_filename(recording)
    relocated = source.with_suffix(".relocated")
    source.rename(relocated)
    with pytest.raises(FileNotFoundError, match=r"No matching .events.arrow"):
        convert_events(recording)
    relocated.rename(source)
    convert_events(recording)
    with duckdb.connect(recording) as db:
        db.execute("update event_recording_state set generation=generation+1")
    with pytest.raises(ValueError, match="identity/generation"):
        ExperimentDatabase(recording, readonly=True)


def test_unrelated_destination_and_locking(recording):
    destination = events_filename(recording, parquet=True)
    source = events_filename(recording)
    destination.write_bytes(b"unrelated")
    with ExperimentDatabase(recording, readonly=True) as db:
        assert db.events_path == source
        with pytest.raises(LockException):
            convert_events(recording)
    with recording_lock(source):
        with pytest.raises(LockException):
            convert_events(recording)
    with pytest.raises(FileExistsError, match="cannot be resumed"):
        convert_events(recording)
    assert destination.read_bytes() == b"unrelated"
    assert source.exists()


@pytest.mark.parametrize("after_commit", [False, True])
def test_interrupted_publication(recording, monkeypatch, after_commit):
    destination = events_filename(recording, parquet=True)
    source = events_filename(recording)
    with ExperimentDatabase(recording) as db:
        before = snapshot(db)

    with monkeypatch.context() as patch:
        if after_commit:
            unlink = Path.unlink

            def fail_cleanup(path, *args, **kwargs):
                if path == source:
                    raise OSError("interrupted")
                return unlink(path, *args, **kwargs)

            patch.setattr(Path, "unlink", fail_cleanup)
        else:
            sync_directory = event_conversion.sync_directory

            def fail_publication(path):
                if path == recording.parent and destination.exists():
                    raise OSError("interrupted")
                sync_directory(path)

            patch.setattr(event_conversion, "sync_directory", fail_publication)
        with pytest.raises(OSError, match="interrupted"):
            convert_events(recording)

    assert destination.exists() and source.exists()
    with ExperimentDatabase(recording, readonly=True) as db:
        assert db.events_path == (destination if after_commit else source)
        assert snapshot(db) == before
    assert source.exists()
    assert convert_events(recording) == destination
    assert not source.exists()
    with ExperimentDatabase(recording, readonly=True) as db:
        assert snapshot(db) == before


def test_failed_validation_and_owned_staging_cleanup(recording, monkeypatch):
    destination = events_filename(recording, parquet=True)
    source = events_filename(recording)
    with duckdb.connect(recording, read_only=True) as db:
        identity, generation = db.execute(
            "select archive_id,generation from event_recording_state"
        ).fetchone()
    owned = recording.parent / f".{destination.name}.convert-crashed"
    unrelated = recording.parent / f".{destination.name}.convert-unrelated"
    for path, marker in ((owned, f"{identity}:{generation + 1}"), (unrelated, "other")):
        path.mkdir()
        (path / "owner").write_text(marker)
        (path / "partial").touch()

    with monkeypatch.context() as patch:

        def fail_validation(*args):
            raise ValueError("bad payload")

        patch.setattr(event_conversion, "_validate_payload", fail_validation)
        with pytest.raises(ValueError, match="bad payload"):
            convert_events(recording)
    assert source.exists() and not destination.exists()
    assert list(recording.parent.glob(f".{destination.name}.convert-*")) == [unrelated]
    convert_events(recording)
