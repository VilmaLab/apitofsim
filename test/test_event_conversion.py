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
def vortex_extension():
    with duckdb.connect() as db:
        try:
            db.load_extension("vortex")
        except duckdb.Error:
            pytest.skip("DuckDB Vortex extension is not installed")


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


@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_convert_and_discover_or_relocate(recording, compression):
    ipc = events_filename(recording)
    with ExperimentDatabase(recording, readonly=True) as db:
        before = snapshot(db)
        identity, generation = db.db.execute(
            "select archive_id,generation from event_recording_state"
        ).fetchone()
    args = ["convert-events", str(recording), "--to", "parquet"]
    if compression == "zstd":
        args.extend(["--compression", compression])
    result = CliRunner().invoke(cli, args)
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
    ] == [(2, 1), (2, 0), (0, 0)]
    assert np.signbit(pq.read_table(destination)["x"][0].as_py())
    parquet = pq.read_metadata(destination)
    assert parquet.num_row_groups == 4
    assert parquet.row_group(0).column(0).compression == (
        "ZSTD" if compression == "zstd" else "UNCOMPRESSED"
    )

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
    with pytest.raises(ValueError, match="compression"):
        convert_events(path, compression="gzip")
    destination = convert_events(path)
    assert pq.read_metadata(destination).num_rows == 0
    with ExperimentDatabase(path, readonly=True) as db:
        assert snapshot(db) == before


def test_vortex_conversion(recording, vortex_extension):
    ipc = events_filename(recording)
    with ExperimentDatabase(recording, readonly=True) as db:
        before = snapshot(db)
    result = CliRunner().invoke(
        cli, ["convert-events", str(recording), "--to", "vortex"]
    )
    assert result.exit_code == 0, result.output
    destination = events_filename(recording, format="vortex")
    assert not ipc.exists()
    with ExperimentDatabase(recording) as db:
        assert db.events_path == destination
        assert snapshot(db) == before
        assert "VORTEX" in db.events_query(result_id=1).explain().upper()
        with pytest.raises(ValueError, match="separate database"):
            with db.recording_transaction():
                pytest.fail("Append opened")
    assert convert_events(recording, format="vortex") == destination
    assert [
        (event.realization_index, event.event_index)
        for event in read_events(destination, events="individual", result_id=1)
    ] == [(2, 1), (2, 0), (0, 0)]
    assert np.signbit(next(read_events(destination, events="individual")).x)
    with pytest.raises(ValueError, match="already vortex"):
        convert_events(recording)

    relocated = destination.with_name("relocated'archive")
    destination.rename(relocated)
    with ExperimentDatabase(recording, readonly=True, events_filename=relocated) as db:
        assert snapshot(db) == before
    relocated.rename(destination)
    with duckdb.connect(recording) as db:
        db.execute("update event_recording_state set generation=generation+1")
    with pytest.raises(ValueError, match="identity/generation"):
        ExperimentDatabase(recording, readonly=True)


def test_vortex_empty_archive(tmp_path, vortex_extension):
    path = tmp_path / "empty.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        publish_events(db, [])
    with pytest.raises(ValueError, match="Parquet"):
        convert_events(path, format="vortex", compression="zstd")
    destination = convert_events(path, format="vortex")
    assert list(read_events(destination)) == []
    with ExperimentDatabase(path, readonly=True) as db:
        assert db.events_query().fetchall() == []


def test_conversion_splits_large_batches_without_reordering(tmp_path, monkeypatch):
    path = tmp_path / "ordered.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        for result in (1, 2):
            db.begin_execution(1, 1, None, [], result_id=result, recorded=True)
        publish_events(
            db,
            [
                dict(result_id=result, realization_index=0, event_index=event)
                for result, count in ((1, 5), (2, 1))
                for event in range(count)
            ],
        )
    monkeypatch.setattr(event_conversion, "_ROW_GROUP_SIZE", 2)
    destination = convert_events(path, compression="zstd")
    footer = pq.read_metadata(destination)
    assert [footer.row_group(i).num_rows for i in range(footer.num_row_groups)] == [
        2,
        2,
        1,
        1,
    ]
    assert [
        (event.result_id, event.event_index)
        for event in read_events(destination, events="individual")
    ] == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0)]


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
@pytest.mark.parametrize("format", ["parquet", "vortex"])
def test_interrupted_publication(recording, monkeypatch, after_commit, format, request):
    if format == "vortex":
        request.getfixturevalue("vortex_extension")
    destination = events_filename(recording, format=format)
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
            convert_events(recording, format=format)

    assert destination.exists() and source.exists()
    with ExperimentDatabase(recording, readonly=True) as db:
        assert db.events_path == (destination if after_commit else source)
        assert snapshot(db) == before
    assert source.exists()
    assert convert_events(recording, format=format) == destination
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
