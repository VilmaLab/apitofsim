"""Parquet publication and reader coverage with and without the native writer."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from apitofsim import api, recording
from apitofsim.workflow import ExperimentDatabase
from apitofsim.workflow.db import auto_db_type
from archive_fixtures import publish_events
from test_recording_stream import prepare


def test_parquet_reader_discovers_and_scopes_archive(tmp_path):
    path = tmp_path / "experiment.duckdb"
    archive = recording.events_filename(path, format="parquet")
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
                dict(result_id=1, realization_index=0, event_index=0, event_kind=0),
                dict(result_id=1, realization_index=2, event_index=0, event_kind=0),
                dict(
                    result_id=1,
                    realization_index=2,
                    event_index=1,
                    event_kind=1,
                    x=1.25,
                ),
                dict(result_id=2, realization_index=5, event_index=0, event_kind=0),
            ],
            format="parquet",
        )
        db.finish_execution(3, "complete")
        identity, generation, size = db.db.sql(
            "select archive_id, generation, file_size from event_recording_state"
        ).fetchone()
    ipc = recording.events_filename(path)
    ipc.write_bytes(b"unrelated stale IPC path")
    journal = Path(str(ipc) + ".journal")
    journal.write_text(
        json.dumps(
            dict(
                version=1,
                owner="database",
                archive_id=identity,
                generation=generation,
                old_size=0,
                new_size=size,
                format="parquet",
            )
        )
    )
    with auto_db_type(path, readonly=True) as db:
        assert db.events_path == archive
        assert db.events_query(run_id=1, result_id=2).fetchall() == []
        assert db.events_query(result_id=3).fetchall() == []
        scope = dict(run_id=1, parent=10, result_id=1)
        query = db.events_query(**scope).filter("realization_index=2").select("x")
        plan = query.explain()
        assert "PARQUET_SCAN" in plan or "READ_PARQUET" in plan
        assert "result_id=1" in plan and "realization_index=2" in plan
        assert query.order("x").fetchall() == [(0.0,), (1.25,)]
        assert len(db.realizations_query(**scope).fetchall()) == 2
        assert db.events_query(**scope).count("*").fetchone() == (3,)
    assert journal.exists()
    with ExperimentDatabase(path) as db:
        assert not journal.exists()
    assert [
        (e.realization_index, e.event_index)
        for e in api.read_events(archive, result_id=1, events="individual")
    ] == [(0, 0), (2, 0), (2, 1)]
    relocated = archive.with_name("relocated archive's data")
    archive.rename(relocated)
    with pytest.raises(FileNotFoundError, match=r"matching .events.parquet"):
        ExperimentDatabase(path, readonly=True)
    with ExperimentDatabase(path, readonly=True, events_filename=relocated) as db:
        assert db.events_query().count("*").fetchone() == (4,)
    archive.write_bytes(relocated.read_bytes()[:-1])
    with pytest.raises(ValueError, match="identity/size"):
        ExperimentDatabase(path, readonly=True)


def test_standalone_overwrite_replaces_unrelated_parquet(tmp_path):
    path = tmp_path / "standalone.events.parquet"
    pq.write_table(pa.table({"value": [1]}), path)
    with pytest.raises(ValueError, match="Cannot append to a Parquet"):
        with recording.RecordingTransaction(recording.IpcRecording(path, "append")):
            pytest.fail("Unexpected Parquet append")

    original = path.read_bytes()
    with recording.RecordingTransaction(recording.IpcRecording(path, "overwrite")):
        assert path.read_bytes() == original
    assert recording.archive_metadata(path)[0].startswith("standalone:")


def test_standalone_overwrite_rejects_database_owned_parquet(tmp_path):
    with ExperimentDatabase(tmp_path / "managed.duckdb") as db:
        db.create_tables()
        publish_events(db, [], format="parquet")
        path = db.events_path

    original = path.read_bytes()
    with pytest.raises(ValueError, match="database-owned"):
        with recording.RecordingTransaction(recording.IpcRecording(path, "overwrite")):
            pytest.fail("Unexpected overwrite of database-owned Parquet")
    assert path.read_bytes() == original


@pytest.mark.parametrize("compression, workers", [("none", 1), ("zstd", 4)])
def test_native_parquet_recording(prepared_case, compression, workers):
    if not api.apitofsimraw.NATIVE_PARQUET:
        pytest.skip("Native Parquet writer is disabled")
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    ipc = db.events_path
    archive = ipc.with_suffix(".parquet")
    db.recording_format = "parquet"
    db.recording_compression = compression
    db.parquet_column_workers = workers
    with runner.iter_experiment(
        prepared, N=4, record_events=True, events="batches", max_batch_items=7
    ) as stream:
        live = pa.Table.from_batches(list(stream))
    assert db.events_path == archive
    assert not ipc.exists()
    assert not Path(str(ipc) + ".journal").exists()
    with pq.ParquetFile(archive) as file:
        assert file.metadata.num_rows == len(live) > 0
        assert all(
            file.metadata.row_group(i).column(0).compression
            == ("ZSTD" if compression == "zstd" else "UNCOMPRESSED")
            for i in range(file.metadata.num_row_groups)
        )
    recorded = pa.Table.from_batches(list(api.read_events(archive)))
    assert live.to_pydict() == recorded.to_pydict()
    assert (
        recorded.to_pydict()
        == recorded.sort_by(
            [
                (name, "ascending")
                for name in ("result_id", "realization_index", "event_index")
            ]
        ).to_pydict()
    )
    info = stream.result.roots[0].result.recording
    assert info.path == archive and info.event_count == len(live)
    assert info.batch_count == 0
    assert db.events_query().count("*").fetchone() == (len(live),)
    with pytest.raises(ValueError, match="separate database"):
        with db.recording_transaction():
            pytest.fail("Unexpected Parquet append")


def test_grouped_parquet_run_returns_sidecar_path(prepared_case):
    if not api.apitofsimraw.NATIVE_PARQUET:
        pytest.skip("Native Parquet writer is disabled")
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    ipc = db.events_path
    archive = ipc.with_suffix(".parquet")
    db.recording_format = "parquet"

    with db.recording_transaction():
        result = runner.run_experiment(prepared, N=4)
        assert db.events_path == ipc
        assert result.events_path == archive

    assert result.events_path == db.events_path
    assert pq.read_metadata(result.events_path).num_rows > 0


def test_parquet_rollback_after_failed_sync(prepared_case, monkeypatch):
    if not api.apitofsimraw.NATIVE_PARQUET:
        pytest.skip("Native Parquet writer is disabled")
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    ipc = db.events_path
    db.recording_format = "parquet"
    db.parquet_column_workers = 4

    def fail_sync(path):
        raise OSError("Injected sidecar sync failure")

    monkeypatch.setattr(recording, "sync_file", fail_sync)
    with pytest.raises(OSError, match="Injected sidecar sync failure"):
        runner.run_experiment(prepared, N=4, record_events=True)
    assert db.events_path == ipc
    assert not ipc.with_suffix(".parquet").exists()
    assert not Path(str(ipc) + ".journal").exists()
    assert db.db.sql(
        "select schema_version, file_size from event_recording_state"
    ).fetchone() == (1, 0)
    assert db.db.sql("select count(*) from event_execution").fetchone() == (0,)


def test_recover_uncommitted_parquet(tmp_path):
    path = tmp_path / "interrupted.duckdb"
    ipc = recording.events_filename(path)
    archive = ipc.with_suffix(".parquet")
    journal = Path(str(ipc) + ".journal")
    with ExperimentDatabase(path) as db:
        db.create_tables()
        identity, generation = db.db.sql(
            "select archive_id, generation from event_recording_state"
        ).fetchone()
    archive.write_bytes(b"PAR1 incomplete")
    journal.write_text(
        json.dumps(
            dict(
                version=1,
                owner="database",
                archive_id=identity,
                generation=generation + 1,
                old_size=0,
                format="parquet",
            )
        )
    )
    with pytest.raises(ValueError, match="owning experiment database"):
        next(api.read_events(archive))
    with pytest.raises(ValueError, match="owning experiment database"):
        with recording.RecordingTransaction(
            recording.IpcRecording(archive, "overwrite")
        ):
            pytest.fail("Unexpected overwrite of interrupted database recording")
    with pytest.raises(ValueError, match="Open the database writable"):
        ExperimentDatabase(path, readonly=True)
    with ExperimentDatabase(path) as db:
        assert db.events_path == ipc
        assert not archive.exists()
        assert not journal.exists()


@pytest.mark.parametrize("workers", [0, -1, 2**31])
def test_parquet_workers_must_be_positive_int(tmp_path, workers):
    path = tmp_path / "invalid.duckdb"
    with pytest.raises(ValueError, match="column workers"):
        ExperimentDatabase(
            path,
            recording_format="parquet",
            parquet_column_workers=workers,
        )
    assert not path.exists()


def test_parquet_recording_requires_native_build(tmp_path):
    if api.apitofsimraw.NATIVE_PARQUET:
        pytest.skip("Native Parquet writer is enabled")
    path = tmp_path / "unavailable.duckdb"
    with ExperimentDatabase(path, recording_format="parquet") as db:
        db.create_tables()
        with pytest.raises(RuntimeError, match="APITOFSIM_NATIVE_PARQUET=ON"):
            with db.recording_transaction():
                pytest.fail("Unexpected Parquet writer")
    assert not recording.events_filename(path, format="parquet").exists()
