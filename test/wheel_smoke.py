"""Exercise native IPC and PyArrow interoperability in every platform wheel."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import apitofsim
import pyarrow as pa
from apitofsim.recording import RecordingTransaction, recording_lock
from apitofsim.workflow import ExperimentDatabase
from archive_fixtures import publish_events
from portalocker.exceptions import LockException

assert apitofsim.precompute_mesh(100.0, 1.0, quantities_strict=False).size == 100
with TemporaryDirectory() as directory:
    path = Path(directory) / "évents.arrow"
    for mode, status in [
        ("create", "cancelled"),
        ("append", "complete"),
        ("overwrite", "complete"),
    ]:
        with RecordingTransaction(apitofsim.IpcRecording(path, mode)) as recording:
            recording.executions["1"] = {"status": status}
        with pa.OSFile(str(path), "rb") as source, pa.ipc.open_file(source) as reader:
            assert reader.schema.field("event_index").type == pa.int64()
            assert reader.num_record_batches == 0
            assert (
                json.loads(reader.metadata[b"apitofsim:executions"])["1"]["status"]
                == status
            )

# Exercise publication and shared locking with the platform's filesystem.
with TemporaryDirectory() as directory:
    database = Path(directory) / "évents.duckdb"
    with ExperimentDatabase(database) as db:
        db.create_tables()
        db.begin_execution(1, 1, None, [], result_id=1, recorded=True)
        publish_events(
            db, [dict(result_id=1, event_kind=0, realization_index=0, event_index=0)]
        )
        ipc = db.events_path
        try:
            with recording_lock(ipc):
                raise AssertionError("Acquired exclusive lock while a reader was open")
        except LockException:
            pass
    parquet = apitofsim.convert_events(database)
    assert parquet.is_file() and not ipc.exists()
    with ExperimentDatabase(database, readonly=True) as db:
        assert db.events_path == parquet
        assert db.events_query(result_id=1).select("event_id").fetchall() == [
            ("1:0:0",)
        ]
    assert apitofsim.convert_events(database) == parquet
