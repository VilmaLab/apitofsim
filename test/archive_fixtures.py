"""Small, explicit IPC fixtures for query tests, independent of the simulator."""

import json
from collections import defaultdict

import pyarrow as pa
from apitofsim.recording import EVENT_SCHEMA


def publish_events(db, rows, *, batch_size=None, statuses=None, format="arrow"):
    db._release_event_reader()
    identity, generation, _ = db.db.execute(
        "select archive_id,generation,file_size from event_recording_state"
    ).fetchall()[0]
    schema = pa.schema(
        [pa.field(f.name, f.type, nullable=False) for f in EVENT_SCHEMA]
    ).with_metadata(
        {
            b"apitofsim:event_schema": b"1",
            b"apitofsim:archive_id": identity.encode(),
            b"apitofsim:generation": str(generation + 1).encode(),
        }
    )
    by_result = defaultdict(list)
    for row in rows:
        by_result[row["result_id"]].append(
            {
                **{field.name: 0 for field in schema},
                "transition_index": -1,
                "next_particle_index": -1,
                **row,
            }
        )
    metadata = {}
    batches = []
    for result_id, values in by_result.items():
        result_batches = pa.Table.from_pylist(values, schema=schema).to_batches(
            max_chunksize=batch_size
        )
        info = metadata[str(result_id)] = dict(
            status=(statuses or {}).get(result_id, "complete"),
            event_count=len(values),
            batch_start=len(batches) if format == "arrow" else 0,
            batch_count=len(result_batches) if format == "arrow" else 0,
        )
        batches.extend(result_batches)
        db.db.execute(
            """update event_execution set recorded=true,status=?,event_count=?,
               batch_start=?,batch_count=? where result_id=?""",
            [*info.values(), result_id],
        )
    path = db.events_path
    if format == "parquet":
        import pyarrow.parquet as pq

        path = path.with_suffix(".parquet")
        pq.write_table(pa.Table.from_batches(batches, schema=schema), path)
    else:
        with pa.ipc.new_file(
            str(path),
            schema,
            metadata={b"apitofsim:executions": json.dumps(metadata).encode()},
        ) as writer:
            for batch in batches:
                writer.write_batch(batch)
    db.db.execute(
        "update event_recording_state set schema_version=?,generation=?,file_size=?",
        [2 if format == "parquet" else 1, generation + 1, path.stat().st_size],
    )
    db._ensure_event_reader()
