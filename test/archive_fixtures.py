"""Small, explicit IPC fixtures for query tests, independent of the simulator."""

import json

import pyarrow as pa
from apitofsim.recording import EVENT_SCHEMA


def publish_events(db, rows):
    db._release_event_reader()
    identity, generation, _ = db.db.execute(
        "select archive_id,generation,file_size from event_recording_state"
    ).fetchall()[0]
    schema = pa.schema(
        [pa.field(f.name, f.type, nullable=False) for f in EVENT_SCHEMA]
    ).with_metadata(
        {b"apitofsim:event_schema": b"1", b"apitofsim:archive_id": identity.encode()}
    )
    values = [
        {
            **{field.name: 0 for field in schema},
            "transition_index": -1,
            "next_particle_index": -1,
            **row,
        }
        for row in rows
    ]
    table = pa.Table.from_pylist(values, schema=schema)
    ids = sorted({row["result_id"] for row in values})
    metadata = {str(id): {"status": "complete"} for id in ids}
    with pa.ipc.new_file(
        str(db.events_path),
        schema,
        metadata={b"apitofsim:executions": json.dumps(metadata).encode()},
    ) as writer:
        writer.write_table(table)
    for id in ids:
        db.db.execute(
            "update event_execution set recorded=true,status='complete' where result_id=?",
            [id],
        )
    db.db.execute(
        "update event_recording_state set generation=?,file_size=?",
        [generation + 1, db.events_path.stat().st_size],
    )
    db._ensure_event_reader()
