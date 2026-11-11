"""Native recording, public delivery, append publication and crash recovery."""

import json
from pathlib import Path

import portalocker
import pyarrow as pa
import pytest
from apitofsim import api
from apitofsim import recording as persistence
from apitofsim.workflow import ExperimentDatabase
from test_recording_stream import assert_counters, prepare


def inputs(case):
    prepared = prepare(case, pathways=[1])
    return prepared.mass_spec, prepared.groups[0].substance


def ordered(table):
    return table.sort_by(
        [
            (name, "ascending")
            for name in ("result_id", "realization_index", "event_index")
        ]
    ).to_pydict()


def test_native_appender_preserves_cpp_written_ipc(tmp_path):
    """PyArrow uses Arrow C++ to create an archive independent of nanoarrow."""
    from apitofsim import apitofsimraw as raw

    path = tmp_path / "cpp-written.arrow"
    schema = pa.schema(
        [
            pa.field(field.name, field.type, nullable=False)
            for field in persistence.EVENT_SCHEMA
        ],
        metadata={b"apitofsim:event_schema": b"1", b"apitofsim:archive_id": b"compat"},
    )
    original = pa.RecordBatch.from_arrays(
        [
            pa.array(
                [1.25] if pa.types.is_floating(field.type) else [7], type=field.type
            )
            for field in schema
        ],
        schema=schema,
    )
    with pa.ipc.new_file(path, schema) as writer:
        writer.write_batch(original)
    before = path.read_bytes()
    footer_size = int.from_bytes(before[-10:-6], "little")
    payload_end = len(before) - footer_size - 18

    writer = raw.IpcEventWriter(str(path), True, "compat")
    assert writer.batch_count == 1
    writer.close('{"7":{"status":"complete"}}')
    assert path.read_bytes()[:payload_end] == before[:payload_end]
    with pa.ipc.open_file(path) as reader:
        assert reader.num_record_batches == 1
        assert reader.schema.equals(schema, check_metadata=True)
        assert reader.get_batch(0).to_pydict() == original.to_pydict()
        assert json.loads(reader.metadata[b"apitofsim:executions"]) == {
            "7": {"status": "complete"}
        }


@pytest.mark.parametrize("case", ["missing-identity", "dictionary"])
def test_native_appender_rejects_incompatible_cpp_schema(tmp_path, case):
    from apitofsim import apitofsimraw as raw

    path = tmp_path / "incompatible.arrow"
    schema = persistence.EVENT_SCHEMA
    identity = "compat"
    if case == "dictionary":
        index = schema.get_field_index("result_id")
        schema = schema.set(
            index, pa.field("result_id", pa.dictionary(pa.int64(), pa.int64()))
        )
        schema = schema.with_metadata({b"apitofsim:archive_id": b"compat"})
    else:
        identity = ""
    with pa.ipc.new_file(path, schema):
        pass
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="identity|schema"):
        raw.IpcEventWriter(str(path), True, identity)
    assert path.read_bytes() == original


def test_recording_and_nonrecording_same_counters(prepared_case, tmp_path):
    ms, substance = inputs(prepared_case)
    plain = api.mass_spec(ms, substance, 3, strict=False)
    assert plain.recording is None
    path = tmp_path / "standalone.arrow"
    recorded = api.mass_spec(ms, substance, 3, strict=False, record_to=path)
    assert_counters(recorded.counters, plain.counters)
    assert recorded.status == "complete"
    assert recorded.recording.event_count > 0
    assert recorded.recording.path == path
    with pa.ipc.open_file(path) as read:
        assert read.schema.metadata[b"apitofsim:event_schema"] == b"1"
        states = json.loads(read.metadata[b"apitofsim:executions"])
        assert states["1"]["status"] == "complete"
        assert states["1"]["ordered"] is True
        assert len(read.read_all()) == recorded.recording.event_count
    with pytest.raises(FileExistsError, match="append or overwrite"):
        api.mass_spec(ms, substance, 1, strict=False, record_to=path)


def test_ordered_stream_and_recording_match(prepared_case, tmp_path):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "ordered.arrow"
    with pytest.raises(ValueError, match="requires ordered"):
        api.mass_spec(ms, substance, 8, record_to=path, ordered=False)
    assert not path.exists()
    with api.mass_spec_iter(
        ms,
        substance,
        8,
        strict=False,
        ordered=True,
        initial_batch_size=2,
        max_batch_size=7,
    ) as stream:
        live = pa.Table.from_batches(list(stream))
    api.mass_spec(ms, substance, 8, strict=False, record_to=path)
    with pa.ipc.open_file(path) as reader:
        recorded = reader.read_all()
    assert live.to_pydict() == ordered(live)
    assert recorded.to_pydict() == ordered(recorded)
    assert (
        live.drop(["result_id"]).to_pydict() == recorded.drop(["result_id"]).to_pydict()
    )


def test_recording_callbacks_run_on_caller(prepared_case, tmp_path):
    from threading import get_ident

    ms, substance = inputs(prepared_case)
    callbacks = []
    result = api.mass_spec(
        ms,
        substance,
        8,
        strict=False,
        record_to=tmp_path / "callbacks.arrow",
        result_callback=lambda counters: callbacks.append((get_ident(), counters)),
    )
    assert callbacks and all(thread == get_ident() for thread, _ in callbacks)
    assert_counters(callbacks[-1][1], result.counters)


@pytest.mark.parametrize("delivery", ["batches", "individual"])
def test_python_delivery_with_and_without_native_recording(
    prepared_case, tmp_path, delivery
):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "delivered.arrow"
    with api.mass_spec_iter(
        ms,
        substance,
        3,
        strict=False,
        events=delivery,
        record_to=path,
        initial_batch_size=2,
        max_batch_size=7,
        max_pending_events=16,
    ) as stream:
        received = list(stream)
    assert stream.result.status == "complete"
    if delivery == "batches":
        assert all(
            isinstance(batch, pa.RecordBatch) and len(batch) <= 7 for batch in received
        )
        actual = pa.Table.from_batches(received)
        expected = pa.Table.from_batches(list(api.read_events(path)))
        assert ordered(actual) == ordered(expected)
    else:
        assert received == list(api.read_events(path, events="individual"))
        assert any(isinstance(event, api.CollisionEvent) for event in received)
        assert isinstance(received[0], api.Event)
        with pytest.raises((AttributeError, TypeError)):
            received[0].t = 123
    with api.mass_spec_iter(
        ms, substance, 3, strict=False, events=delivery
    ) as unrecorded:
        without_file = list(unrecorded)
    assert unrecorded.result.recording is None
    assert len(received) > 0 and len(without_file) > 0
    assert_counters(unrecorded.result.counters, stream.result.counters)


def test_reopen_append_and_explicit_overwrite(prepared_case, tmp_path):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "append.arrow"
    api.mass_spec(ms, substance, 2, strict=False, record_to=path)
    original = list(api.read_events(path, events="individual"))
    result = api.mass_spec(
        ms, substance, 2, strict=False, record_to=api.IpcRecording(path, mode="append")
    )
    assert result.recording.result_id == 2
    combined = list(api.read_events(path, events="individual"))
    assert [event for event in combined if event.result_id == 1] == original
    assert {event.result_id for event in combined} == {1, 2}
    batches = list(api.read_events(path, result_id=1))
    assert batches and all(batch.num_rows for batch in batches)
    assert sum(batch.num_rows for batch in batches) == len(original)
    assert list(api.read_events(path, result_id=999)) == []
    replaced = api.mass_spec(
        ms,
        substance,
        1,
        strict=False,
        record_to=api.IpcRecording(path, mode="overwrite"),
    )
    assert replaced.recording.result_id == 1
    assert {
        event.result_id for event in api.read_events(path, events="individual")
    } == {1}


def test_early_exit_retains_incomplete_events_and_unblocks(prepared_case, tmp_path):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "cancelled.arrow"
    with api.mass_spec_iter(
        ms,
        substance,
        10000,
        strict=False,
        record_to=path,
        initial_batch_size=1,
        max_batch_size=2,
        max_pending_events=8,
    ) as stream:
        next(stream)
    assert stream.result.cancelled
    assert stream.result.status == "cancelled"
    with pa.ipc.open_file(path) as reader:
        assert (
            json.loads(reader.metadata[b"apitofsim:executions"])["1"]["status"]
            == "cancelled"
        )
        assert reader.num_record_batches > 0
    assert not Path(str(path) + ".journal").exists()


@pytest.mark.parametrize("mode", ["append", "overwrite"])
def test_failed_sync_restores_previous_archive(
    prepared_case, tmp_path, monkeypatch, mode
):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "restore.arrow"
    api.mass_spec(ms, substance, 2, strict=False, record_to=path)
    original = path.read_bytes()
    original_sync = persistence.sync_file

    def fail_sync(file):
        if Path(file).name.startswith(path.name):
            raise OSError("simulated disk failure")
        original_sync(file)

    monkeypatch.setattr(persistence, "sync_file", fail_sync)
    with pytest.raises(OSError, match="simulated disk failure"):
        api.mass_spec(
            ms, substance, 1, strict=False, record_to=api.IpcRecording(path, mode)
        )
    assert path.read_bytes() == original
    assert not Path(str(path) + ".journal").exists()


def test_workflow_roots_share_sidecar_and_reopen(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case)
    with db.recording_transaction():
        for group in prepared.groups:
            runner.run_mass_spec(
                prepared.mass_spec,
                group.substance,
                2,
                cluster_id=group.cluster_id,
                pathway_ids=group.pathway_ids,
            )
        with pytest.raises(RuntimeError, match="during a recording"):
            db.events_query()
    assert db.events_path.name == "fixture.events.arrow"
    assert db.db.execute("select generation from event_recording_state").fetchone() == (
        1,
    )
    assert db.events_query().count("*").fetchall()[0][0] > 0
    ids = db.db.execute(
        "select result_id from event_execution order by result_id"
    ).fetchall()
    assert len(ids) == 2
    before = db.events_query(sort=True).fetchall()
    scoped = db.events_query(result_id=ids[0][0], sort=True)
    group = prepared.groups[0]
    runner.run_mass_spec(
        prepared.mass_spec,
        group.substance,
        1,
        cluster_id=group.cluster_id,
        pathway_ids=group.pathway_ids,
        record_events=True,
    )
    assert db.db.execute("select generation from event_recording_state").fetchone() == (
        2,
    )
    assert len(db.events_query(sort=True).fetchall()) > len(before)
    assert scoped.fetchall() == [row for row in before if row[2] == ids[0][0]]
    physical = {
        row[0]
        for row in db.db.execute("select table_name from duckdb_tables()").fetchall()
    }
    assert (
        not {"event_info", "realization", "collision_event", "fragmentation_event"}
        & physical
    )


@pytest.mark.parametrize("delivery", ["batches", "individual"])
def test_workflow_event_stream(prepared_case, delivery):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    with runner.iter_mass_spec(
        prepared,
        prepared.groups[0],
        2,
        events=delivery,
        record_events=True,
        max_batch_items=7,
    ) as stream:
        values = list(stream)
    assert values
    assert stream.result.result.status == "complete"
    assert (
        stream.result.result.recording.event_count
        == db.events_query().count("*").fetchall()[0][0]
    )
    if delivery == "batches":
        assert all(
            isinstance(batch, pa.RecordBatch) and len(batch) <= 7 for batch in values
        )
    else:
        assert all(isinstance(event, api.Event) for event in values)


@pytest.mark.parametrize("seed_events", [False, True])
def test_writer_excludes_other_readers(prepared_case, seed_events):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    group = prepared.groups[0]
    if seed_events:
        runner.run_mass_spec(
            prepared.mass_spec,
            group.substance,
            1,
            cluster_id=group.cluster_id,
            pathway_ids=group.pathway_ids,
            record_events=True,
        )
    filename = db.db.execute("pragma database_list").fetchall()[0][2]
    with ExperimentDatabase(filename) as reader:
        with pytest.raises(portalocker.exceptions.LockException):
            with db.recording_transaction():
                pass
        count = reader.events_query().count("*").fetchall()[0][0]
        assert (count > 0) == seed_events


def test_recover_interrupted_append(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    group = prepared.groups[0]
    runner.run_mass_spec(
        prepared.mass_spec,
        group.substance,
        1,
        cluster_id=group.cluster_id,
        pathway_ids=group.pathway_ids,
        record_events=True,
    )
    path = db.events_path
    original = path.read_bytes()
    db._release_event_reader()
    transaction = persistence.RecordingTransaction(
        api.IpcRecording(path, "append"), database=db
    )
    transaction.__enter__()
    transaction.writer.abort()
    db.db.execute("rollback")
    transaction.started = False
    # Simulate process loss after overwriting part of the ending, before commit.
    with path.open("r+b") as file:
        file.seek(transaction.journal["offset"])
        file.write(b"interrupted native append")
    transaction.lock.release()
    db._ensure_event_reader()
    assert path.read_bytes() == original
    assert not transaction.journal_path.exists()


def test_logical_deletion_does_not_rewrite_archive(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    group = prepared.groups[0]
    runner.run_mass_spec(
        prepared.mass_spec,
        group.substance,
        1,
        cluster_id=group.cluster_id,
        pathway_ids=group.pathway_ids,
        record_events=True,
    )
    size = db.events_path.stat().st_size
    db.db.execute("delete from event_execution")
    assert db.events_query().count("*").fetchone() == (0,)
    assert db.events_path.stat().st_size == size


@pytest.mark.parametrize("mode", ["SINGLE_CLUSTER", "CLUSTER_TREE"])
@pytest.mark.parametrize("delivery", [None, "batches", "individual"])
def test_public_whole_experiment(prepared_case, mode, delivery):
    from apitofsim.workflow import SimulationMode

    db, runner, config, config_id = prepared_case
    runner.start_run(config_id, simulation_mode=SimulationMode[mode])
    prepared = runner.prepare_from_config(config, cached_densityandrate=(1, 2))
    if delivery is None:
        result = runner.run_experiment(prepared, N=2)
    else:
        with runner.iter_experiment(
            prepared, N=2, events=delivery, max_batch_items=7
        ) as stream:
            values = list(stream)
        result = stream.result
        assert values
        if delivery == "individual":
            assert {event.result_id for event in values} == {
                root.result_id for root in result.roots
            }
    assert result.status == "complete"
    assert len(result.roots) == len(prepared.groups) == 2
    if db.record_events:
        assert db.db.execute(
            "select generation from event_recording_state"
        ).fetchone() == (1,)
        assert db.events_query().count("*").fetchall()[0][0] > 0
    else:
        assert db.db.execute(
            "select generation from event_recording_state"
        ).fetchone() == (0,)
        assert not db.events_path.exists()


def test_experiment_early_exit_does_not_launch_remaining_roots(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case)
    with runner.iter_experiment(
        prepared, N=10000, events="individual", max_batch_items=1, record_events=True
    ) as stream:
        first = next(stream)
    assert stream.result.cancelled
    assert len(stream.result.roots) == 1
    assert db.db.execute("select count(*) from event_execution").fetchone() == (1,)
    assert (
        db.events_query()
        .filter(f"result_id = {first.result_id}")
        .count("*")
        .fetchall()[0][0]
        > 0
    )


def test_native_failure_retains_explicitly_failed_recording(prepared_case, tmp_path):
    from dataclasses import replace

    import numpy as np

    ms, substance = inputs(prepared_case)
    bad = replace(
        ms,
        voltages=api.Q_(-np.arange(5.0) * 1000, "V"),
        pressures=api.Q_(np.array([0.001, 0.001]), "Pa"),
        radius_pinhole=None,
    )
    path = tmp_path / "failed.arrow"
    with pytest.raises(api.ApiTofParticleEscapeError) as error:
        api.mass_spec(bad, substance, 8, strict=False, record_to=path)
    assert error.value.recording.status == "failed"
    with pa.ipc.open_file(path) as reader:
        assert (
            json.loads(reader.metadata[b"apitofsim:executions"])["1"]["status"]
            == "failed"
        )
        assert len(reader.read_all()) > 0


def test_workflow_failure_metadata_and_partial_archive(prepared_case):
    from dataclasses import replace

    import numpy as np
    from apitofsim.workflow import recorded_failure

    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    bad = replace(
        prepared.mass_spec,
        voltages=api.Q_(-np.arange(5.0) * 1000, "V"),
        pressures=api.Q_(np.array([0.001, 0.001]), "Pa"),
        radius_pinhole=None,
    )
    prepared = replace(prepared, mass_spec=bad)
    with pytest.raises(api.ApiTofParticleEscapeError) as error:
        with runner.iter_experiment(prepared, N=8, record_events=True) as stream:
            list(stream)
    assert stream.result.status == "failed"
    assert recorded_failure(error.value) is not None
    assert db.db.execute("select status from event_execution").fetchone() == ("failed",)
    assert db.events_query().count("*").fetchall()[0][0] > 0


def test_unicode_path_and_empty_recording(prepared_case, tmp_path):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "mólécule-λ.arrow"
    result = api.mass_spec(ms, substance, 0, record_to=path)
    assert result.status == "complete" and result.recording.event_count == 0
    with pa.ipc.open_file(path) as reader:
        assert reader.num_record_batches == 0
        assert (
            json.loads(reader.metadata[b"apitofsim:executions"])["1"]["status"]
            == "complete"
        )


@pytest.mark.parametrize("mode", ["append", "overwrite"])
def test_sidecar_append_requires_database_owner(prepared_case, mode):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    runner.run_experiment(prepared, N=1, record_events=True)
    # Close our reader so this checks ownership, rather than lock exclusion.
    db._release_event_reader()
    with pytest.raises(ValueError, match="database-owned"):
        api.mass_spec(
            prepared.mass_spec,
            prepared.groups[0].substance,
            1,
            record_to=api.IpcRecording(db.events_path, mode),
        )
    db._ensure_event_reader()
    assert db.events_query().count("*").fetchall()[0][0] > 0


def test_interruption_after_database_commit_keeps_published_archive(prepared_case):
    db, _, _, _ = prepared_case
    db._release_event_reader()
    transaction = persistence.RecordingTransaction(
        api.IpcRecording(db.events_path, "append"), database=db
    )
    transaction.__enter__()
    transaction.writer.set_metadata(json.dumps({"1": {"status": "cancelled"}}))
    transaction.writer.close()
    persistence.sync_file(db.events_path)
    transaction.journal["new_size"] = db.events_path.stat().st_size
    transaction._write_journal()
    db.db.execute(
        "update event_recording_state set generation=?,file_size=?",
        [transaction.generation, transaction.journal["new_size"]],
    )
    db.db.execute("commit")
    # The native archive and DB commit succeeded, but Python did not get to
    # clear `started`. Cleanup must consult the durable DB generation.
    transaction.rollback()
    assert transaction.committed
    with pa.ipc.open_file(db.events_path) as reader:
        assert (
            json.loads(reader.metadata[b"apitofsim:executions"])["1"]["status"]
            == "cancelled"
        )
    db._ensure_event_reader()


@pytest.mark.parametrize("mode", ["append", "overwrite"])
def test_committed_cleanup_can_be_retried(prepared_case, tmp_path, monkeypatch, mode):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "cleanup.arrow"
    api.mass_spec(ms, substance, 1, strict=False, record_to=path)
    unlink = Path.unlink

    def fail_journal_cleanup(file, *args, **kwargs):
        if str(file).endswith(".journal"):
            raise OSError("simulated cleanup failure")
        return unlink(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_journal_cleanup)
        result = api.mass_spec(
            ms, substance, 1, strict=False, record_to=api.IpcRecording(path, mode)
        )
    assert result.status == "complete"
    api.mass_spec(
        ms, substance, 1, strict=False, record_to=api.IpcRecording(path, "append")
    )
    assert not list(tmp_path.glob("*.journal"))


def test_workflow_io_failure_rolls_back_append(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    group = prepared.groups[0]

    def run():
        return runner.run_mass_spec(
            prepared.mass_spec,
            group.substance,
            1,
            cluster_id=group.cluster_id,
            pathway_ids=group.pathway_ids,
            record_events=True,
        )

    run()
    original = db.events_path.read_bytes()
    state = db.db.table("event_recording_state").fetchall()
    count = db.events_query().count("*").fetchone()

    def fail_sync(file):
        raise OSError("simulated sidecar sync failure")

    monkeypatch.setattr(persistence, "sync_file", fail_sync)
    with pytest.raises(OSError, match="simulated sidecar sync failure"):
        run()
    assert db.events_path.read_bytes() == original
    assert db.db.table("event_recording_state").fetchall() == state
    assert db.events_query().count("*").fetchone() == count


def test_recorded_roots_cannot_share_writer_concurrently(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    group = prepared.groups[0]
    with db.recording_transaction():
        with runner.iter_mass_spec(prepared, group, 1) as first:
            with pytest.raises(RuntimeError, match="Finish the active root"):
                runner.iter_mass_spec(prepared, group, 1)
            for _ in first:
                pass
        with runner.iter_mass_spec(prepared, group, 1) as second:
            for _ in second:
                pass
    assert first.result.result.status == second.result.result.status == "complete"


def test_writer_lease_rejects_overlapping_runs_and_finalization(
    prepared_case, tmp_path
):
    ms, substance = inputs(prepared_case)
    path = tmp_path / "leased.arrow"
    with persistence.RecordingTransaction(path) as transaction:
        writer = transaction.writer

        def start(result_id):
            return api._mass_spec_batches(
                ms,
                substance,
                8,
                strict=False,
                logconf=(0, True),
                initial_batch_size=1,
                max_batch_size=1,
                queue_capacity=1,
                _stream=api.apitofsimraw.StreamDelivery(
                    writer, result_id, True, True, False, False
                ),
            )

        first = start(1)
        try:
            for action in (
                lambda: start(2),
                transaction.finish,
                transaction.rollback,
                lambda: transaction.__exit__(None, None, None),
                lambda: writer.rows_written,
                lambda: writer.set_metadata("{}"),
            ):
                with pytest.raises(api.apitofsimraw.IpcWriterBusy):
                    action()
            assert transaction.writer is writer and not transaction.done
        finally:
            first.close()
        first.close()
        first_range = first.recording_range
        second = start(2)
        try:
            for _ in second:
                pass
        finally:
            second.close()
        assert second.recording_range[0] == first_range[1]
        assert first.recording_range == first_range
    with pa.ipc.open_file(path) as reader:
        assert reader.num_record_batches == first_range[1] + second.recording_range[1]


@pytest.mark.parametrize("workflow", [False, True])
def test_batch_conversion_failure_stops_recording(
    prepared_case, tmp_path, monkeypatch, workflow
):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    group = prepared.groups[0]
    path = db.events_path if workflow else tmp_path / "failed-conversion.arrow"
    if workflow:
        stream = runner.iter_mass_spec(
            prepared,
            group,
            8,
            strict_dos=False,
            record_events=True,
            max_batch_items=1,
            max_pending_events=1,
        )
        transaction = db._recording
    else:
        stream = api.mass_spec_iter(
            prepared.mass_spec,
            group.substance,
            8,
            strict=False,
            record_to=path,
            initial_batch_size=1,
            max_batch_size=1,
            max_pending_events=1,
        )
        transaction = stream.recording

    def fail_conversion(self, batch):
        with pytest.raises(api.apitofsimraw.IpcWriterBusy):
            transaction.writer.close()
        raise ValueError("injected batch conversion failure")

    monkeypatch.setattr(api._MassSpecBatchIterator, "_convert_batch", fail_conversion)
    with pytest.raises(ValueError, match="injected batch conversion failure"):
        with stream:
            next(stream)
    assert transaction.committed
    with pa.ipc.open_file(path) as reader:
        states = json.loads(reader.metadata[b"apitofsim:executions"])
        assert list(states.values())[0]["status"] == "failed"
        assert list(states.values())[0]["event_count"] == len(reader.read_all()) > 0


def test_overwrite_preserves_unmanaged_staging_file(tmp_path):
    path = tmp_path / "events.arrow"
    staging = Path(str(path) + ".replacement")
    path.write_bytes(b"original destination")
    staging.write_bytes(b"unrelated file")
    with pytest.raises(FileExistsError, match="Unmanaged recording staging file"):
        with persistence.RecordingTransaction(api.IpcRecording(path, "overwrite")):
            pass
    assert path.read_bytes() == b"original destination"
    assert staging.read_bytes() == b"unrelated file"


def test_failed_reader_open_closes_connection(tmp_path):
    import duckdb
    from archive_fixtures import publish_events

    path = tmp_path / "missing-sidecar.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        publish_events(db, [])
        sidecar = db.events_path
    sidecar.unlink()

    connection = duckdb.connect(path, read_only=True)
    cleaned = []

    def cleanup():
        with pytest.raises(duckdb.ConnectionException, match="closed"):
            connection.execute("select 1")
        cleaned.append(True)

    with pytest.raises(FileNotFoundError, match="matching .events.arrow") as error:
        ExperimentDatabase(
            (connection, cleanup), readonly=True, events_filename=sidecar
        )
    assert error.value.__traceback__ is not None
    assert cleaned == [True]
    with duckdb.connect(path) as writable:
        assert writable.execute("select 1").fetchone() == (1,)


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_rolled_back_failure_metadata_is_cleared(
    prepared_case, monkeypatch, outer_transaction
):
    from contextlib import nullcontext
    from dataclasses import replace

    import numpy as np
    from apitofsim.workflow import recorded_failure

    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    prepared = replace(
        prepared,
        mass_spec=replace(
            prepared.mass_spec,
            voltages=api.Q_(-np.arange(5.0) * 1000, "V"),
            pressures=api.Q_(np.array([0.001, 0.001]), "Pa"),
            radius_pinhole=None,
        ),
    )

    def fail_sync(file):
        raise OSError("simulated sidecar sync failure")

    monkeypatch.setattr(persistence, "sync_file", fail_sync)
    scope = db.recording_transaction() if outer_transaction else nullcontext()
    with pytest.raises(api.ApiTofParticleEscapeError) as error:
        with scope:
            with runner.iter_mass_spec(
                prepared, prepared.groups[0], 8, record_events=True
            ) as stream:
                list(stream)

    assert any(
        "simulated sidecar sync failure" in note for note in error.value.__notes__
    )
    assert recorded_failure(error.value) is None
    assert stream.failure is None
    assert error.value.recording is None
    assert db.db.table("experiment_failure").count("*").fetchone() == (0,)
