"""Public event delivery: native batches, optional IPC recording, typed rows."""

from collections import deque
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from .recording import RecordingInfo, RecordingTransaction

if TYPE_CHECKING:
    from .api import Counters, Timings


class EventKind(IntEnum):
    init = 0
    collision = 1
    postcollision = 2
    fragmentation = 3
    postfragmentation = 4
    escape = 5


@dataclass(frozen=True, slots=True)
class Event:
    result_id: int
    realization_index: int
    event_index: int
    particle_index: int
    x: float
    y: float
    z: float
    t: float
    vx: float
    vy: float
    vz: float
    ox: float
    oy: float
    oz: float
    rot_energy: float
    vibrational_energy: float
    kind: ClassVar[EventKind]


@dataclass(frozen=True, slots=True)
class InitEvent(Event):
    kind: ClassVar[EventKind] = EventKind.init


@dataclass(frozen=True, slots=True)
class CollisionEvent(Event):
    theta: float
    u_norm: float
    accepted: bool
    kind: ClassVar[EventKind] = EventKind.collision


@dataclass(frozen=True, slots=True)
class PostCollisionEvent(Event):
    kind: ClassVar[EventKind] = EventKind.postcollision


@dataclass(frozen=True, slots=True)
class FragmentationEvent(Event):
    transition_index: int
    next_particle_index: int
    kind: ClassVar[EventKind] = EventKind.fragmentation


@dataclass(frozen=True, slots=True)
class PostFragmentationEvent(Event):
    kind: ClassVar[EventKind] = EventKind.postfragmentation


@dataclass(frozen=True, slots=True)
class EscapeEvent(Event):
    kind: ClassVar[EventKind] = EventKind.escape


EVENT_CLASSES = (
    InitEvent,
    CollisionEvent,
    PostCollisionEvent,
    FragmentationEvent,
    PostFragmentationEvent,
    EscapeEvent,
)
EVENT_FIELDS = tuple(
    tuple(field.name for field in fields(cls)) for cls in EVENT_CLASSES
)


def event_at(batch, index):
    kind = batch.column("event_kind")[index].as_py()
    values = {name: batch.column(name)[index].as_py() for name in EVENT_FIELDS[kind]}
    if kind == EventKind.collision:
        values["accepted"] = bool(values["accepted"])
    return EVENT_CLASSES[kind](**values)


@dataclass(frozen=True)
class MassSpecResult:
    counters: "Counters"
    timings: "Timings"
    cancelled: bool = False
    recording: RecordingInfo | None = None
    status: str = "complete"


def need_ordered_delivery(ordered: bool | None, recording: bool) -> bool:
    if recording and ordered is False:
        raise ValueError("Event recording requires ordered delivery")
    return recording or bool(ordered)


class MassSpecIterator:
    """A single lossless Python consumer, independent of the native IPC sink.

    Exhaustion finishes owned persistence. Early context exit cooperatively
    cancels, discards further Python delivery, then joins. User
    exceptions propagate; successful cleanup does not turn them into success.
    """

    def __init__(
        self,
        source,
        *,
        events: str | None = "batches",
        recording=None,
        owns_recording=False,
        result_id=0,
        result_callback=None,
        log_callback=None,
    ):
        if events not in (None, "batches", "individual"):
            raise ValueError("events must be batches or individual")
        self.source = source
        self.events = events
        self.recording = recording
        self.owns_recording = owns_recording
        self.result_id = result_id
        self.result_callback = result_callback
        self.log_callback = log_callback
        self.result: MassSpecResult | None = None
        self.failure: BaseException | None = None
        self._ready = deque()
        self._row_batch = None
        self._row = 0
        self._done = False

    def request_cancel(self):
        self.source.request_cancel()

    def __iter__(self):
        return self

    def _completion(self, final=None, *, error=None):
        from .api import Timings, counters_named_tuple

        status = (
            "failed"
            if error is not None
            else "cancelled"
            if self.source.cancelled
            else "complete"
        )
        info = None
        transaction = self.recording
        if transaction is not None:
            batch_start, batch_count, event_count = self.source.recording_range
            info = RecordingInfo(
                transaction.path,
                transaction.archive_id,
                self.result_id,
                event_count,
                batch_start,
                batch_count,
                status,
            )
            transaction.executions[str(self.result_id)] = dict(
                status=status,
                event_count=info.event_count,
                batch_start=info.batch_start,
                batch_count=info.batch_count,
                ordered=True,
            )
            if self.owns_recording:
                try:
                    transaction.finish()
                except BaseException as failure:
                    self._done = True
                    self.failure = failure
                    transaction.rollback()
                    raise
                finally:
                    transaction.__exit__(None, None, None)
        counters = (
            final.counters
            if final is not None
            else counters_named_tuple(self.source.partial_counters)
        )
        timings = final.timings if final is not None else Timings(None, None)
        self.result = MassSpecResult(
            counters, timings, status == "cancelled", info, status
        )
        self._done = True
        if error is not None:
            error.__dict__["recording"] = info
            if transaction is not None:
                transaction.failure_rollbacks.append(
                    lambda: setattr(error, "recording", None)
                )

    def _read(self, *, discard=False):
        from .api import (
            MassSpecFinalResult,
            MassSpecIntermediateCounter,
            MassSpecLogItem,
        )

        try:
            batch = next(self.source)
        except BaseException as error:
            if isinstance(error, StopIteration):
                error = RuntimeError("Simulation ended without a final result")
            self._fail(error)
            raise error
        for item in batch.items:
            if isinstance(item, MassSpecFinalResult):
                self._completion(item)
            elif isinstance(item, MassSpecIntermediateCounter) and self.result_callback:
                self.result_callback(item.counters)
            elif isinstance(item, MassSpecLogItem) and self.log_callback:
                self.log_callback(item.type, item.name)
        if not discard and self.events is not None and len(batch.events):
            table = batch.events.rename_columns(
                [
                    "realization_index" if name == "realization" else name
                    for name in batch.events.column_names
                ]
            )
            self._ready.extend(table.to_batches())

    def __next__(self):
        if self.events == "individual" and self._row_batch is not None:
            event = event_at(self._row_batch, self._row)
            self._row += 1
            if self._row == self._row_batch.num_rows:
                self._row_batch = None
            return event
        while not self._ready:
            if self._done:
                raise StopIteration
            self._read()
        batch = self._ready.popleft()
        if self.events == "individual":
            self._row_batch, self._row = batch, 0
            return next(self)
        return batch

    def finish(self):
        """Drain remaining delivery and wait for native execution/persistence."""
        self._ready.clear()
        self._row_batch = None
        while not self._done:
            self._read(discard=True)
        return self.result

    def _fail(self, error):
        self.source.close()
        self.failure = error
        self._done = True
        try:
            self._completion(error=error)
        except BaseException as cleanup:
            error.add_note(f"Finishing event recording: {cleanup}")

    def close(self):
        """Cancel and wait for native writing, without converting queued batches."""
        self._ready.clear()
        self._row_batch = None
        if not self._done:
            self.source.close()
            try:
                final = self.source.final_item()
            except BaseException as error:
                self._fail(error)
                raise
            self._completion(final)
        return self.result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Simulation cleanup: {cleanup}")


def start_stream(
    ms,
    substance,
    N,
    *,
    events: str | None = "batches",
    record_to=None,
    ordered=None,
    result_callback=None,
    log_callback=None,
    max_pending_events=262144,
    **options,
):
    from .api import _mass_spec_batches
    from .apitofsimraw import StreamDelivery

    if events not in (None, "batches", "individual"):
        raise ValueError("events must be batches or individual")
    from operator import index

    N = index(N)
    max_pending_events = index(max_pending_events)
    if N < 0 or max_pending_events < 1:
        raise ValueError(
            "N must be nonnegative and max_pending_events must be positive"
        )
    ordered = need_ordered_delivery(ordered, record_to is not None)
    transaction = None
    if record_to is not None:
        transaction = RecordingTransaction(record_to)
        transaction.__enter__()
    source = None
    try:
        loglevel = options.pop("loglevel", 0)
        result_id = transaction.next_result_id if transaction is not None else 0
        source = _mass_spec_batches(
            ms,
            substance,
            N,
            logconf=(loglevel, transaction is not None or events is not None),
            queue_capacity=max_pending_events,
            _stream=StreamDelivery(
                writer=transaction.writer if transaction is not None else None,
                result_id=result_id,
                events=events is not None,
                ordered=ordered,
                progress=result_callback is not None,
                logs=log_callback is not None,
            ),
            **options,
        )
        return MassSpecIterator(
            source,
            events=events,
            recording=transaction,
            owns_recording=transaction is not None,
            result_id=result_id,
            result_callback=result_callback,
            log_callback=log_callback,
        )
    except BaseException:
        if source is not None:
            source.close()
        if transaction is not None:
            transaction.rollback()
        raise


def read_events(path, *, events="batches", result_id=None):
    """Read a completed standalone archive with the same batch/row vocabulary."""
    from .recording import RecordingTransaction, archive_metadata, recording_lock

    if events not in ("batches", "individual"):
        raise ValueError("events must be batches or individual")
    from pathlib import Path

    if Path(str(path) + ".journal").exists():
        with recording_lock(path):
            RecordingTransaction.recover(path)
    with recording_lock(path, shared=True):
        archive_metadata(path)
        with pa.OSFile(str(path), "rb") as source, pa.ipc.open_file(source) as reader:
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                if result_id is not None:
                    import pyarrow.compute as pc

                    batch = batch.filter(
                        pc.call_function(
                            "equal",
                            [
                                batch.column("result_id"),
                                pa.scalar(result_id, pa.int64()),
                            ],
                        )
                    )
                if events == "batches":
                    if batch.num_rows:
                        yield batch
                else:
                    for j in range(batch.num_rows):
                        yield event_at(batch, j)
