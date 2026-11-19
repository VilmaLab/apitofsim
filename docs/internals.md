# 🤖 Internals

Execution and event-storage design. See [Development environment](development.md)
for build and test instructions, and [Benchmarks](benchmark.md) for recording
performance measurements.

## Execution and shutdown

Parallel work is scheduled by oneTBB. Applications that need a scoped limit
should use `oneapi::tbb::global_control` or `oneapi::tbb::task_arena`. The
single-threaded build uses the same `global_control` mechanism internally.
`OMP_NUM_THREADS` has no effect. CMake enables the remaining OpenMP SIMD hints
with `-fopenmp-simd` when supported by the compiler.

The native Python extension retains a oneTBB scheduler handle and finalizes it
through `Py_AtExit`, before native library teardown. This also covers direct
extension imports. If other scheduler users prevent finalization, it prints a
native diagnostic to stderr. Unfinished simulation work still needs to
complete before interpreter shutdown. Cooperative application cleanup is
tracked in [issue #85](https://github.com/VilmaLab/apitofsim/issues/85).

Long-running simulations cooperatively cancel on `SIGINT` and `SIGTERM`,
restore the previous signal handler, and then re-raise the signal on the
calling thread. `SIGABRT` is intentionally not intercepted and remains
immediately fatal.

## Event recording

The Parquet recorder writes `EventColumns` vectors directly through carquet's
C API, using the existing bounded batch queue and recording transactions. It
caps row groups at 65,536 rows and enables dictionary encoding, statistics,
archive metadata and Zstandard level 3 where requested. `parquet_column_workers`
controls carquet encoding concurrency through the OpenMP setting on the
dedicated writer thread; it does not change TBB's simulation concurrency.
Carquet also uses its default asynchronous I/O thread. Nanoarrow retains the
in-place IPC append path and execution metadata in the footer.

`EventWriter` is the format-neutral recording session (`IpcEventWriter` remains
an alias for compatibility). `event_recording.cpp` contains its lease, lifecycle,
failure state and counters. Only the selected codec is constructed:
`arrow_event_writer.cpp` owns nanoarrow IPC encoding/append/footer state, while
`parquet_event_writer.cpp` owns carquet and its bounded asynchronous queue.

`EventCollator` assembles columns and ordinals in shared, non-template code.
Only its batch dispatch policy is templated on the writer and optional
`CallbackArrow`/`NullCallbackArrow` delivery. IPC encoding borrows a batch
synchronously. Parquet-only recording transfers a `unique_ptr` to its queue;
IPC plus Python delivery encodes first, then transfers ownership to delivery.
Only Parquet plus Python delivery promotes the batch to
`shared_ptr<const EventColumns>` for the two independent consumers. All IDs are
filled before dispatch, and published batches are immutable. Summary-only runs
allocate no event columns or ordinal arrays.

Python arrays retain their `MassSpecBatch` wrapper, which owns the batch.
Repeated exports and retained arrays remain valid after iterator destruction.
The raw iterator keeps two Python-owned batch wrappers for reuse, checking that
the pool is the sole owner before mutating their unique column storage.

`EventColumns` is the common native column layout. A oneTBB pipeline assigns
realization IDs, simulates histories in parallel, and delivers completed tokens
through a serial output stage. Recording uses in-order output and the stage
coalesces events into format-independent batches. Final-only summaries can group
realizations and reduce counters without collecting events. Python iterators
use a bounded handoff for events and callbacks. The raw iterator retains its eager queue
for join-before-read callers. Retained record batches own their buffers. The IPC
writer borrows the event vectors for synchronous encoding.

Managed streams acquire an exclusive event writer lease before starting the
worker. The run releases it after its last write and batch-range
snapshot. An active lease prevents a second run, writer finalization, and
reads of mutable writer counters. Stream shutdown discards pending Python
delivery, wakes the producer, and joins it before transaction cleanup.

A database owns one event sidecar. Arrow IPC has fixed primitive fields,
a schema version/archive UUID, and per-execution footer metadata. An appender
verifies the previous footer, retains its batch descriptors, then replaces only
the EOS/footer and adds new batches. Existing event bodies are never copied.
The Parquet writer streams the same columns into a fresh `.events.parquet` file.

Publication uses an exclusive portable sidecar lock and a durable small journal
containing the old ending and committed length. Event data, its footer and parent
directory are synced before committing DuckDB's archive generation/length and
execution metadata. Reopening compares generations and either retains the
committed file, restores the old IPC ending, or removes an uncommitted
Parquet file.
Standalone files use a committed journal marker; explicit overwrite stages an
atomic replacement. A journal is recovery state, not a second event store.

DuckDB keeps execution/root/tree/counter metadata. It does not insert or allocate
IDs for event/realization rows. PyArrow IPC datasets and built-in Parquet
scans supply temporary DuckDB event views. SQL IDs are opaque composite
strings; numeric `result_id`, `realization_index`, `event_index` define order.
Deletion is logical until explicit compaction; a schema change requires migration.

Downstream changes required before updating their dependency pins:

- `apitofsim-resultviewer` must use `ExperimentDatabase`, transport/open the event
  sidecar, and sort on numeric identity components instead of opaque event IDs.
- `apitofsim-web` must replace `RealizationDatabase` construction, pass an explicit
  events filename for attached/in-memory result connections, and store/download
  the sidecar alongside DuckDB. Shared-cluster attachments remain supported.

The old relational-event schema and `EventRecorder` are removed. Historical
schemas fail clearly and require an explicit migration; this PR provides none.
