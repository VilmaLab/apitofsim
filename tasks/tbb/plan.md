# OpenMP runtime to oneTBB migration plan

## Goal

Replace OpenMP runtime parallelism with oneAPI Threading Building Blocks (oneTBB) while preserving public APIs, numerical results, streamed progress, exception propagation, and signal handling. Keep `#pragma omp simd` directives as compiler hints and enable them, where supported, with `-fopenmp-simd` without linking an OpenMP runtime.

The detailed implementation checklist is in [`todo.md`](todo.md).

## Current behavior and migration scope

Runtime OpenMP is used in four areas:

| Area | Current mechanism | Target |
|---|---|---|
| Density/rate kernels | `parallel for`, array reductions, dynamic scheduling | `parallel_for`, `parallel_reduce`, suitable partitioners |
| Independent density calculations | `parallel sections` | `parallel_invoke` |
| Mass-spec realizations | guided `parallel for`, Eigen counter reduction | range-based TBB algorithm with task-local reduction state |
| Synchronization/runtime queries | `critical`, `atomic`, thread IDs/counts | TBB cancellation plus standard C++ synchronization; no scheduler-thread identity in results |

Eleven standalone SIMD loops, plus the SIMD part of one combined `parallel for simd`, remain OpenMP SIMD. OpenMP reduction declarations, `OMP_VISIBILITY_NONE`, `omp_get_*`, and all non-SIMD runtime pragmas are removed.

Build and packaging scope includes Meson options/targets, developer native files, GitHub Actions, wheel repair, the Conda recipe, debug output, and the current single-thread test library.

## Compatibility contract

- Keep C++ and Python function signatures, `MeshMode` values/names, queue/event variants, and CLI behavior.
- With a fixed seed, preserve per-realization outcomes and final integer counters. Log/event arrival order is already scheduler-dependent and remains unspecified.
- Preserve numerical equivalence to the serial implementations within explicit floating-point tolerances. Do not promise bitwise equality for floating-point reductions.
- A task-body exception cancels pending work and is rethrown on the thread that invoked the top-level API. Exceptions crossing the existing background `std::thread` boundary are still transported to its caller.
- `SIGINT` and `SIGTERM` retain precedence over an application exception. Restore the previous handler before re-raising a signal on the owning/calling thread.
- Do not install a `SIGABRT` handler. Abort remains an immediate fatal condition under the process's existing/default handling rather than entering cooperative cancellation.
- Once cancellation is observed, do not emit new progress callbacks or partial-result messages. Already-running work exits at explicit checkpoints rather than waiting for an entire large batch.
- Replace worker-indexed cumulative partial counters with scheduler-independent progress data. Consumers must not allocate arrays from a maximum worker count.
- Do not read or emulate `OMP_NUM_THREADS`, and do not add a replacement environment variable. Use oneTBB's default scheduler; tests and explicitly constrained callers use scoped `task_arena`/`global_control` APIs.

## Target design

### Parallel algorithms

- Use `oneapi::tbb::parallel_for` for independent output columns/realizations and `parallel_invoke` for the four independent density calculations.
- Use `parallel_reduce` only where a real reduction exists, notably the mesh and final mass-spec counters. Prefer range-local accumulators and a clear join operation over shared atomics.
- Start with TBB's automatic partitioning. Benchmark before selecting explicit grain sizes or partitioners; OpenMP `guided`/`dynamic,1` clauses are performance policies, not compatibility requirements.
- Keep SIMD inside each TBB range's serial inner loop. This retains vectorization while TBB owns task scheduling.

### Cancellation, signals, and exceptions

Split the current `OMPExceptionHelper` responsibilities:

1. An operation-scoped interrupt/cancellation object owns a `task_group_context`, records the first pending signal using a verified signal-safe primitive, exposes a cheap checkpoint, and restores handlers through RAII.
2. A small exception transport remains only where an exception must cross the explicit background `std::thread` used by the CLI/Python streaming API.
3. Exceptions inside TBB bodies are allowed to escape. oneTBB then cancels pending tasks and rethrows at the algorithm call site; do not catch every iteration merely to emulate OpenMP.
4. The top-level caller owns signal installation and re-raising. Inner mass-spec routines receive the operation context instead of installing nested process-wide handlers.

Checkpoints belong before expensive iterations, inside the long mass-spec trajectory loop, and before callbacks/queue writes. The signal handler itself only records state; it must not call TBB or other non-signal-safe code.

Only `SIGINT` and `SIGTERM` participate in this mechanism. `SIGABRT` is deliberately excluded from handler installation and cooperative cancellation.

### Progress and counters

Do not replace `omp_get_thread_num()` with a TBB arena index: TBB tasks are not owned by stable application-visible worker IDs. Use per-range reduction state and emit one counter delta for each completed realization, with these requirements:

- each completed realization contributes exactly once;
- streamed totals are monotonic and never exceed final totals;
- final counters do not depend on scheduling or concurrency;
- no progress/counter update occurs after cancellation is observed.

Serialize the `compute_k_total_batch` callback and pass strictly increasing completed counts. This improves the current best-effort “only OpenMP thread 0 reports” behavior while retaining its absolute-count callback contract.

### Build and distribution

- Require oneTBB `>=2023.0` through Meson's normal dependency mechanisms, validating pkg-config on Unix and the exported `TBB::tbb` CMake target where needed.
- Make the SIMD flag compiler-checked. Unsupported compilers may ignore the pragmas; they must not acquire an OpenMP runtime dependency.
- Remove `libapitofsim_no_omp` as soon as oneTBB build plumbing is available. Link generators and tests to the normal library and constrain them with a TBB arena/global control.
- Remove `openmp`, the Intel-specific OpenMP branch, and LLVM OpenMP packages immediately after the last runtime pragma is ported; there is no dual-runtime bake period.
- Add oneTBB development/runtime packages to CI, wheel, and Conda builds, and verify that produced wheels contain or declare the required TBB runtime correctly.

## Task list

### Phase 1: Foundation

- Tasks 1-3: characterize behavior, establish oneTBB `2023.0` build integration, remove the duplicate serial library, and establish operation-scoped cancellation.

### Phase 2: Runtime cutover

- Task 4: port density/rate algorithms.
- Task 5: decouple streamed counters from OpenMP workers.
- Task 6: port mass-spec realization loops.

### Phase 3: Removal and distribution

- Tasks 7-9: remove source/build-time OpenMP runtime support and stale developer/CI configuration.
- Tasks 10-11: update distribution artifacts, benchmark, document, and audit.

## Dependency order

```text
Characterization tests
        |
        v
TBB/SIMD build + remove duplicate library ---> operation cancellation context
        |                         |
        +------------+------------+
                     v
       density/rate port and mass-spec port
                     |
                     v
       remove OpenMP runtime/build plumbing
                     |
                     v
       packaging matrix, performance, final audit
```

The density/rate and mass-spec ports can proceed independently after the operation-context contract is fixed. Build cleanup must wait until both are complete.

## Checkpoints

### Foundation

- Existing OpenMP behavior is characterized, including subprocess signal tests.
- oneTBB `>=2023.0` is discoverable on supported toolchains, the duplicate serial library is gone, and SIMD compiles without an OpenMP runtime.
- The operation cancellation/exception contract is covered by focused tests.

### Runtime cutover

- Density/rate and mass-spec paths use TBB with numerical and streaming parity.
- Cancellation stops pending TBB tasks and reaches safe points in active long-running bodies.
- No consumer depends on an OpenMP worker ID or maximum thread count.

### Completion

- `rg` finds no OpenMP runtime APIs/pragmas/dependencies; only SIMD pragmas, SIMD flags, and explanatory comments remain.
- GCC and Clang builds, C++/Python tests, sanitizer/Valgrind jobs, wheels, and Conda packaging pass.
- Benchmarks show no unexplained material regression at one thread or default parallelism.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Signal handlers are process-wide while operation contexts are scoped | High | Install only at top-level; make nesting/concurrency policy explicit; test handler restoration and overlapping operations |
| TBB cancellation does not preempt a running function body | High | Add cheap bounded checkpoints inside long loops and before side effects |
| Counter snapshots assume stable OpenMP worker IDs | High | Move to logical states or deltas and test exactly-once aggregation |
| Reduction order changes floating-point results | Medium | Compare against serial results with domain tolerances and document non-bitwise behavior |
| Different scheduling changes stream order | Medium | Test message contents/counts, not ordering that was never guaranteed |
| TBB shared scheduling changes nested/concurrent performance | Medium | Use automatic partitioning first; benchmark concurrent Python calls and avoid unnecessary isolated arenas |
| TBB runtime is missing from wheels/Conda artifacts | High | Inspect repaired artifacts and import/run them in clean environments |
| oneTBB 2023.0 is unavailable in a supported build environment | High | Fail configuration clearly and verify every CI/package platform during the foundation task |

## Decisions fixed for implementation

1. **Minimum version:** require oneTBB `2023.0` or newer.
2. **Single implementation:** remove `libapitofsim_no_omp`; there is no downstream consumer and no alternate serial backend.
3. **Thread limits:** drop `OMP_NUM_THREADS` behavior without a compatibility shim. oneTBB owns default scheduling, with standard oneTBB controls available to constrained callers/tests.
4. **Fatal abort:** never intercept `SIGABRT`; leave immediate abort handling to the process/runtime.

## Teardown order

1. Remove `libapitofsim_no_omp` when the oneTBB dependency and one-thread test control land.
2. Replace `openmp_helper` and `OMPExceptionHelper` when the generic operation context lands; retain only the minimum temporary exception guard needed by unported OpenMP loops.
3. Delete each runtime pragma/reduction as its owning algorithm moves to TBB.
4. Remove the OpenMP dependency, option, profiles, and packages in the immediately following cleanup sequence, with no dual-runtime release or bake period.

## Authoritative references

- [oneTBB exceptions and cancellation](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Exceptions_and_Cancellation.html)
- [`task_group_context` cancellation contract](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/scheduling_controls/task_group_context_cls.html)
- [Cancellation and nested parallelism](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Cancellation_and_Nested_Parallelism.html)
- [`parallel_for`](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/parallel_for_os.html), [`parallel_reduce`](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/parallel_reduce.html), and [`parallel_invoke`](https://uxlfoundation.github.io/oneTBB/main/specification/source/algorithms/functions/parallel_invoke_func.html)
- [Concurrency controls](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Migration_Guide/Task_Scheduler_Init.html)
- [TBB work isolation and thread-local caveats](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/work_isolation.html)
- [Meson dependency discovery](https://mesonbuild.com/Dependencies.html)
- [oneTBB CMake package targets](https://github.com/uxlfoundation/oneTBB/blob/master/cmake/README.md)
