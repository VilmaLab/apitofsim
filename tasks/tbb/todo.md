# oneTBB migration task list

## Task 1: Characterize existing parallel behavior

**Description:** Add focused regression coverage before changing runtimes. Exercise currently untested parallel builds, numerical parity, progress/counter streaming, task exceptions, and signal forwarding in subprocesses.

**Acceptance criteria:**

- [ ] Fixed-seed and serial/parallel numerical expectations are recorded with explicit tolerances.
- [ ] Exceptions and `SIGINT`/`SIGTERM`/`SIGABRT` behavior are tested without terminating the test runner.
- [ ] Progress and partial-counter tests capture monotonicity, final totals, and cancellation side-effect boundaries.

**Verification:**

- [ ] `uv run meson test --print-errorlogs -C build`
- [ ] The signal tests assert child exit/exception behavior and handler restoration.

**Dependencies:** None

**Files likely touched:** `test/main.cpp`, `test/test_python.py`, `test/meson.build`, new focused test helpers under `test/`

**Estimated scope:** Medium

## Task 2: Prove oneTBB and SIMD-only build plumbing

**Description:** Add oneTBB discovery alongside the existing runtime temporarily, compile a minimal TBB target on supported toolchains, and add compiler-checked `-fopenmp-simd` without relying on the OpenMP runtime for SIMD-only translation units.

**Acceptance criteria:**

- [ ] Meson discovers oneTBB through tested pkg-config/CMake paths and records a justified minimum version.
- [ ] GCC and Clang compile/link a TBB smoke target; a SIMD-only target has no OpenMP runtime linkage.
- [ ] Linux, wheel, Windows, and Conda dependency names/availability are documented for later packaging work.

**Verification:**

- [ ] `meson setup --wipe build && meson compile -C build`
- [ ] Inspect a SIMD-only test binary with the platform's dynamic dependency tool.

**Dependencies:** Task 1

**Files likely touched:** `meson.build`, `meson_options.txt`, `test/meson.build`, optional `subprojects/` wrap metadata

**Estimated scope:** Medium

## Task 3: Introduce operation-scoped cancellation and exception transport

**Description:** Replace the mixed responsibilities of `OMPExceptionHelper` with generic operation cancellation/signal handling and a separate cross-`std::thread` exception transport. Keep the OpenMP call sites working until their own cutover.

**Acceptance criteria:**

- [ ] A top-level RAII scope restores prior handlers on success, exception, and cancellation; nested use does not overwrite saved handlers.
- [ ] A shared `task_group_context` can be cancelled at a safe checkpoint, with first-signal precedence preserved.
- [ ] Background mass-spec exceptions/signals still arrive at the invoking CLI/Python thread, and no exception can be silently destroyed.

**Verification:**

- [ ] Focused C++ cancellation/exception tests pass under normal and sanitizer builds.
- [ ] Python subprocess tests deliver signals during active work and terminate within the stated bound.

**Dependencies:** Task 2

**Files likely touched:** replacements for `src/include/openmp_helper.h` and `src/openmp_helper.cpp`, `python/apitofsim.cpp`, `src/cli/mass_spec.cpp`, `test/`

**Estimated scope:** Medium

## Checkpoint: Foundation

- [ ] Tasks 1-3 pass with the old runtime paths still available.
- [ ] The cancellation and thread-limit compatibility contracts have review approval.

## Task 4: Port density and rate parallelism

**Description:** Convert mesh, batch, and independent-section parallelism in `densityandrate.cpp` to `parallel_for`, `parallel_reduce`, and `parallel_invoke`; retain SIMD inside serial range bodies.

**Acceptance criteria:**

- [ ] All density/rate OpenMP runtime directives and array reduction declarations are gone; SIMD directives remain where valid.
- [ ] Serial and TBB results meet the characterization tolerances across mesh modes and batch sizes including zero/one item.
- [ ] The progress callback is serialized, strictly increasing, finishes at the batch size, and stops after cancellation.

**Verification:**

- [ ] Focused density/rate C++ and Python tests pass at concurrency 1 and default concurrency.
- [ ] `meson benchmark -C build density_of_states compute_k_total` records before/after results.

**Dependencies:** Task 3

**Files likely touched:** `src/densityandrate.cpp`, `src/include/densityandrate.h`, `test/main.cpp`, `test/test_python.py`, `test/bench/`

**Estimated scope:** Medium

## Task 5: Decouple streamed counters from worker threads

**Description:** Change partial results from OpenMP-worker-indexed cumulative snapshots to scheduler-independent deltas, and update all queue consumers to aggregate those deltas directly. Keep the OpenMP producer temporarily so this can land and be verified before the loop port.

**Acceptance criteria:**

- [ ] Each completed realization emits exactly one counter delta and consumers aggregate it exactly once.
- [ ] Python and CLI consumers no longer allocate a matrix from `omp_get_max_threads()` or key state by a worker ID.
- [ ] Streamed totals are monotonic, match final totals, and remain unchanged after observed cancellation.

**Verification:**

- [ ] C++ queue tests, Python callback/iterator tests, and CLI smoke tests pass with the current runtime.
- [ ] Stress tests cover many more realizations than worker threads and repeated cancellation.

**Dependencies:** Task 3

**Files likely touched:** `src/mass_spec.cpp`, `src/include/warnlogcount.h`, `python/apitofsim.cpp`, `src/cli/mass_spec.cpp`, `test/`

**Estimated scope:** Medium

## Task 6: Port mass-spec realization loops

**Description:** Convert both mass-spec realization loops to TBB using range-local reduction state and the delta transport established in Task 5. Add checkpoints inside long trajectories and before queue writes.

**Acceptance criteria:**

- [ ] Both substance variants use TBB and produce the fixed-seed final counters/results captured by Task 1.
- [ ] Final counter reduction is independent of scheduling and uses no stable-worker assumption.
- [ ] Task exceptions and signals cancel pending realizations; active realizations stop at bounded checkpoints without post-cancel messages.

**Verification:**

- [ ] Mass-spec C++/Python tests pass at concurrency 1 and default concurrency.
- [ ] Repeated stress runs pass under ThreadSanitizer where supported.

**Dependencies:** Tasks 3 and 5

**Files likely touched:** `src/mass_spec.cpp`, operation-context headers, `test/main.cpp`, `test/test_python.py`

**Estimated scope:** Medium

## Checkpoint: Runtime cutover

- [ ] Tasks 4-6 pass together with nested and concurrent API calls covered.
- [ ] No TBB algorithm relies on a stable worker-thread identity.
- [ ] Numerical, cancellation, and streaming contracts pass before build cleanup.

## Task 7: Remove remaining OpenMP runtime code

**Description:** Replace runtime queries/debug output, remove or rewrite unused OpenMP synchronization helpers, rename runtime-specific files/symbols, and audit all sources.

**Acceptance criteria:**

- [ ] `debug_info()` reports oneTBB version/concurrency and no longer references `_OPENMP` except as needed for SIMD compilation diagnostics.
- [ ] `OMPExceptionHelper`, `OMP_VISIBILITY_NONE`, `omp_get_*`, `declare reduction`, runtime `atomic`/`critical`, and runtime parallel pragmas are absent.
- [ ] Only intentional `#pragma omp simd` directives and SIMD-related comments remain.

**Verification:**

- [ ] `rg -n 'OMPExceptionHelper|OMP_VISIBILITY_NONE|omp_get_|parallel for|parallel sections|declare reduction|#pragma omp (atomic|critical)' src python` returns no runtime usage.
- [ ] Full C++ and Python tests pass.

**Dependencies:** Tasks 4 and 6

**Files likely touched:** `src/apitofsim.cpp`, `src/cli/common_io.h`, runtime helper files, `src/meson.build`, include call sites

**Estimated scope:** Medium

## Task 8: Remove OpenMP build configuration

**Description:** Make oneTBB the sole parallel runtime in Meson, remove the OpenMP option/duplicate library, and use TBB concurrency control for deterministic single-thread tests.

**Acceptance criteria:**

- [ ] Meson targets link oneTBB and never link an OpenMP runtime; SIMD flags are compiler-checked and compile-only.
- [ ] `openmp`/Intel profile branches and `libapitofsim_no_omp` are removed or replaced according to the approved runtime-free-build decision.
- [ ] Existing test generation and single-thread checks use the normal library under a one-thread TBB limit.

**Verification:**

- [ ] Clean GCC and Clang builds pass with `--warnlevel 3 --werror`.
- [ ] Dynamic dependency inspection shows TBB and no OpenMP runtime.

**Dependencies:** Task 7

**Files likely touched:** `meson.build`, `meson_options.txt`, `src/meson.build`, `src/cli/meson.build`, `python/meson.build`, `test/meson.build`, `meson/dev/*.ini`

**Estimated scope:** Medium; land target cleanup before developer-profile cleanup

## Task 9: Update CI and distribution packaging

**Description:** Replace LLVM OpenMP packages with oneTBB development/runtime dependencies and verify repaired wheels and Conda artifacts in clean environments.

**Acceptance criteria:**

- [ ] Test/QA workflows install oneTBB and run the same compiler, sanitizer, and Valgrind coverage as before.
- [ ] Wheel builds bundle or correctly depend on TBB and pass a clean-environment import plus parallel smoke test.
- [ ] The Conda recipe uses `tbb-devel`/`tbb` as appropriate and no longer includes `llvm-openmp`.

**Verification:**

- [ ] GitHub Actions test, QA, wheel, and Conda jobs pass.
- [ ] Artifact inspection and clean-environment smoke tests are recorded in the PR.

**Dependencies:** Task 8

**Files likely touched:** `.github/workflows/test.yml`, `.github/workflows/qa.yml`, `.github/workflows/wheels.yml`, `ci/setup_linux.sh`, `conda.recipe/recipe.yaml`

**Estimated scope:** Medium

## Task 10: Benchmark, document, and perform the removal audit

**Description:** Compare serial/default-concurrency performance, document runtime/thread-control changes, resolve the `SIGABRT` follow-up decision, and prove that the OpenMP runtime is fully removed.

**Acceptance criteria:**

- [ ] Benchmarks cover density/rate and mass-spec workloads at one and default concurrency with any material regression explained.
- [ ] User/developer documentation covers the TBB dependency, thread limit, SIMD-only OpenMP pragmas, signal semantics, and removed build options.
- [ ] Source, build metadata, lock/package files, binaries, and artifacts contain no unintended OpenMP runtime dependency.

**Verification:**

- [ ] `uv run meson test --print-errorlogs -C build` and relevant benchmarks pass on the final tree.
- [ ] `rg` audit plus platform dynamic-dependency inspection is clean.

**Dependencies:** Task 9

**Files likely touched:** `README.md`/developer docs, benchmark notes, package/build metadata

**Estimated scope:** Small

## Checkpoint: Complete

- [ ] All acceptance criteria above are met.
- [ ] Full CI and artifact tests pass with no unexplained performance regression.
- [ ] OpenMP remains only as SIMD compiler hints with no OpenMP runtime linkage.
- [ ] The migration and any intentional compatibility breaks are ready for review.
