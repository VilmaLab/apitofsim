# oneTBB migration task list

## Task 1: Characterize existing parallel behavior

**Description:** Add focused regression coverage before changing runtimes. Exercise currently untested parallel builds, numerical parity, progress/counter streaming, task exceptions, and signal forwarding in subprocesses.

**Acceptance criteria:**

- [ ] Fixed-seed and serial/parallel numerical expectations are recorded with explicit tolerances.
- [ ] Existing `SIGINT`/`SIGTERM` forwarding and `SIGABRT` behavior are safely characterized in subprocesses.
- [ ] Progress and partial-counter tests capture monotonicity, final totals, and cancellation side-effect boundaries.

**Verification:**

- [ ] `uv run meson test --print-errorlogs -C build`
- [ ] The signal tests assert child exit/exception behavior and handler restoration.

**Dependencies:** None

**Files likely touched:** `test/main.cpp`, `test/test_python.py`, `test/meson.build`, new focused test helpers under `test/`

**Estimated scope:** Medium

## Task 2: Establish oneTBB and remove the duplicate serial library

**Description:** Require oneTBB 2023.0 alongside the existing runtime temporarily, prove compiler/package discovery, add compiler-checked `-fopenmp-simd`, and immediately remove `libapitofsim_no_omp` in favor of the normal library under scoped oneTBB concurrency control.

**Acceptance criteria:**

- [ ] Meson requires oneTBB `>=2023.0` through tested pkg-config/CMake paths and fails clearly for an older version.
- [ ] GCC and Clang compile/link a TBB smoke target; a SIMD-only target has no OpenMP runtime linkage.
- [ ] `libapitofsim_no_omp` is removed; generators/tests link the normal library and obtain deterministic one-thread execution through oneTBB controls.

**Verification:**

- [ ] `meson setup --wipe build && meson compile -C build`
- [ ] Inspect a SIMD-only test binary with the platform's dynamic dependency tool.

**Dependencies:** Task 1

**Files likely touched:** `meson.build`, `src/meson.build`, `test/meson.build`, optional `subprojects/` wrap metadata

**Estimated scope:** Medium

## Task 3: Introduce operation-scoped cancellation and exception transport

**Description:** Delete the OpenMP-specific helper interface and replace its mixed responsibilities with generic operation cancellation/signal handling plus a separate cross-`std::thread` exception transport. Keep only the minimum generic exception guard needed by OpenMP loops awaiting their own cutover.

**Acceptance criteria:**

- [ ] A top-level RAII scope handles only `SIGINT`/`SIGTERM` and restores prior handlers on success, exception, and cancellation; nested use does not overwrite saved handlers.
- [ ] A shared `task_group_context` can be cancelled at a safe checkpoint, with first-signal precedence preserved.
- [ ] `SIGABRT` is never installed or converted, while background mass-spec exceptions and cooperative signals still arrive at the invoking CLI/Python thread.

**Verification:**

- [ ] Focused C++ cancellation/exception tests pass under normal and sanitizer builds.
- [ ] Python subprocess tests deliver `SIGINT`/`SIGTERM` during active work within the stated bound and verify immediate fatal `SIGABRT` behavior.

**Dependencies:** Task 2

**Files likely touched:** replacements for `src/include/openmp_helper.h` and `src/openmp_helper.cpp`, `python/apitofsim.cpp`, `src/cli/mass_spec.cpp`, `test/`

**Estimated scope:** Medium

## Checkpoint: Foundation

- [ ] Tasks 1-3 pass with the old runtime paths still available.
- [ ] The cancellation contract and deliberate removal of `OMP_NUM_THREADS` behavior have review approval.

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

**Description:** Make oneTBB the sole parallel runtime in Meson and remove the OpenMP option and target dependencies. Deterministic tests already use the normal library under oneTBB concurrency control from Task 2.

**Acceptance criteria:**

- [ ] Meson targets link oneTBB and never link an OpenMP runtime; SIMD flags are compiler-checked and compile-only.
- [ ] `openmp` and Intel-specific OpenMP profile branches are removed; no alternate serial backend or compatibility option remains.
- [ ] Existing test generation and single-thread checks use the normal library under a one-thread TBB limit.

**Verification:**

- [ ] Clean GCC and Clang builds pass with `--warnlevel 3 --werror`.
- [ ] Dynamic dependency inspection shows TBB and no OpenMP runtime.

**Dependencies:** Task 7

**Files likely touched:** `meson.build`, `meson_options.txt`, `src/meson.build`, `src/cli/meson.build`, `python/meson.build`

**Estimated scope:** Medium

## Task 9: Remove stale developer and CI OpenMP setup

**Description:** Update developer profiles and core test/QA workflows immediately after the Meson cutover so they install and exercise oneTBB 2023.0 without OpenMP-specific modes or packages.

**Acceptance criteria:**

- [ ] Single-thread/slow/profile native files use oneTBB controls and contain no OpenMP project option.
- [ ] Test and QA workflows install oneTBB `>=2023.0`, omit LLVM OpenMP packages, and retain compiler/sanitizer/Valgrind coverage.
- [ ] Clean GCC and Clang CI builds exercise default and constrained concurrency.

**Verification:**

- [ ] GitHub Actions test and QA jobs pass.
- [ ] `rg -n 'openmp|libomp|llvm-openmp' meson/dev .github/workflows/test.yml .github/workflows/qa.yml` finds no runtime setup.

**Dependencies:** Task 8

**Files likely touched:** `meson/dev/icxprofile.ini`, `meson/dev/clangsingle.ini`, `meson/dev/clangslow.ini`, `.github/workflows/test.yml`, `.github/workflows/qa.yml`

**Estimated scope:** Medium

## Task 10: Update wheel and Conda packaging

**Description:** Replace OpenMP distribution dependencies with oneTBB 2023.0 and verify repaired wheels and Conda artifacts in clean environments.

**Acceptance criteria:**

- [ ] Wheel builds bundle or correctly depend on a compatible TBB runtime and pass a clean-environment import plus parallel smoke test.
- [ ] The Conda recipe uses `tbb-devel`/`tbb` with a `>=2023.0` constraint as appropriate and contains no `llvm-openmp`.
- [ ] Artifact dependency inspection finds TBB and no OpenMP runtime.

**Verification:**

- [ ] GitHub Actions wheel and Conda jobs pass.
- [ ] Artifact inspection and clean-environment smoke tests are recorded in the PR.

**Dependencies:** Task 9

**Files likely touched:** `.github/workflows/wheels.yml`, `.github/workflows/conda.yml`, `ci/setup_linux.sh`, `conda.recipe/recipe.yaml`, `pyproject.toml`

**Estimated scope:** Medium

## Task 11: Benchmark, document, and perform the removal audit

**Description:** Compare one-thread/default-concurrency performance, document runtime/thread-control changes and immediate `SIGABRT` handling, and prove that the OpenMP runtime is fully removed.

**Acceptance criteria:**

- [ ] Benchmarks cover density/rate and mass-spec workloads at one and default concurrency with any material regression explained.
- [ ] User/developer documentation covers oneTBB `>=2023.0`, standard oneTBB concurrency controls, removal of `OMP_NUM_THREADS` behavior, SIMD-only OpenMP pragmas, `SIGINT`/`SIGTERM` cancellation, immediate `SIGABRT`, and removed build options.
- [ ] Source, build metadata, lock/package files, binaries, and artifacts contain no unintended OpenMP runtime dependency.

**Verification:**

- [ ] `uv run meson test --print-errorlogs -C build` and relevant benchmarks pass on the final tree.
- [ ] `rg` audit plus platform dynamic-dependency inspection is clean.

**Dependencies:** Task 10

**Files likely touched:** `README.md`/developer docs, benchmark notes, package/build metadata

**Estimated scope:** Small

## Checkpoint: Complete

- [ ] All acceptance criteria above are met.
- [ ] Full CI and artifact tests pass with no unexplained performance regression.
- [ ] OpenMP remains only as SIMD compiler hints with no OpenMP runtime linkage.
- [ ] The migration and any intentional compatibility breaks are ready for review.
