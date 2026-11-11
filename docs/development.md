# Building from source & development

This page includes instructions for building the source code from source, and a number of notes for debugging/profiling when developing the package.

## Building from source

Install CMake 3.28 or later, Ninja, and a C/C++20 toolchain. Bootstrap the vcpkg
revision pinned in `vcpkg.json`, then set `VCPKG_ROOT` to that checkout. The
following commands use Bash, including Git Bash on Windows.

```bash
export VCPKG_ROOT="$HOME/.scratch/apitofsim-vcpkg"
bash ci/setup_vcpkg.sh "$VCPKG_ROOT"
```

Configure and build the native executables:

```bash
cmake -S . -B builds/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build builds/release --parallel
```

The three executables are in `builds/release`. Use `cmake --install` to install
them under the configured prefix. On Windows, keep the vcpkg runtime DLLs on
`PATH` if installing the native executables outside their build directory.

For a Python source build, use Python 3.12 or later and run `uv pip install .`
or `pip install .` with the same `VCPKG_ROOT`. Scikit-build-core invokes CMake
and builds the nanobind extension with the CPython 3.12 stable ABI. The built
wheel includes the native executables and generated Python type stubs.

For an installed dependency environment such as conda-forge, configure with
`-DAPITOFSIM_USE_VCPKG=OFF` in a fresh build directory. CMake discovers Eigen,
oneTBB and compression libraries through the environment's package prefix.
The build fetches pinned carquet and nanoarrow sources. For pip builds, pass `-Ccmake.define.APITOFSIM_USE_VCPKG=OFF`.

## Development

### CMake build presets

The shared [CMake presets](https://cmake.org/cmake/help/v3.28/manual/cmake-presets.7.html)
provide three native Clang builds, each with its own directory under `builds/`.
All three use Ninja and treat compiler warnings as errors.

- `clang-release` optimizes for the host CPU and disables assertions.
- `clang-profile` uses the same optimization with debug symbols for profiling.
- `clang-debug` keeps assertions enabled and builds with `-O2` and debug
  symbols for use with GDB or LLDB.

```bash
cmake --preset clang-profile
cmake --build builds/cmake-clang-profile --parallel
./test.sh --test-dir builds/cmake-clang-profile
```

Use an untracked `CMakeUserPresets.json` to inherit a preset and customize
compiler paths, the linker, or a compiler launcher such as `ccache`. Keep the
platform-default C++ standard library to match oneTBB. For a single-threaded
simulation build, include `-DAPITOFSIM_MAX_PARALLELISM=1` in `CMAKE_CXX_FLAGS`
alongside any existing flags.

### Editable installation

```bash
uv sync
```

Python changes are visible immediately. For C++ rebuilds on import, add the
following to your local `uv.toml`. Replace old Meson-specific config settings
and use a fresh build directory.

```toml
[config-settings]
build-dir = "builds/uvbuild"
"editable.rebuild" = "true"
"cmake.build-type" = "RelWithDebInfo"
```

For editable builds, set compiler choices separately from the native presets.
For example, use `CC=clang CXX=clang++` on the first `uv sync` in a fresh build
directory, and use `cmake.define.*` entries in `uv.toml` for CMake options.

For clangd, symlink `compile_commands.json` from your build directory:

```bash
ln -sf builds/uvbuild/compile_commands.json compile_commands.json
```

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

### Using debug Python builds

You can get Python stack traces in gdb by adding something like the following to your `~/.gdbinit` file:

```
source /usr/share/gdb/auto-load/usr/bin/python3.12-gdb.py
```

You then need to build against the debug Python build like so:

```bash
uv sync -p /usr/bin/python3.12-dbg --reinstall-package apitofsim
```

### Building docs

The mathematical reference is written in `docs/simulation-methods.tex`. Install
`latexmk`, `texlive-latex-extra`, and `texlive-fonts-recommended`, then build its
PDF before building or serving the site:

```bash
bash ci/build-methods.sh
```

The script uses `${TMPDIR:-$HOME/.scratch}/apitofsim-methods` for intermediate
files (or the directory passed as its first argument) and copies the PDF to
`docs/simulation-methods.pdf`. The generated PDF is not committed. CI builds it
as the `simulation-methods-pdf` artifact and includes it in the deployed site.

The following will run a live-reloading Zensical server:

```bash
uv sync --group docs
uv run --group docs zensical serve
```

To build the static site in `site/`:

```bash
uv run --group docs zensical build --clean
```

Site settings, navigation, Markdown extensions, and watched source directories are
configured in `zensical.toml`.

### Running tests

To build and test the C++ and Python targets together, use the project's Python
environment so CTest can find pytest and the runtime dependencies:

```bash
uv sync --no-install-project
uv run --no-sync cmake -S . -B builds/test -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAPITOFSIM_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build builds/test --parallel
./test.sh --test-dir builds/test
```

The C++ suite runs with one thread and again at oneTBB's default concurrency.
CTest supplies the native signal helper to the Python suite. The native
benchmark executables are under `builds/test/test`. Set
`APITOFSIM_BENCH_CONCURRENCY=1` to run them with a one-thread limit.

### Native event storage

Native event recording uses carquet for Parquet and nanoarrow 0.9.0 for Arrow
IPC. CMake fetches pinned sources; an existing checkout can be supplied with
`FETCHCONTENT_SOURCE_DIR_CARQUET` or `FETCHCONTENT_SOURCE_DIR_NANOARROW`.
The native build requires Zstandard, zlib, LZ4 and OpenMP support for C and C++.
With Clang on Linux, install the matching `libomp` development package; on macOS,
install Homebrew `libomp` and set `OpenMP_ROOT` to its prefix. Release wheels
build a pinned OpenMP runtime with deployment target 12.0 using
`ci/setup_macos.sh`. vcpkg supplies the
compression libraries. Nanoarrow uses the private `apitofsim_` symbol prefix.
Memory-check CI jobs set `KMP_AFFINITY=disabled` to avoid a libomp x2APIC
topology-probe allocation leak reproduced independently of carquet. Native leak
detection remains enabled.

PyArrow remains a Python runtime dependency for event reading and processing.
The native extension uses the Arrow C Data interface for Python delivery;
carquet and nanoarrow are linked statically into release wheels. The wheel
smoke tests assert that no Arrow C++/Parquet libraries are bundled, load the
native extension without Python dependencies, check both import orders with
PyArrow, and exercise Zstandard and parallel column encoding. The macOS arm64
wheel requires macOS 12.

The Parquet recorder writes `EventColumns` vectors directly through carquet's
C API, using the existing bounded batch queue and recording transactions. It
caps row groups at 65,536 rows and enables dictionary encoding, statistics,
archive metadata and Zstandard level 3 where requested. `parquet_column_workers`
controls carquet encoding concurrency through the OpenMP setting on the
dedicated writer thread; it does not change TBB's simulation concurrency.
Carquet also uses its default asynchronous I/O thread. Nanoarrow retains the
in-place IPC append path and execution metadata in the footer.

`test/bench/parquet_matrix.py` compares separately configured Release builds
using a cached private model database. For example:

```bash
python test/bench/parquet_matrix.py /path/to/model.duckdb "$HOME/.scratch/parquet-bench" \
  --python /path/to/environment/bin/python \
  --arrow-build "$HOME/.scratch/arrow-build" \
  --carquet-build "$HOME/.scratch/carquet-build" \
  --workers 1 2 4 8 --realizations 5000 --repeats 1
```

The Arrow comparison build must use an older revision with Arrow C++ recording
(the recorded baseline below is `287d81c`). Both builds need
`APITOFSIM_BUILD_PYTHON=ON` and `BUILD_TESTING=ON`, which mirrors
the package into `python/Release` next to its extension. The benchmark selects
cluster-tree roots 1 and 35 with seed 42. Each case runs in a fresh process;
preparation is excluded, and execution through durable database close is
included. Peak RSS includes preparation but excludes the subsequent full-file
verification. Every event column is hashed in physical order, with realization
ordering, continuous event indices, row-group limits and compression checked.
The matrix compares those hashes and simulation counters across all cases,
retains JSON reports and small logs, and removes its generated databases and
sidecars after successful verification.


#### Original carquet prototype benchmark (2026-11-11)

This repeats the recording workload from [PR #173](https://github.com/VilmaLab/apitofsim/pull/173):
a private Besel2020 model, cluster-tree roots 1 and 35, 5,000 realizations each,
seed 42, and 45,762,568 recorded events. Each cell is one end-to-end measurement,
including simulation, encoding, event sync, database commit/close and database
sync. Preparation and verification are excluded. Peak RSS includes preparation
and is sampled before verification. The OS cache was left in place, with normal
desktop background load. Treat small differences between worker counts as noise.

Hardware: Intel Core Ultra 5 225U, 14 logical CPUs, NVMe/Btrfs, Linux x86-64.
Both simulation builds use Clang 22.1.8, Release (`-O3 -DNDEBUG -march=native`),
and LLD. Arrow C++/Parquet 26.0.1 comes from conda-forge; carquet 0.8.0 is built
from the pinned source with its upstream Release defaults. Both writers link the
same conda Zstandard library. Python is 3.12.16, PyArrow 26.0.1, DuckDB 1.5.6,
and oneTBB 2023.1. These fresh measurements use a different library build and
background load from the original PR, so compare backends within this table.

Both writers request dictionary encoding, statistics, a 65,536-row group limit
and Zstandard level 3 where enabled. The worker count controls encoding:
Arrow uses its private column pool, while carquet uses the writer thread's
OpenMP concurrency setting for its internal encoding workers. Carquet also has
its default asynchronous I/O thread. Simulation retains default TBB concurrency
in every case; the table does not count total application threads.

| Recording | Encoding workers | Wall time | Sidecar | Peak process RSS |
|---|---:|---:|---:|---:|
| Summary only | — | 32.4 s | — | 1.60 GiB |
| Arrow IPC | — | 36.0 s | 7.50 GiB | 1.69 GiB |
| Arrow C++ Parquet, none | 1 | 90.8 s | 4.01 GiB | 2.06 GiB |
| carquet Parquet, none | 1 | 42.5 s | 4.05 GiB | 1.97 GiB |
| Arrow C++ Parquet, none | 2 | 66.2 s | 4.01 GiB | 2.10 GiB |
| carquet Parquet, none | 2 | 40.5 s | 4.05 GiB | 2.10 GiB |
| Arrow C++ Parquet, none | 4 | 69.6 s | 4.01 GiB | 2.14 GiB |
| carquet Parquet, none | 4 | 39.2 s | 4.05 GiB | 2.09 GiB |
| Arrow C++ Parquet, none | 8 | 55.2 s | 4.01 GiB | 2.19 GiB |
| carquet Parquet, none | 8 | 43.2 s | 4.05 GiB | 2.06 GiB |
| Arrow C++ Parquet, Zstandard | 1 | 103.2 s | 3.74 GiB | 2.05 GiB |
| carquet Parquet, Zstandard | 1 | 61.4 s | 3.77 GiB | 1.98 GiB |
| Arrow C++ Parquet, Zstandard | 2 | 80.5 s | 3.74 GiB | 2.10 GiB |
| carquet Parquet, Zstandard | 2 | 42.0 s | 3.77 GiB | 2.16 GiB |
| Arrow C++ Parquet, Zstandard | 4 | 64.8 s | 3.74 GiB | 2.12 GiB |
| carquet Parquet, Zstandard | 4 | 46.7 s | 3.77 GiB | 2.18 GiB |
| Arrow C++ Parquet, Zstandard | 8 | 60.3 s | 3.74 GiB | 2.21 GiB |
| carquet Parquet, Zstandard | 8 | 40.4 s | 3.77 GiB | 2.12 GiB |

Comparing the fastest observed configurations, carquet is 1.41× faster
uncompressed (39.2 versus 55.2 seconds) and 1.49× faster with Zstandard (40.4
versus 60.3 seconds). Its files are about 1% larger. With Zstandard, two carquet
encoding workers already reach 42.0 seconds; higher counts provide no consistent
improvement in this single-run sweep. These figures measure end-to-end recording.

Every recorded variant has identical simulation counters and identical SHA-256
hashes for all 22 event columns in physical order. Verification also checks
result/realization ordering, continuous event indices, row-group limits and
compression. One interrupted execution-environment session was discarded and
its incomplete case rerun successfully. The small raw report, source hashes,
environment and QA details are in `test/bench/results/carquet-20261111.json`.
Generated databases and sidecars were removed after successful verification.

The benchmark used the initial carquet prototype before the full migration.
The current implementation uses carquet for all Parquet recording and the
restored nanoarrow implementation for IPC. The recording API, schema,
transactions and bounded queue are preserved; downstream readers need no
changes. The prototype figures above retain their original build provenance.

Validation includes both native concurrency configurations for both backends,
the full carquet-enabled Python suite (563 passed, one expected skip), strict
warnings builds, formatting, lint/types, and ASan/UBSan writer tests. Leak
checking was disabled because the sandbox uses ptrace. A full clang-tidy run
found one signed-index warning, fixed and rechecked in all changed writer files;
there were no other project diagnostics. The raw report describes validation of that prototype; the migration PR
also runs the platform wheel and documentation workflows.


#### Migration recording validation (2026-11-11)

The migration was checked separately on the same full Besel2020 workload after
restoring nanoarrow IPC and making carquet unconditional. Background load
differed from the earlier sweep, so these rows validate the final recording
paths rather than provide a new cross-backend speedup comparison. The core
uses the same Release flags; carquet's upstream implicit LTO is disabled so
ordinary Clang builds can use their configured linker.

| Recording | Encoding workers | Wall time | Sidecar | Peak process RSS |
|---|---:|---:|---:|---:|
| nanoarrow IPC | — | 33.1 s | 7.50 GiB | 1.70 GiB |
| carquet Parquet, none | 4 | 31.6 s | 4.05 GiB | 2.07 GiB |
| carquet Parquet, Zstandard | 4 | 32.5 s | 3.77 GiB | 2.08 GiB |

All 45,762,568 events match the original Arrow C++/IPC baseline in every one
of the 22 physical-order column hashes and all simulation counters. Full-file
checks again verify row-group limits, codecs, ordering and ordinal continuity.
The raw records are in `test/bench/results/carquet-nanoarrow-20261111.json`.

`EventColumns` is the common native column layout. A oneTBB pipeline assigns
realization IDs, simulates histories in parallel, and delivers completed tokens
through a serial output stage. Recording uses in-order output and the stage
coalesces events directly into IPC batches. Final-only summaries can group
realizations and reduce counters without collecting events. Python iterators
use a bounded handoff for events and callbacks. The raw iterator retains its eager queue
for join-before-read callers. Retained record batches own their buffers. The IPC
writer borrows the event vectors for synchronous encoding.

Managed streams acquire an exclusive IPC writer lease before starting the
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

### Bumping the downstream pins

`apitofsim-web` and `apitofsim-resultviewer` pin the wheels and the conda package of one release
by URL. `bump_downstream.py` reads the assets and their sha256 digests from the release page and
rewrites the pins in one file, `pyproject.toml` in the current directory by default:

```bash
cd ../apitofsim-web
../apitofsim/bump_downstream.py v0.0.18          # pyproject.toml, plus uv.lock
../apitofsim/bump_downstream.py v0.0.18 env.yaml # conda environment
```

A `.toml` file counts as a uv project, so the `uv.lock` next to it is rewritten as well and
`uv lock --check` verifies both. A `.yaml` file is a conda environment. Add `--dry-run` to see
the changes without writing them.
