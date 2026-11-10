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
oneTBB and the optional Arrow libraries through the environment's package
prefix. For pip builds, pass `-Ccmake.define.APITOFSIM_USE_VCPKG=OFF`.

## Development

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

Use `CC=clang CXX=clang++` when first configuring to select Clang, and keep the
platform-default C++ standard library to match oneTBB. For native builds,
`Debug` enables assertions with minimal optimization and `RelWithDebInfo`
keeps symbols for profiling. To reproduce the old single-thread build, add
`-DCMAKE_CXX_FLAGS=-DAPITOFSIM_MAX_PARALLELISM=1`. An optional
`-DAPITOFSIM_STRICT_WARNINGS=ON` matches CI's warnings-as-errors checks.

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

The standard build compiles a pinned, namespaced copy of Apache nanoarrow with
IPC support. To build the Apache C++ Parquet writer, add
`-DAPITOFSIM_NATIVE_PARQUET=ON` to CMake, or use
`-Ccmake.define.APITOFSIM_NATIVE_PARQUET=ON` for a pip build. vcpkg then enables
the `parquet` manifest feature and supplies Arrow C++ with Zstandard support.
Use PyArrow matching the native Arrow version when loading both in one process.
Conda packages enable the writer using matching conda-forge dependencies.
GitHub Release wheels use `APITOFSIM_PYARROW_PARQUET=ON` to link the Python
extension directly against the Arrow and Parquet libraries in the pinned PyArrow
wheel. The Python CLI can use the native writer. The standalone C++ executables
shipped with the wheel use the IPC build and can start independently of PyArrow.
The macOS arm64 wheel requires macOS 12, matching PyArrow.

When updating the PyArrow pin, update the build and runtime requirements and the
macOS wheel repair environment together. The release wheel tests install the
repaired wheel, check both import orders, and exercise Zstandard and parallel
column encoding.

The Parquet recorder imports the ordered nanoarrow stream, caps row groups at
65,536 rows, and uses level 3 for Zstandard. A worker count above one uses a
private Arrow column encoder pool.

`EventColumns` is the common native column layout. A oneTBB pipeline assigns
realization IDs, simulates histories in parallel, and delivers completed tokens
through a serial output stage. Recording uses in-order output and the stage
coalesces events directly into IPC batches. Final-only summaries can group
realizations and reduce counters without collecting events. Python iterators
use a bounded handoff for events and callbacks. The raw iterator retains its eager queue
for join-before-read callers. Record batches own retained buffers, and the
writer borrows C Data arrays for synchronous encoding.

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
