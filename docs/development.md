# Building from source & development

This page includes instructions for building the source code from source, and a number of notes for debugging/profiling when developing the package.

## Building the sources for users

This section describes how to build the sources from scratch.

### Using Meson to compile the package

**Note: You need Meson 1.10 or later and oneTBB 2023.0 or later to build the package. You can install Meson with `uv tool install meson`; install the oneTBB development package supplied by your platform.**

You can build with slightly older versions of Meson by disabling typestub generation by passing `-Dstubgen=disabled` to `meson setup` or adding it to your `setup-args` in `uv.toml` (see below).

To compile using your system's default compiler:

```bash
meson setup --buildtype release build
meson compile -C build
```

The binaries are then in `build/src/cli`.
You add this directory to your PATH or symlink to them.
You can also add `build/python` to your `PYTHONPATH` to use the compiled Python extension, or else see the next section for a pip/uv based approach.

### Using python-meson to compile the Python extension

[meson-python](https://mesonbuild.com/meson-python/) is used to build the Python extension. Since this works as a normal Python build backend you can install using one of the following:

```bash
pip install git+https://github.com/VilmaLab/apitofsim
pip install /path/to/apitofsim
```

Or using [uv](https://docs.astral.sh/uv/):

```
uv add git+https://github.com/VilmaLab/apitofsim
uv add /path/to/apitofsim
```

See the next section for `--editable` installs

## Development

This section is currently just a selection of tips for setting up a development environment.

### Clangd based IDE completions

You might need to symlink `compile_commands.json` to the root of the repository for your IDE to pick it up:

```bash
ln -sf build/compile_commands.json compile_commands.json
```

### Meson native files

Meson can be configured through so-called ["native build configuration" native files](https://mesonbuild.com/Native-environments.html).
A number of these are provided, with many based on Clang and tested on Linux, with `meson/test` containing configurations for automated tests and `meson/pkg` having configurations for building packages, both of which are run in continuous integration with GitHub Actions.
Use the platform-default C++ standard library so it matches the installed oneTBB package: normally libstdc++ on Linux and libc++ on macOS.

The directory `meson/dev` contains a number of configurations including `clangrelease.ini` which builds the fastest configuration for running locally, `clangprofile.ini`, which includes minimal debugging symbols and no asserts for profiling, `clangdebug.ini` an optimized debug with asserts build for debugging with `gdb` or `lldb`, `clangsingle.ini` a fast single-threaded build which can be run deterministically, and `clangslow.ini`, which is a single threaded unoptimized build, providing the most deterministic and debug-friendly build.

Parallel work is scheduled by oneTBB. Applications that need a scoped limit should use `oneapi::tbb::global_control` or `oneapi::tbb::task_arena`; the single-threaded native files use the same `global_control` mechanism internally. `OMP_NUM_THREADS` is no longer read, and the removed `openmp` Meson option has no replacement.

The native Python extension retains a oneTBB scheduler handle and finalizes it through `Py_AtExit`, before native library teardown. This also covers direct extension imports. If other scheduler users prevent finalization, it prints a native diagnostic to stderr. Unfinished simulation work still needs to complete before interpreter shutdown; cooperative application cleanup is tracked in [issue #85](https://github.com/VilmaLab/apitofsim/issues/85).

OpenMP is not a runtime dependency. The remaining `#pragma omp simd` directives are compiler vectorization hints only; Meson enables them with the compiler-checked `-fopenmp-simd` flag where supported.

Long-running simulations cooperatively cancel on `SIGINT` and `SIGTERM`, restore the previous signal handler, and then re-raise the signal on the calling thread. `SIGABRT` is intentionally not intercepted and remains immediately fatal.

### Configuring meson-python for development

You need to disable build isolation to install this as a development/editable package which will recompile at import:

Using [uv](https://docs.astral.sh/uv/):

```bash
cd /path/to/apitofsim
uv sync
uv pip install --no-build-isolation -e .
```

During development you can use a native file, and add options to show compiler errors when imports trigger recompilation.
If you are using uv.toml, this can all be specified in `uv.toml`.
For example:

```toml
no-build-isolation-package = ["apitofsim"]

[pip]
python = "3.13"

[config-settings]
editable-verbose = "true"
setup-args = [
  "--native-file=/path/to/apitofsim/meson/dev/clangdebug.ini",
  "-Dbuildtype=debug",
]
build-dir = "uvbuild"
```

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
PYTHONPATH=./python/ uv run --group docs zensical serve
```

To build the static site in `site/`:

```bash
PYTHONPATH=./python/ uv run --group docs zensical build --clean
```

Site settings, navigation, Markdown extensions, and watched source directories are
configured in `zensical.toml`.

### Running tests

Given the `uv.toml` given above, you can run the tests like so:

```
uv run meson test --print-errorlogs -C uvbuild
```

This should run the C++ tests and Python tests as two separate test items.

The C++ suite runs once with a one-thread limit and once at oneTBB's default concurrency. Benchmarks cover both modes:

```bash
uv run meson test --print-errorlogs -C build
uv run meson test --benchmark --num-processes 1 -C build
```

### Native IPC event storage

The small, pinned Apache nanoarrow static dependency encodes IPC in C/C++ and
is built through a Meson wrap/overlay. No Arrow C++ runtime ABI or PyArrow version
pin is introduced. The project now requires a C compiler as well as C++20.

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

A database records to a `.events.arrow` sidecar by default. The IPC format has fixed primitive fields,
a schema version/archive UUID, and per-execution footer metadata. An appender
verifies the previous footer, retains its batch descriptors, then replaces only
the EOS/footer and adds new batches. Existing event bodies are never copied.

Publication uses an exclusive portable sidecar lock and a durable small journal
containing the old ending and committed length. Event data, its footer and parent
directory are synced before committing DuckDB's archive generation/length and
execution metadata. Reopening compares generations and either retains the
committed file or restores the old ending/truncates the uncommitted append.
Standalone files use a committed journal marker; explicit overwrite stages an
atomic replacement. A journal is recovery state, not a second event store.

Conversion uses the same exclusive lock and streams IPC batches in file order.
It validates the staged Parquet payload and publishes it at `.events.parquet` before changing
the format marker, generation and size in a DuckDB transaction. Conversion
clears IPC batch ranges and scopes the Parquet scan by result ID instead.
Checkpointing makes the new state visible to database file snapshots before the
IPC sidecar is removed. A retry uses the database state to resume or clean up.
Readers keep using the committed format throughout recovery.

DuckDB keeps execution/root/tree/counter metadata. It does not insert or allocate
IDs for event/realization rows. PyArrow datasets supply temporary DuckDB event
views for IPC. Parquet views use DuckDB's native scan. SQL IDs are opaque composite
strings; numeric `result_id`, `realization_index`, `event_index` define order.
Deletion is logical until explicit compaction; a schema change requires migration.

Downstream changes required before updating their dependency pins:

- `apitofsim-resultviewer` must use `ExperimentDatabase`, transport/open the IPC
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
