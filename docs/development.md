# Development environment

Build, test, and debug apitofsim from source. See [Internals](internals.md) for
execution and event-storage design, and [Benchmarks](benchmark.md) for recording
measurements and reproduction instructions.

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

### Editable installation with Intel oneAPI on Linux

Load the Intel oneAPI compiler environment so `icx` and `icpx` are on
`PATH` and CMake can find Intel's OpenMP runtime. Set `VCPKG_ROOT` as above
and use a separate build directory for the Intel toolchain.

```toml
no-build-isolation-package = ["apitofsim"]

[config-settings]
build-dir = "builds/uv-intel"
"cmake.build-type" = "Release"
"cmake.define.CMAKE_C_COMPILER" = "icx"
"cmake.define.CMAKE_CXX_COMPILER" = "icpx"
"cmake.define.CMAKE_CXX_FLAGS_RELEASE" = "-O3 -DNDEBUG -march=native"
"cmake.define.CMAKE_MODULE_LINKER_FLAGS" = "-static-intel"
```

Run `./uv_sync.sh` from the project root. The `-march=native` setting tunes
the C++ code for the build host. Omit the
`cmake.define.CMAKE_CXX_FLAGS_RELEASE` entry if the build will run on older
CPUs.

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
CTest supplies the native signal helper to the Python suite.

### Native recording dependencies

Native event recording uses carquet for Parquet and nanoarrow 0.9.0 for Arrow
IPC. CMake fetches pinned sources; an existing checkout can be supplied with
`FETCHCONTENT_SOURCE_DIR_CARQUET` or `FETCHCONTENT_SOURCE_DIR_NANOARROW`.
The native build requires Zstandard, zlib, LZ4 and OpenMP support for C and C++.
With Clang on Linux, install the matching `libomp` development package; on macOS,
install Homebrew `libomp` and set `OpenMP_ROOT` to its prefix. Release wheels
build a pinned OpenMP runtime with deployment target 12.0 using
`ci/setup_openmp.sh`. vcpkg supplies the
compression libraries. Nanoarrow uses the private `apitofsim_` symbol prefix.
Memory-check CI jobs set `KMP_AFFINITY=none` and `KMP_TOPOLOGY_METHOD=flat` to
avoid a libomp x2APIC topology-probe allocation leak reproduced independently
of carquet. Native leak detection remains enabled.

PyArrow remains a Python runtime dependency for event reading and processing.
The native extension exports read-only NumPy views that Python wraps in Arrow tables;
carquet and nanoarrow are linked statically into release wheels. The wheel
smoke tests assert that no Arrow C++/Parquet libraries are bundled, load the
native extension without Python dependencies, check both import orders with
PyArrow, and exercise Zstandard and parallel column encoding. The macOS arm64
wheel requires macOS 12.

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
