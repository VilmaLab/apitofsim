# Building from source & development

This page includes instructions for building the source code from source, and a number of notes for debugging/profiling when developing the package.

## Building the sources for users

This section describes how to build the sources from scratch.

### Using Meson to compile the package

**Note: You need Meson 1.10 or later to build the package. You can install this with `uv tool install meson`.**

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

```
See the next section for `--editable` installs
```

## Development

This section is currently just a selection of tips for setting up a development environment.

### Clangd based IDE completions

You might need to symlink `compile_commands.json` to the root of the repository for your IDE to pick it up:

```bash
ln -sf build/compile_commands.json compile_commands.json
```

### Meson native files

Meson can be configured through so-called ["native build configuration" native files](https://mesonbuild.com/Native-environments.html).
A number of these are provided, with many based on Clang and tested on Linux, with `/meson/test` containing configurations for automated tests and `/meson/pkg` having configurations for building packages, both of which are run in continuous integration with GitHub Actions.
On Linux, you may need to install `libc++` (from LLVM rather than GNU) e.g. `apt install 'libc++1' 'libc++-dev'`.

The directory `/meson/dev` contains a number of configurations including `clangrelease.ini` which builds the fastest configuration for running locally, `clangprofile.ini`, which includes minimal debugging symbols and no asserts for profiling, `clangdebug.ini` an optimized debug with asserts build for debugging with `gdb` or `lldb`, `clangsingle.ini` a fast single-threaded build which can be run deterministically, and `clangslow.ini`, which is a single threaded unoptimized build, providing the most deterministic and debug-friendly build.

### Configuring meson-python for development

You need disable build isolation to install this as a development/editable package which will recompile at import:

Using [uv](https://docs.astral.sh/uv/):

```bash
cd /path/to/apitofsim
uv sync
uv pip install --no-build-isolation -e .
```

During development you can use a native file, and add options to enable get compiler errors on recompile-triggered-by-import.
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

The following will run a live-reloading mkdocs server:

```bash
uv sync --all-groups
PYTHONPATH=./python/ uv run mkdocs serve --livereload --watch=python --watch=docs --watch=src
```

### Running tests

Given the `uv.toml` given above, you can run the tests like so:

```
uv run meson test --print-errorlogs -C uvbuild
```

This should run the C++ tests and Python tests as two separate test items.
