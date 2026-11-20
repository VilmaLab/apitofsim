#!/usr/bin/bash
# Run clang-tidy on the given source files using a build directory's
# compile_commands.json, e.g. the uv build-dir from uv.toml after
# ./setup_checks.sh. Usage: ./checks.sh <build-dir> <file>...
set -euo pipefail
build_dir="${1:?usage: ./checks.sh <build-dir> <file>...}"
shift
"${CLANG_TIDY:-clang-tidy}" -p "$build_dir" "$@"
