#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
build_dir="${1:-${TMPDIR:-$HOME/.scratch}/apitofsim-methods}"
mkdir -p "$build_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$build_dir" docs/simulation-methods.tex
if grep -E 'Overfull \\[hv]box|LaTeX Warning: (Reference|Citation).*undefined|There were undefined references' \
  "$build_dir/simulation-methods.log"; then
  exit 1
fi
cp "$build_dir/simulation-methods.pdf" docs/simulation-methods.pdf
