#!/usr/bin/env bash

set -euo pipefail

vcpkg_dir=$1
baseline=$(sed -n 's/.*"builtin-baseline": "\([0-9a-f]*\)".*/\1/p' "$(dirname "$0")/../vcpkg.json")
test "${#baseline}" -eq 40
if [[ ! -e "$vcpkg_dir/.git" ]]; then
  git init "$vcpkg_dir"
fi
# Versioned ports can refer to trees from earlier commits in the registry.
if [[ $(git -C "$vcpkg_dir" rev-parse --is-shallow-repository) == true ]]; then
  git -C "$vcpkg_dir" fetch --unshallow --no-tags https://github.com/microsoft/vcpkg.git "$baseline"
elif [[ $(git -C "$vcpkg_dir" rev-parse --verify HEAD 2>/dev/null || true) != "$baseline" ]]; then
  git -C "$vcpkg_dir" fetch --no-tags https://github.com/microsoft/vcpkg.git "$baseline"
fi
git -C "$vcpkg_dir" checkout --detach "$baseline"
case "$(uname -s)" in
  MSYS*|MINGW*) (cd "$vcpkg_dir" && cmd.exe //c bootstrap-vcpkg.bat -disableMetrics) ;;
  *) "$vcpkg_dir/bootstrap-vcpkg.sh" -disableMetrics ;;
esac
