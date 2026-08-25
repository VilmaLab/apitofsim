#!/usr/bin/env bash

set -euo pipefail

tbb_commit=3046c8b0c29df995980003ea24f4d78c80ec0c8d
tbb_sha256=44763279bb3ac76fb67572cc3336d7421521bc882e9fde484ea662389f4fda0b
tbb_root=/tmp/apitofsim-oneapi-tbb

if [[ ! -f "${tbb_root}/include/oneapi/tbb/version.h" ]]; then
  tbb_tmp=$(mktemp -d)
  trap 'rm -rf "${tbb_tmp}"' EXIT
  tbb_archive="${tbb_tmp}/oneTBB-${tbb_commit}.tar.gz"

  curl -L --fail --silent --show-error \
    "https://github.com/uxlfoundation/oneTBB/archive/${tbb_commit}.tar.gz" \
    -o "${tbb_archive}"
  echo "${tbb_sha256}  ${tbb_archive}" | shasum -a 256 --check
  tar -xzf "${tbb_archive}" -C "${tbb_tmp}"

  cmake \
    -S "${tbb_tmp}/oneTBB-${tbb_commit}" \
    -B "${tbb_tmp}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${tbb_root}" \
    -DTBB_STRICT=OFF \
    -DTBB_TEST=OFF
  cmake --build "${tbb_tmp}/build" --parallel
  cmake --install "${tbb_tmp}/build"
fi
