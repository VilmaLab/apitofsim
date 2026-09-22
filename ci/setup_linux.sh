#!/usr/bin/env bash

set -euo pipefail

dnf -y install clang clang-libs curl
./ci/install_tbb.sh
PIPX_DEFAULT_PYTHON=/usr/bin/python3 pipx install -f patchelf==0.19.1.0rc1
