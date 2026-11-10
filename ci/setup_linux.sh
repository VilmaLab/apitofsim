#!/usr/bin/env bash

set -euo pipefail

dnf -y install clang clang-libs curl flex git perl zip unzip
bash "$(dirname "$0")/setup_vcpkg.sh" /opt/apitofsim-vcpkg
PIPX_DEFAULT_PYTHON=/usr/bin/python3 pipx install -f patchelf==0.19.1.0rc1
