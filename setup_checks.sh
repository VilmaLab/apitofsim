#!/usr/bin/bash
# Set up the local check environment: install all dependency groups/extras via
# ./uv_sync.sh. That also builds the native extension into the uv build-dir
# configured in uv.toml, which provides compile_commands.json for checks.sh
# and clangd.
set -euo pipefail
./uv_sync.sh
