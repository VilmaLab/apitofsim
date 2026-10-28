#!/usr/bin/bash

# signal-helper is deliberately killed with SIGINT/SIGTERM/SIGABRT to exercise
# signal-as-exception handling, so with the default unlimited core limit every
# SIGABRT run leaves a core dump behind. Disable cores for the whole test tree.
ulimit -c 0
exec uv run meson test --print-errorlogs "$@"
