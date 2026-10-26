#!/usr/bin/env bash
# Local presentation console, defaults to demo. Never starts robot control.
set -euo pipefail
G1_CONSOLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$G1_CONSOLE_ROOT"
export PYTHONPATH="$G1_CONSOLE_ROOT"
exec "${RAX_PYTHON:-python3}" -m services.console.server "$@"
