#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAX_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAX_RUN_PYTHON="${RAX_PYTHON:-/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python}"
export PYTHONPATH="$RAX_PROJECT_ROOT"
exec "$RAX_RUN_PYTHON" -m harness.codex_cli.launcher "$@"
