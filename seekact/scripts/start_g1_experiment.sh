#!/usr/bin/env bash
# One entry point; the Python coordinator checks/reuses services and opens the UI.
set -euo pipefail
G1_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$G1_PROJECT_ROOT"
export PYTHONPATH="$G1_PROJECT_ROOT"
exec "${G1_PYTHON:-/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python}" \
  scripts/g1_start_experiment.py "$@"
