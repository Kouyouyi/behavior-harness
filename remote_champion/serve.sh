#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CHAMPION_PYTHON:?Set the dedicated Python3.11 JAX/OpenPI executable}"
: "${CHAMPION_WEIGHTS:?Set the downloaded checkpoint_2 parent directory}"
export PYTHONPATH="$repo/third_party/champion/src:$repo/third_party/champion/openpi/src:$repo/third_party/champion/openpi/packages/openpi-client/src:$repo/seekact${PYTHONPATH:+:$PYTHONPATH}"
exec "$CHAMPION_PYTHON" -m services.behavior_ap.champion_server --weights "$CHAMPION_WEIGHTS" --port "${CHAMPION_PORT:-18778}" --output "${CHAMPION_REPORT_DIR:-$repo/outputs/champion_service}" "$@"
