#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CHAMPION_PYTHON:?Set the dedicated Python3.11 JAX/OpenPI executable}"
export PYTHONPATH="$repo/third_party/champion/src:$repo/third_party/champion/openpi/src:$repo/third_party/champion/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo/third_party/champion"
exec "$CHAMPION_PYTHON" scripts/train.py "$@"
