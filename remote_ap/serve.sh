#!/usr/bin/env bash
set -euo pipefail
AP_CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AP_PYTHON:?Set AP_PYTHON to the remote Python executable}"
: "${AP_CHECKPOINT:?Set AP_CHECKPOINT to the downloaded checkpoint directory}"
export PYTHONPATH="$AP_CODE_ROOT/vendor:$AP_CODE_ROOT/serving_runtime${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
exec "$AP_PYTHON" "$AP_CODE_ROOT/serving_runtime/vla_http_server.py" --checkpoint "$AP_CHECKPOINT" --port "${AP_PORT:-18770}"
