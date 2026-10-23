#!/usr/bin/env bash
set -euo pipefail
AP_CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AP_PYTHON:?Set AP_PYTHON to the training Python executable}"
: "${AP_PROJECT_ROOT:?Set AP_PROJECT_ROOT to output project root}"
: "${AP_RAW_DATA_ROOT:?Set AP_RAW_DATA_ROOT to raw converted dataset root}"
: "${AP_TRAINING_INDEX:?Set AP_TRAINING_INDEX to training_horizon_v2}"
: "${AP_BASE_MODEL_PATH:?Set AP_BASE_MODEL_PATH to pi05_base}"
export AP_TOKENIZER_PATH=${AP_TOKENIZER_PATH:-$AP_CODE_ROOT/tokenizer}
export PYTHONPATH="$AP_CODE_ROOT/vendor:$AP_CODE_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$AP_PYTHON" "$AP_CODE_ROOT/scripts/train_pi05.py" --data-root "$AP_TRAINING_INDEX" "$@"
