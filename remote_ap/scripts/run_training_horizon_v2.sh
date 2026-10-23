#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTHONPATH=/cs/kyy/behavior_ap_experiment/vendor:/cs/kyy/behavior_ap_experiment/scripts
ROOT=/cs/kyy/behavior_ap_experiment
PY="$ROOT/.venv/bin/python"
exec 9>"$ROOT/runs/pi05_goal_horizon_v2.lock"
flock -n 9
"$PY" - <<'PY'
import json
from pathlib import Path
from training_version import dataset_contract
from training_data import STATE_IDX,IMAGE_KEYS
root=Path("/cs/kyy/behavior_ap_experiment/training_horizon_v2")
report=json.loads((root/"validation_report.json").read_text())
assert report["passed"] and report["classes"]==31
assert report["dataset_contract"]==dataset_contract(root,STATE_IDX,IMAGE_KEYS)
PY
exec "$PY" "$ROOT/scripts/train_pi05.py" --data-root "$ROOT/training_horizon_v2" --steps 30000 --batch-size 4 --accumulation 8 --workers 4 --dtype float32 --memory-fraction 0.82 --run-name pi05_goal_horizon_v2
