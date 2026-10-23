#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/cs/kyy/behavior_ap_experiment/vendor
ROOT=/cs/kyy/behavior_ap_experiment
PY="$ROOT/.venv/bin/python"
"$PY" - <<'PY'
import json
from pathlib import Path
r=Path('/cs/kyy/behavior_ap_experiment')
assert json.loads((r/'runs/smoke_fp32_v1/status.json').read_text())['status']=='complete'
assert (r/'runs/smoke_fp32_v1/checkpoint/READY.json').exists()
assert json.loads((r/'reports/full_prompt_contract_check.json').read_text())['passed']
assert json.loads((r/'reports/training_contract_check.json').read_text())['passed']
assert json.loads((r/'runs/smoke_full_v1/open_loop_validation.json').read_text())['samples']==4
PY
exec "$PY" "$ROOT/scripts/train_pi05.py" --steps 30000 --batch-size 4 --accumulation 8 --workers 4 --dtype float32 --memory-fraction 0.82 --run-name pi05_goal_tempered_v1
