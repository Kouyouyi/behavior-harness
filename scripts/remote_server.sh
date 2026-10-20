#!/usr/bin/env bash
set -euo pipefail
# Fixed, verified checkpoint whitelist; use a dedicated terminal.
STEP=${1:-010000}
case "$STEP" in
  010000) PORT=18769 ;;
  024000) PORT=18770 ;;
  *) echo 'Expected checkpoint 010000 or 024000' >&2; exit 2 ;;
esac
BUNDLE=/cs/kyy/behavior_ap_experiment/handoff/pi05_harness_20261017
ssh -t huoshan "cd $BUNDLE && exec env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=$BUNDLE/vendor:$BUNDLE/scripts /cs/kyy/behavior_ap_experiment/.venv/bin/python -u scripts/vla_http_server.py --checkpoint /cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_$STEP --port $PORT"
