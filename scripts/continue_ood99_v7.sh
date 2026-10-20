#!/usr/bin/env bash
set -euo pipefail
cd '/media/cody/软件/behavior_harness'
# Let the existing task finish; its runner sees STOP_AFTER_CURRENT afterwards.
while [[ -r /proc/1527372/cmdline ]] && tr '\0' ' ' < /proc/1527372/cmdline | rg -q 'scripts/run_ood99.py'; do
  sleep 15
done
if [[ -f outputs/ood99_20261019_v5/STOP_AFTER_CURRENT ]]; then
  mv outputs/ood99_20261019_v5/STOP_AFTER_CURRENT outputs/ood99_20261019_v5/STOP_AFTER_CURRENT_consumed_v7
fi
export AP_HARNESS_PLANNER_MODEL=gpt-6.1-sol
exec /home/cody/miniconda3/envs/behavior-2026/bin/python scripts/run_ood99.py \
  --output outputs/ood99_20261019_v5 --attempt v7 --retry-infrastructure-fresh-episode --wait-for-gpu
