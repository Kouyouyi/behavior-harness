#!/bin/bash
set -euo pipefail
until test -f /cs/kyy/models/behavior-2025-champion/DOWNLOAD_VERIFIED.json; do sleep 20; done
export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME=/cs/kyy/cache/task_openpi
export HF_HOME=/cs/kyy/cache/champion_hf
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/cs/kyy/behavior_ap_experiment/vendor/behavior_champion/src:/cs/kyy/behavior_ap_experiment/vendor/behavior_champion/openpi/src:/cs/kyy/behavior_ap_experiment/vendor/behavior_champion/openpi/packages/openpi-client/src
exec /cs/kyy/behavior_ap_experiment/vendor/behavior_champion/.venv/bin/python -u /cs/kyy/behavior_ap_experiment/scripts/champion_server.py
