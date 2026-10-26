#!/usr/bin/env bash
# Trusted operator dashboard (loopback only):
#   bash scripts/run_dashboard.sh [--host 127.0.0.1] [--port 8770]
# Open the printed tokenized URL. Remote: ssh -L 8770:127.0.0.1:8770 <workstation>.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${RAX_PYTHON:-$(command -v python3 || true)}"
[[ -x "$PYTHON" ]] || { echo "Missing Python: '$PYTHON' (set RAX_PYTHON to the robot service environment's python)" >&2; exit 2; }
export PYTHONPATH="$PROJECT_ROOT"
# The agent's own scripts run in RAX_AGENT_PYTHON, an environment without robot SDKs (a session's
# agent_python overrides it).
cd "$PROJECT_ROOT"
exec "$PYTHON" -m services.dashboard.server "$@"
