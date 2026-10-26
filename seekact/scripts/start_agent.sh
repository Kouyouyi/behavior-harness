#!/usr/bin/env bash
# Trusted operator launcher: host-run Claude Code agent + robot service for one session.
#   bash scripts/start_agent.sh --config <session>/operator-input.json [--resume] [--gemini-key-file F]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# The robot service runs in this interpreter, so it needs the backend's dependencies.
PYTHON="${RAX_PYTHON:-$(command -v python3 || true)}"
[[ -x "$PYTHON" ]] || { echo "Missing Python: '$PYTHON' (set RAX_PYTHON to the robot service environment's python)" >&2; exit 2; }
export PYTHONPATH="$PROJECT_ROOT"
# The agent's own scripts run in RAX_AGENT_PYTHON, an environment without robot SDKs (a session's
# agent_python overrides it).
exec "$PYTHON" -m harness.claude_cli.launcher "$@"
