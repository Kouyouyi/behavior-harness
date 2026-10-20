#!/usr/bin/env bash
set -euo pipefail
PORT=${1:-18769}
case "$PORT" in 18769|18770) ;; *) echo 'Expected port 18769 or 18770' >&2; exit 2 ;; esac
exec ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L "127.0.0.1:$PORT:127.0.0.1:$PORT" huoshan
