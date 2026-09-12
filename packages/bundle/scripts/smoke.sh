#!/usr/bin/env bash
set -euo pipefail
BIN="${1:?usage: smoke.sh <path-to-binary>}"
DATA="$(mktemp -d)"
PORT="${SEPILOT_SMOKE_PORT:-38655}"
export SEPILOTD_DATA_DIR="$DATA" SEPILOTD_PORT="$PORT" SEPILOTD_URL="http://127.0.0.1:$PORT"
echo "== version =="
"$BIN" --version
echo "== status (self-launches daemon) =="
"$BIN" status
echo "== health =="
curl -sf "http://127.0.0.1:$PORT/api/v1/health" >/dev/null && echo "health OK"
echo "== stop =="
"$BIN" stop || true
sleep 1
pkill -f "$DATA" 2>/dev/null || true
echo "smoke OK"
