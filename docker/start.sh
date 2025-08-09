#!/usr/bin/env bash
set -euo pipefail
cd /app

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# Prefer serve:app if present (registers overlay middleware), otherwise fall back to main:app
if python - <<'PY' 2>/dev/null; then
import importlib.util, sys
spec = importlib.util.find_spec("serve")
print("serve" if spec else "main")
PY
then
  MODULE=$(python - <<'PY'
import importlib.util
print("serve:app" if importlib.util.find_spec("serve") else "main:app")
PY
)
else
  MODULE="main:app"
fi

echo "Starting uvicorn ${MODULE} on ${HOST}:${PORT} (workers=${WORKERS})"
exec uvicorn "${MODULE}" --host "${HOST}" --port "${PORT}" --workers "${WORKERS}"
