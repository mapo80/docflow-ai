#!/usr/bin/env bash
set -euo pipefail
cd /app
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-8000}"; WORKERS="${WORKERS:-1}"
MODULE=$(python - <<'PY'
import importlib.util
print("serve:app" if importlib.util.find_spec("serve") else "main:app")
PY
)
echo "Starting uvicorn ${MODULE} on ${HOST}:${PORT} (workers=${WORKERS})"
exec uvicorn "${MODULE}" --host "${HOST}" --port "${PORT}" --workers "${WORKERS}"
