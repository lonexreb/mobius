#!/usr/bin/env bash
# Start the reader demo on loopback. The process handles individual requests as they arrive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -z "${WATCH_EGBE_ANALYZER:-}" ]]; then
  for helper in "$HOME"/.codex/plugins/cache/egbe/egbe/*/skills/egbe-os-analytics/scripts/umami_funnel.py; do
    if [[ -f "$helper" ]]; then export WATCH_EGBE_ANALYZER="$helper"; fi
  done
fi
args=(-m uvicorn demo:app --app-dir apps/inference_watch --host 127.0.0.1 --port "${WATCH_PORT:-8766}" --no-access-log)
if [[ -n "${WATCH_PYTHON:-}" ]]; then exec "$WATCH_PYTHON" "${args[@]}"; fi
if [[ -x apps/api/.venv/bin/python ]]; then exec apps/api/.venv/bin/python "${args[@]}"; fi
exec uv run --no-project --python 3.12 --with ./apps/api python "${args[@]}"
