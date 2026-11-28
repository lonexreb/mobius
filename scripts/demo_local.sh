#!/usr/bin/env bash
# Start the Möbius Lab API and web UI on localhost. Spends nothing; measures nothing until you start a study.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)" >&2; exit 1; }; }
need uv "https://docs.astral.sh/uv/"; need node "Node 20+"; need npm "Node 20+"; need cargo "https://rustup.rs"
command -v llama-server >/dev/null || echo "warning: llama-server not found; local studies will be blocked (brew install llama.cpp)" >&2

cd "$ROOT"
[ -x target/debug/mobius ] || [ -x target/release/mobius ] || cargo build -p mobius-cli
[ -f experiments/frontier2026/speculative/data/prompts_dev.jsonl ] || \
  uv run --python 3.12 --with pyarrow python experiments/frontier2026/speculative/fetch_prompts.py

cd "$ROOT/apps/api"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install -q -p .venv -e ".[dev]"
.venv/bin/uvicorn mobius_lab.main:app --host 127.0.0.1 --port 8000 --log-level warning --timeout-graceful-shutdown 3 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

cd "$ROOT/apps/web"
[ -d node_modules ] || npm install
for _ in $(seq 30); do curl -fsS http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:8000/api/v1/health >/dev/null || { echo "API did not start" >&2; exit 1; }
echo "API  http://127.0.0.1:8000/api/v1/health"
echo "UI   http://127.0.0.1:5173/"
npm run dev
