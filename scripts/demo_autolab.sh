#!/usr/bin/env bash
# Read-only Autolab prechecks. Never logs in, submits, starts, rents or spends.
set -uo pipefail
echo "== Autolab readiness (read-only) =="
if ! command -v autolab >/dev/null; then
  echo "autolab CLI: NOT INSTALLED"
  echo "Next (owner approval required): see docs/MOBIUS_LAB_CODEX_REQUESTS.md item 3."
  exit 0
fi
echo "autolab CLI: $(autolab --version 2>&1 | head -1)"
autolab whoami >/dev/null 2>&1 && echo "login: present" || echo "login: MISSING (owner runs: autolab login)"
echo
echo "This script stops here. Commands that create projects, attach compute or start a Climb"
echo "are listed in docs/MOBIUS_LAB_CODEX_REQUESTS.md and must be run deliberately, with an approved cap."
