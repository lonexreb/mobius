#!/usr/bin/env bash
# Validate the public event contract. Optionally check the connected project privately.
set -euo pipefail
curl -fsS --max-time 15 https://os.hz.egbe.dev/analytics/events | python3 -c '
import json, sys
body = json.load(sys.stdin)
data = body["data"]
events = {e["name"]: e for e in data["events"]}
if body["code"] != 200 or not {"activated", "cta_click"} <= events.keys():
    raise SystemExit("Egbe event contract missing expected events")
print("Verified public contract", data["schema_version"])
for name in ("cta_click", "activated"):
    print(name, ", ".join(events[name]["required_payload"]))
'
if [[ $# -eq 0 ]]; then exit 0; fi
if [[ $# -ne 1 || ! -f "$1" ]]; then echo "Pass the installed Egbe quickstart helper path" >&2; exit 2; fi
for resource in /me/limits /wallet/balance /wallet/policy; do
  python3 "$1" --egbe-request "$resource" -s | python3 -c '
import json, sys
if json.load(sys.stdin).get("code") != 200:
    raise SystemExit("Authenticated check failed")
print("Read", sys.argv[1], "(inspect details privately before enabling)")
' "$resource"
done
