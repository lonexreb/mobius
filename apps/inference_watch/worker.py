"""Serve local benchmark briefs from a durable queue. Uses only the standard library."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sqlite3
import time

HILL = Path(__file__).resolve().parents[2] / "experiments" / "inference-watch"
WORKLOADS = ("code", "math", "chat")


def build_brief(workload):
    spec = importlib.util.spec_from_file_location("watch_hill", HILL / "eval.py")
    hill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hill)
    raw = hill.PILOT.read_bytes()
    pilot = json.loads(raw)
    lines = []
    options = [(1.0, "target-only")]
    for config in sorted(pilot[workload]):
        passed, ratio, rows = hill.assess(pilot, workload, config)
        if passed and ratio > 1:
            options.append((ratio, config))
        matches = " and ".join(f"{r['matched']}/{r['prompts']}" for r in rows)
        ids = ", ".join(r["run"] for r in rows)
        verdict = "matches target" if passed else "fails exactness"
        lines.append(f"| {config} | {matches} | {ratio:.3f}× | {verdict} | {ids} |")
    best = max(options)[1]
    return "\n".join(
        [
            f"# Inference brief for {workload} workloads",
            "",
            f"**Recorded pilot choice: {best}.**",
            "",
            "Apple M4 Max (64 GB), llama.cpp build 7480 on Metal, Qwen2.5 7B target and 1.5B draft.",
            "The reference scores 1.0×. Each configuration was evaluated twice on the same eight development prompts.",
            "The table reports identical outputs per recording and the smaller paired throughput ratio. "
            "A draft must match every token in both recordings to qualify.",
            "",
            "| Draft setting | Exact outputs | Paired speedup (minimum) | Outcome | Source runs |",
            "|---|---|---|---|---|",
            *lines,
            "",
            "This brief reuses existing measurements. Fresh hardware, model or engine versions need a new test.",
            f"Pilot data SHA-256: {sha256(raw).hexdigest()}",
            "",
        ]
    )


def tick(db):
    # Claim and fulfill one workload in the same short transaction. Rollback makes it retryable.
    with db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT workload, COUNT(*) AS pending FROM requests
                            WHERE brief IS NULL GROUP BY workload
                            ORDER BY pending DESC, MIN(rowid), workload LIMIT 1""").fetchone()
        if not row:
            return "idle"
        workload, count = row
        brief = build_brief(workload)
        stamp = datetime.now(timezone.utc).isoformat()
        db.execute(
            "UPDATE requests SET brief=?, finished_at=? WHERE workload=? AND brief IS NULL",
            (brief, stamp, workload),
        )
        return f"{stamp} fulfilled {count} {workload} request(s)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    subs = parser.add_subparsers(dest="action", required=True)
    request = subs.add_parser("request")
    request.add_argument("id")
    request.add_argument("workload", choices=WORKLOADS)
    subs.add_parser("tick")
    watch = subs.add_parser("watch")
    watch.add_argument("--interval", type=int, default=60)
    subs.add_parser("brief").add_argument("id")
    args = parser.parse_args()
    if args.action == "watch" and args.interval < 1:
        parser.error("interval must be at least one second")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db, timeout=15)
    db.execute("""CREATE TABLE IF NOT EXISTS requests
                  (id TEXT PRIMARY KEY, workload TEXT NOT NULL, brief TEXT, finished_at TEXT)""")
    if args.action == "request":
        with db:
            db.execute(
                "INSERT INTO requests(id, workload) VALUES (?,?) ON CONFLICT(id) DO NOTHING",
                (args.id, args.workload),
            )
            (existing,) = db.execute(
                "SELECT workload FROM requests WHERE id=?", (args.id,)
            ).fetchone()
            if existing != args.workload:
                parser.error("request id already belongs to another workload")
        print(args.id)
    elif args.action == "brief":
        row = db.execute("SELECT brief FROM requests WHERE id=?", (args.id,)).fetchone()
        if not row or not row[0]:
            parser.exit(1, "Pending request.\n" if row else "Unknown request.\n")
        print(row[0], end="")
    elif args.action == "tick":
        print(tick(db))
    else:
        while True:
            outcome = tick(db)
            if outcome != "idle":
                print(outcome, flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
