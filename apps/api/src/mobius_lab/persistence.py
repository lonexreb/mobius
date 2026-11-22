"""Authoritative research ledger: one SQLite file (WAL), explicit state transitions.

The Rust Möbius history is a derived optimizer store; this ledger is the system of record.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  id TEXT PRIMARY KEY, goal TEXT NOT NULL, template_id TEXT NOT NULL, mode TEXT NOT NULL,
  policy_arm TEXT NOT NULL, state TEXT NOT NULL, max_trials INTEGER NOT NULL,
  approved_spend_usd REAL NOT NULL, budget_approved INTEGER NOT NULL DEFAULT 0,
  max_wall_seconds INTEGER NOT NULL, frozen_eval_hash TEXT, replay_of TEXT, notes TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies(id), seq INTEGER NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE, policy_arm TEXT NOT NULL, kind TEXT NOT NULL, action TEXT NOT NULL,
  config TEXT, split TEXT NOT NULL, hypothesis TEXT, rationale TEXT, parent_id TEXT,
  state TEXT NOT NULL, verdict TEXT, mode TEXT NOT NULL, evidence_class TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT '{}', repro TEXT NOT NULL DEFAULT '{}', metrics TEXT NOT NULL DEFAULT '{}',
  validation TEXT NOT NULL DEFAULT '{}', failure TEXT, artifact_path TEXT, wall_seconds REAL,
  rust_told INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, finished_at TEXT);
CREATE INDEX IF NOT EXISTS ix_exp_study ON experiments(study_id, seq);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT NOT NULL REFERENCES studies(id), experiment_id TEXT,
  action TEXT NOT NULL, rationale TEXT NOT NULL, evidence_ids TEXT NOT NULL, uncertainty TEXT,
  controller_version TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_dec_study ON decisions(study_id, id);
CREATE TABLE IF NOT EXISTS budget_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT NOT NULL REFERENCES studies(id), experiment_id TEXT,
  kind TEXT NOT NULL CHECK (kind IN ('reserve','release','spend')), category TEXT NOT NULL,
  amount_usd REAL, source TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_bud_study ON budget_events(study_id);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_evt_study ON events(study_id, id);
CREATE TABLE IF NOT EXISTS imported_source_events (
  platform TEXT NOT NULL, external_id TEXT NOT NULL, payload_hash TEXT NOT NULL, experiment_id TEXT,
  first_seen_at TEXT NOT NULL, PRIMARY KEY (platform, external_id));
"""

STUDY_TRANSITIONS = {
    "DRAFT": {"RUNNING", "CANCELLED"},
    "RUNNING": {"PAUSED", "COMPLETED", "CANCELLED", "FAILED"},
    "PAUSED": {"RUNNING", "CANCELLED"},
    "FAILED": {"RUNNING", "CANCELLED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
}
EXPERIMENT_TRANSITIONS = {
    "QUEUED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"VALID", "INVALID", "FAILED", "TIMED_OUT", "CANCELLED"},
}
TERMINAL = {"VALID", "INVALID", "FAILED", "TIMED_OUT", "CANCELLED"}
JSON_COLS = {"config", "source", "repro", "metrics", "validation", "evidence_ids", "data"}


class LedgerError(Exception):
    """Base class for refused ledger writes."""


class InvalidTransition(LedgerError):
    pass


class BudgetExceeded(LedgerError):
    pass


class DuplicateRun(LedgerError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {k: (json.loads(row[k]) if k in JSON_COLS and row[k] is not None else row[k]) for k in row.keys()}


class Ledger:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.lock = threading.RLock()
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def _all(self, sql: str, *args: Any) -> list[dict]:
        with self.lock:
            return [_row(r) for r in self.conn.execute(sql, args).fetchall()]

    def _one(self, sql: str, *args: Any) -> dict | None:
        with self.lock:
            return _row(self.conn.execute(sql, args).fetchone())

    # ---- events
    def emit(self, study_id: str, type_: str, data: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO events (study_id, type, data, created_at) VALUES (?,?,?,?)",
                              (study_id, type_, json.dumps({"study_id": study_id, **data}), now()))

    def last_error(self, study_id: str) -> str | None:
        row = self._one("SELECT data FROM events WHERE study_id=? AND type='source.error' ORDER BY id DESC LIMIT 1",
                        study_id)
        return row and (row["data"].get("reason") or row["data"].get("error"))

    def events_after(self, study_id: str, after: int, limit: int = 200) -> list[dict]:
        return self._all("SELECT * FROM events WHERE study_id=? AND id>? ORDER BY id LIMIT ?", study_id, after, limit)

    # ---- studies
    def create_study(self, **f: Any) -> dict:
        sid = "study_" + uuid.uuid4().hex[:10]
        with self.tx() as c:
            c.execute(
                "INSERT INTO studies (id, goal, template_id, mode, policy_arm, state, max_trials, approved_spend_usd,"
                " max_wall_seconds, frozen_eval_hash, replay_of, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, f["goal"], f["template_id"], f["mode"], f["policy_arm"], "DRAFT", f["max_trials"],
                 f["approved_spend_usd"], f["max_wall_seconds"], f["frozen_eval_hash"], f.get("replay_of"),
                 f.get("notes"), now()))
            self.emit(sid, "study.created", {"mode": f["mode"], "policy_arm": f["policy_arm"]})
        return self.study(sid)

    def study(self, sid: str) -> dict | None:
        return self._one("SELECT * FROM studies WHERE id=?", sid)

    def studies(self) -> list[dict]:
        return self._all("SELECT * FROM studies ORDER BY created_at DESC")

    def set_study_state(self, sid: str, new: str) -> bool:
        """Apply a transition. Returns False if already in `new` (idempotent); raises if illegal."""
        with self.tx() as c:
            cur = c.execute("SELECT state FROM studies WHERE id=?", (sid,)).fetchone()["state"]
            if cur == new:
                return False
            if new not in STUDY_TRANSITIONS[cur]:
                raise InvalidTransition(f"study {cur} -> {new} is not allowed")
            c.execute("UPDATE studies SET state=? WHERE id=?", (new, sid))
        return True

    def approve_budget(self, sid: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE studies SET budget_approved=1 WHERE id=?", (sid,))

    # ---- experiments
    def experiments(self, sid: str, include_cancelled: bool = False) -> list[dict]:
        rows = self._all("SELECT * FROM experiments WHERE study_id=? ORDER BY seq", sid)
        return rows if include_cancelled else [r for r in rows if r["state"] != "CANCELLED"]

    def experiment(self, eid: str) -> dict | None:
        return self._one("SELECT * FROM experiments WHERE id=?", eid)

    def reserve_experiment(self, study: dict, *, kind: str, action: str, config: dict | None, split: str,
                           hypothesis: str | None, rationale: str, parent_id: str | None, evidence_class: str,
                           estimated_cost_usd: float | None, source: dict | None = None) -> dict:
        """Atomically check every budget and create a QUEUED experiment with its reservation.

        Double-run protection is the single study loop per process (Orchestrator.launch) plus the
        single-process ledger lock (main.create_app). The UNIQUE (study, step) key is a last line of
        defence if either is ever bypassed.
        """
        sid = study["id"]
        with self.tx() as c:
            live = c.execute("SELECT COUNT(*) n, COALESCE(SUM(wall_seconds),0) w FROM experiments"
                             " WHERE study_id=? AND state!='CANCELLED'", (sid,)).fetchone()
            seq = c.execute("SELECT COUNT(*) n FROM experiments WHERE study_id=?", (sid,)).fetchone()["n"]
            if live["n"] >= study["max_trials"]:
                raise BudgetExceeded(f"trial budget exhausted ({live['n']}/{study['max_trials']})")
            if live["w"] >= study["max_wall_seconds"]:
                raise BudgetExceeded(f"wall-time budget exhausted ({live['w']:.0f}s/{study['max_wall_seconds']}s)")
            if estimated_cost_usd is None:
                raise BudgetExceeded("cost of this run is unknown; refusing to reserve an unbounded amount")
            committed = c.execute(
                "SELECT COALESCE(SUM(CASE kind WHEN 'reserve' THEN amount_usd WHEN 'release' THEN -amount_usd"
                " ELSE amount_usd END),0) s FROM budget_events WHERE study_id=? AND amount_usd IS NOT NULL",
                (sid,)).fetchone()["s"]
            if committed + estimated_cost_usd > study["approved_spend_usd"] + 1e-9:
                raise BudgetExceeded(f"reserving ${estimated_cost_usd:.2f} would exceed the approved "
                                     f"${study['approved_spend_usd']:.2f} cap (committed ${committed:.2f})")
            eid = "run_" + uuid.uuid4().hex[:10]
            try:
                c.execute(
                    "INSERT INTO experiments (id, study_id, seq, idempotency_key, policy_arm, kind, action, config,"
                    " split, hypothesis, rationale, parent_id, state, mode, evidence_class, source, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (eid, sid, seq, f"{sid}:{seq}", study["policy_arm"], kind, action,
                     json.dumps(config) if config is not None else None, split, hypothesis, rationale, parent_id,
                     "QUEUED", study["mode"], evidence_class, json.dumps(source or {}), now()))
            except sqlite3.IntegrityError as exc:
                raise DuplicateRun(str(exc)) from exc
            c.execute("INSERT INTO budget_events (study_id, experiment_id, kind, category, amount_usd, source,"
                      " created_at) VALUES (?,?,?,?,?,?,?)",
                      (sid, eid, "reserve", "run", estimated_cost_usd, "estimate before dispatch", now()))
        return self.experiment(eid)

    def transition(self, eid: str, new: str, **fields: Any) -> dict:
        with self.tx() as c:
            cur = c.execute("SELECT state FROM experiments WHERE id=?", (eid,)).fetchone()["state"]
            if new not in EXPERIMENT_TRANSITIONS.get(cur, set()):
                raise InvalidTransition(f"experiment {cur} -> {new} is not allowed")
            sets, vals = ["state=?"], [new]
            for k, v in fields.items():
                sets.append(f"{k}=?")
                vals.append(json.dumps(v) if k in JSON_COLS else v)
            if new in TERMINAL:
                sets.append("finished_at=?")
                vals.append(now())
            c.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE id=?", (*vals, eid))  # noqa: S608 (column names are code constants)
        return self.experiment(eid)

    def settle(self, exp: dict, actual: dict[str, float | None], source: str) -> None:
        """Release the reservation and record actual spend per category (None = unknown, never 0)."""
        with self.tx() as c:
            reserved = c.execute("SELECT amount_usd FROM budget_events WHERE experiment_id=? AND kind='reserve'",
                                 (exp["id"],)).fetchone()
            released = c.execute("SELECT 1 FROM budget_events WHERE experiment_id=? AND kind='release'",
                                 (exp["id"],)).fetchone()
            if released:
                return
            c.execute("INSERT INTO budget_events (study_id, experiment_id, kind, category, amount_usd, source,"
                      " created_at) VALUES (?,?,?,?,?,?,?)",
                      (exp["study_id"], exp["id"], "release", "run", reserved["amount_usd"] if reserved else 0,
                       "reconcile", now()))
            for category, amount in actual.items():
                c.execute("INSERT INTO budget_events (study_id, experiment_id, kind, category, amount_usd, source,"
                          " created_at) VALUES (?,?,?,?,?,?,?)",
                          (exp["study_id"], exp["id"], "spend", category, amount, source, now()))

    def set_verdict(self, eids: str | list[str], verdict: str) -> None:
        with self.lock:
            for eid in [eids] if isinstance(eids, str) else eids:
                self.conn.execute("UPDATE experiments SET verdict=? WHERE id=?", (verdict, eid))

    def force_study_state(self, sid: str, state: str) -> None:
        """Crash path only: record FAILED even where the normal transition table would refuse."""
        with self.lock:
            self.conn.execute("UPDATE studies SET state=? WHERE id=? AND state NOT IN ('COMPLETED','CANCELLED')",
                              (state, sid))

    def untold(self, sid: str) -> list[dict]:
        return self._all("SELECT * FROM experiments WHERE study_id=? AND state='VALID' AND kind='candidate'"
                         " AND split='dev' AND rust_told=0 ORDER BY seq", sid)

    def unmark_told(self, eid: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE experiments SET rust_told=0 WHERE id=?", (eid,))

    def mark_told(self, eid: str) -> bool:
        """True exactly once per experiment: guards the derived Rust history against re-import."""
        with self.tx() as c:
            return c.execute("UPDATE experiments SET rust_told=1 WHERE id=? AND rust_told=0", (eid,)).rowcount == 1

    # ---- decisions
    def add_decision(self, sid: str, experiment_id: str | None, action: str, rationale: str,
                     evidence_ids: list[str], uncertainty: str | None, controller_version: str) -> dict:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO decisions (study_id, experiment_id, action, rationale, evidence_ids, uncertainty,"
                " controller_version, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (sid, experiment_id, action, rationale, json.dumps(evidence_ids), uncertainty, controller_version, now()))
            return self._one("SELECT * FROM decisions WHERE id=?", cur.lastrowid)

    def decisions(self, sid: str) -> list[dict]:
        return self._all("SELECT * FROM decisions WHERE study_id=? ORDER BY id", sid)

    # ---- imports
    def claim_import(self, platform: str, external_id: str, payload_hash: str) -> dict | None:
        """None if this is the first sighting (claimed); otherwise the existing record."""
        with self.tx() as c:
            row = c.execute("SELECT * FROM imported_source_events WHERE platform=? AND external_id=?",
                            (platform, external_id)).fetchone()
            if row:
                return dict(row)
            c.execute("INSERT INTO imported_source_events (platform, external_id, payload_hash, first_seen_at)"
                      " VALUES (?,?,?,?)", (platform, external_id, payload_hash, now()))
        return None

    def link_import(self, platform: str, external_id: str, experiment_id: str | None) -> None:
        with self.lock:
            if experiment_id is None:  # ingestion failed: free the claim so a corrected payload can retry
                self.conn.execute("DELETE FROM imported_source_events WHERE platform=? AND external_id=?",
                                  (platform, external_id))
            else:
                self.conn.execute("UPDATE imported_source_events SET experiment_id=? WHERE platform=? AND external_id=?",
                                  (experiment_id, platform, external_id))

    # ---- budget
    def budget(self, study: dict) -> dict:
        sid = study["id"]
        ev = self._all("SELECT kind, category, amount_usd FROM budget_events WHERE study_id=?", sid)
        reserved = sum((e["amount_usd"] or 0) * (1 if e["kind"] == "reserve" else -1)
                       for e in ev if e["kind"] in ("reserve", "release"))
        spend = [e for e in ev if e["kind"] == "spend"]
        by_cat: dict[str, dict] = {}
        for e in spend:
            cat = by_cat.setdefault(e["category"], {"actual_usd": 0.0, "unknown_events": 0})
            if e["amount_usd"] is None:
                cat["unknown_events"] += 1
            else:
                cat["actual_usd"] += e["amount_usd"]
        exps = self.experiments(sid)
        return {
            "approved_spend_usd": study["approved_spend_usd"],
            "budget_approved": bool(study["budget_approved"]),
            "reserved_usd": round(reserved, 6),
            "observed_spend_usd": round(sum(c["actual_usd"] for c in by_cat.values()), 6),
            "unknown_cost_events": sum(c["unknown_events"] for c in by_cat.values()),
            "by_category": by_cat,
            "max_trials": study["max_trials"],
            "trials_used": len(exps),
            "max_wall_seconds": study["max_wall_seconds"],
            "wall_seconds_used": round(sum(e["wall_seconds"] or 0 for e in exps), 1),
        }

    # ---- restart recovery
    def recover_interrupted(self) -> int:
        """Runs that were in flight when the service died produced no evidence: cancel them, pause their studies."""
        n = 0
        for exp in self._all("SELECT * FROM experiments WHERE state IN ('QUEUED','RUNNING')"):
            with self.lock:
                self.conn.execute("UPDATE experiments SET state='CANCELLED', failure=?, finished_at=? WHERE id=?",
                                  ("interrupted by service restart; no measurement recorded", now(), exp["id"]))
            self.settle(exp, {}, "service restart")
            n += 1
        for study in self._all("SELECT * FROM studies WHERE state='RUNNING'"):
            self.set_study_state(study["id"], "PAUSED")
            self.emit(study["id"], "study.paused", {"reason": "service restarted; resume to continue"})
        return n
