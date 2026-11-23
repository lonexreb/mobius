"""Loopback-only reader demo. Each request starts bounded work, including after a tab closes."""

from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import UUID
import json
import os
import sqlite3
import subprocess
import sys
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware
import worker

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parents[1] / "docs/evidence/inference-watch"
DB = Path(os.environ.get("WATCH_DB", ROOT.parent / "api/var/reader-demo.sqlite"))
DOMAINS = Literal["code", "math", "chat"]


class Intake(BaseModel):
    id: UUID
    workload: DOMAINS


class Reply(BaseModel):
    choice: Literal["target-only", "draft", "undecided"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def connection():
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def deliver():
    with connection() as db:
        while worker.tick(db) != "idle":
            pass


@asynccontextmanager
async def lifespan(app):
    DB.parent.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        db.executescript("""
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, workload TEXT NOT NULL, brief TEXT, finished_at TEXT);
          CREATE TABLE IF NOT EXISTS readers(id TEXT PRIMARY KEY, requested TEXT NOT NULL, variant TEXT NOT NULL,
            viewed TEXT, choice TEXT, answered TEXT);
          CREATE TABLE IF NOT EXISTS trials(workload TEXT PRIMARY KEY, started TEXT NOT NULL);
        """)
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.state.executor = executor
        executor.submit(deliver)
        yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
)


@app.middleware("http")
async def local_only(request: Request, call_next):
    origin = request.headers.get("origin")
    if (
        request.method == "POST"
        and origin
        and origin != str(request.base_url).rstrip("/")
    ):
        return JSONResponse({"detail": "Use the local reader page"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/")
@app.get("/loop")
def page():
    return FileResponse(ROOT / "reader.html")


@app.get("/evidence/{report}")
def report_file(report: Literal["reference", "rejected-draft"]):
    return FileResponse(
        EVIDENCE / f"autolab-{report}.json", media_type="application/json"
    )


@app.post("/api/requests")
def intake(body: Intake):
    rid = str(body.id)
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute(
            "SELECT workload FROM requests WHERE id=?", (rid,)
        ).fetchone()
        if previous and previous["workload"] != body.workload:
            raise HTTPException(409, "This receipt belongs to another workload")
        if not previous:
            trial = db.execute(
                "SELECT 1 FROM trials WHERE workload=?", (body.workload,)
            ).fetchone()
            assigned = db.execute(
                """SELECT COUNT(*) FROM readers JOIN requests USING(id)
                WHERE workload=? AND variant='comparison'""",
                (body.workload,),
            ).fetchone()[0]
            variant = "comparison" if trial and assigned < 3 else "summary"
            db.execute(
                "INSERT INTO requests(id,workload) VALUES (?,?)", (rid, body.workload)
            )
            db.execute(
                "INSERT INTO readers(id,requested,variant) VALUES (?,?,?)",
                (rid, now(), variant),
            )
    app.state.executor.submit(deliver)
    return {"id": rid}


def reader(db, rid):
    row = db.execute(
        "SELECT * FROM requests JOIN readers USING(id) WHERE id=?", (str(rid),)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Receipt not found")
    return row


@app.get("/api/requests/{rid}")
def read_brief(rid: UUID):
    with connection() as db:
        row = reader(db, rid)
    result = {
        k: row[k] for k in ("workload", "variant", "finished_at", "choice", "brief")
    }
    if row["brief"]:
        raw = (worker.HILL / "pilot.json").read_bytes()
        if sha256(raw).hexdigest() in row["brief"]:
            result["records"] = json.loads(raw)[row["workload"]]
    return result


@app.post("/api/requests/{rid}/view")
def view(rid: UUID):
    with connection() as db:
        if not reader(db, rid)["brief"]:
            raise HTTPException(409, "Brief pending")
        db.execute(
            "UPDATE readers SET viewed=COALESCE(viewed,?) WHERE id=?", (now(), str(rid))
        )
    return {"ok": True}


@app.post("/api/requests/{rid}/feedback")
def feedback(rid: UUID, body: Reply):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = reader(db, rid)
        if not row["viewed"]:
            raise HTTPException(409, "Open the brief before responding")
        if row["choice"] and row["choice"] != body.choice:
            raise HTTPException(409, "A decision has already been recorded")
        db.execute(
            "UPDATE readers SET choice=?, answered=COALESCE(answered,?) WHERE id=?",
            (body.choice, now(), str(rid)),
        )
        count = db.execute(
            """SELECT COUNT(*) FROM readers JOIN requests USING(id)
            WHERE workload=? AND variant='summary' AND choice='undecided'""",
            (row["workload"],),
        ).fetchone()[0]
        if count >= 2:
            db.execute(
                "INSERT OR IGNORE INTO trials VALUES (?,?)", (row["workload"], now())
            )
    return {"ok": True}


@app.get("/api/loop")
def results():
    with connection() as db:
        cells = [
            dict(row)
            for row in db.execute("""
            SELECT workload,variant,COUNT(*) AS requested, COUNT(finished_at) AS ready,
            COUNT(viewed) AS opened,COUNT(choice) AS answered,
            SUM(CASE WHEN choice IN ('target-only','draft') THEN 1 ELSE 0 END) AS decided,
            SUM(CASE WHEN choice='undecided' THEN 1 ELSE 0 END) AS undecided
            FROM requests JOIN readers USING(id) GROUP BY workload,variant ORDER BY workload,variant""")
        ]
        trials = [
            dict(row) for row in db.execute("SELECT * FROM trials ORDER BY started")
        ]
    return {"scope": "local demo receipts", "groups": cells, "trials": trials}


@app.get("/api/egbe-snapshot")
def egbe_snapshot():
    # Adapt observed local receipts to Egbe's public offline format; expose no receipt tokens.
    snapshot = json.loads((ROOT / "rehearsal/egbe-events.json").read_text())
    snapshot.update(
        evidence_class="LOCAL DEMO: receipt-based sessions",
        rows=[],
        website={"id": "local", "name": "Inference Watch demo", "domain": "localhost"},
    )
    with connection() as db:
        for index, row in enumerate(
            db.execute("SELECT requested,viewed FROM readers ORDER BY rowid"), 1
        ):
            for suffix, event, stamp in [
                ("open", None, row["viewed"]),
                ("cta", "cta_click", row["requested"]),
                ("ready", "activated", row["viewed"]),
            ]:
                if stamp:
                    snapshot["rows"].append(
                        {
                            "id": f"demo-{index}-{suffix}",
                            "sessionId": f"demo-{index}",
                            "createdAt": stamp,
                            "urlPath": "/brief",
                            "urlQuery": "utm_source=local-demo",
                            "eventType": 2 if event else 1,
                            **({"eventName": event} if event else {}),
                        }
                    )
    stamps = [
        int(datetime.fromisoformat(r["createdAt"]).timestamp() * 1000)
        for r in snapshot["rows"]
    ]
    cutoff = int(datetime.now(timezone.utc).timestamp() * 1000)
    snapshot["window"] = {"startAt": min(stamps, default=cutoff), "endAt": cutoff}
    snapshot["coverage"] = {
        "base_events": "local receipt workflow",
        "event_data": "local demo only",
    }
    return snapshot


@app.get("/api/egbe-report")
def egbe_report():
    helper = os.environ.get("WATCH_EGBE_ANALYZER")
    if not helper or not Path(helper).is_file():
        raise HTTPException(
            503,
            "Set WATCH_EGBE_ANALYZER to Egbe's official umami_funnel.py to analyze local receipts.",
        )
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "events.json"
        report = Path(directory) / "report.json"
        source.write_text(json.dumps(egbe_snapshot()))
        try:
            subprocess.run(
                [
                    sys.executable,
                    helper,
                    "--from-events",
                    str(source),
                    "--json",
                    str(report),
                ],
                check=True,
                timeout=12,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError):
            raise HTTPException(
                503,
                "Egbe's local analyzer did not finish. The receipt ledger remains available.",
            )
        data = json.loads(report.read_text())
    return {
        "scope": "Egbe offline analysis of local demo receipts",
        "included": data["included_sessions"],
        "requests": data["events"].get("cta_click", {}).get("unique_sessions", 0),
        "activations": data["events"].get("activated", {}).get("unique_sessions", 0),
    }
