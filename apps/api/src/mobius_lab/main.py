"""Möbius Lab API (v1). Localhost demo service: no auth, so bind to 127.0.0.1 only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from . import adapters, policies, report
from .adapters import RunResult
from .orchestrator import Orchestrator, git_sha
from .persistence import BudgetExceeded, InvalidTransition, Ledger
from .schemas import (AutolabImport, DecisionOut, ExperimentDetail, ExperimentOut, StudyCreate, StudyOut)

VERSION = "0.1.0"
API_DIR = Path(__file__).resolve().parents[2]


def create_app(db_path: str | Path | None = None, var_dir: str | Path | None = None,
               run_inline: bool = False) -> FastAPI:
    var = Path(var_dir or API_DIR / "var")
    ledger = Ledger(db_path or os.environ.get("MOBIUS_LAB_DB") or var / "lab.sqlite")
    ev = adapters.load_evaluator()
    orch = Orchestrator(ledger, var, ev, run_inline=run_inline)
    recovered = ledger.recover_interrupted()

    app = FastAPI(title="Möbius Lab", version=VERSION)
    app.state.ledger, app.state.orchestrator, app.state.recovered = ledger, orch, recovered
    origins = os.environ.get("MOBIUS_LAB_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])
    api = APIRouter(prefix="/api/v1")

    def get_study(sid: str) -> dict:
        study = ledger.study(sid)
        if study is None:
            raise HTTPException(404, f"study {sid} not found")
        return study

    def study_out(study: dict) -> dict[str, Any]:
        exps = ledger.experiments(study["id"])
        valid = [e for e in exps if e["kind"] == "candidate" and e["state"] == "VALID" and e["split"] == "dev"]
        decisions = ledger.decisions(study["id"])
        return {**study, "budget": ledger.budget(study), "evidence_label": report.MODE_LABELS[study["mode"]],
                "best_valid": max(valid, key=lambda e: e["metrics"]["speedup_vs_target"]) if valid else None,
                "promoted": next((e for e in exps if e["verdict"] == "promoted"), None),
                "latest_decision": decisions[-1] if decisions else None}

    def start_blockers(study: dict) -> list[str]:
        conn = adapters.connections()
        blockers = []
        if study["approved_spend_usd"] > 0 and not study["budget_approved"]:
            blockers.append("non-zero spend cap has not been explicitly approved")
        if study["mode"] == "autolab":
            blockers += [f"autolab: {conn['autolab']['blocker']}", f"gpu: {conn['gpu']['blocker']}",
                         "remote dispatch is not implemented; import scored runs via POST /api/v1/import/autolab"]
        if study["mode"] == "local" and not conn["local"]["ready"] and orch.adapter_factory == orch._default_adapter:
            blockers.append(f"local evaluator: {conn['local']['blocker']}")
        if study["policy_arm"] == "tpe" and not conn["mobius_cli"]["ready"]:
            blockers.append(f"tpe arm: {conn['mobius_cli']['blocker']}")
        if study["mode"] == "replay" and ledger.study(study["replay_of"] or "") is None:
            blockers.append("replay source study no longer exists")
        return blockers

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": VERSION, **git_sha(), "evaluator_sha256": ev.evaluator_sha256(),
                "controller_version": policies.CONTROLLER_VERSION, "recovered_interrupted_runs": recovered}

    @api.get("/templates")
    def templates() -> dict:
        registry = json.loads(ev.REGISTRY_PATH.read_text())
        m = ev.MANIFEST
        sources = [{k: s[k] for k in ("domain", "repo", "revision", "file", "license", "selected", "url")}
                   for s in registry["sources"]]
        splits = {k: v for k, v in registry["splits"].items()}
        return {"templates": [
            {"id": "local_speculative_qwen25", "title": "LLM inference efficiency (local validation)",
             "runnable_modes": ["local", "replay"], "label": m["local_engine"]["label"],
             "target_model": m["local_engine"]["target"]["ollama"], "draft_model": m["local_engine"]["draft"]["ollama"],
             "engine": m["local_engine"]["name"], "primary_metric": "speedup_vs_target (paired decode tokens/s)",
             "validity": "exact token-ID match with target-only greedy decoding",
             "search_space": m["search_space"], "fixed_spec": m["fixed_spec"], "generation": m["generation"],
             "sources": sources, "splits": splits},
            {"id": "qwen3_speculative_ood", "title": "Qwen3-8B + EAGLE-3 speculative decoding (flagship)",
             "runnable_modes": [], "label": m["flagship"]["status"], "target_model": m["flagship"]["target_model"],
             "draft_model": m["flagship"]["speculator"], "engine": m["flagship"]["engine"],
             "primary_metric": "speedup_vs_target at matched hardware", "validity": "exact token-ID match",
             "search_space": m["flagship"]["search_space_unverified"], "sources": sources, "splits": splits},
        ], "policy_arms": list(policies.ARMS)}

    @api.get("/connections")
    def connections() -> dict:
        return adapters.connections()

    @api.post("/studies", response_model=StudyOut, status_code=201)
    def create_study(body: StudyCreate) -> dict:
        if body.template_id == "qwen3_speculative_ood" and body.mode != "autolab":
            raise HTTPException(422, "the Qwen3-8B flagship needs CUDA + vLLM; only mode=autolab can host it. "
                                     "Use template local_speculative_qwen25 for local validation.")
        frozen = ev.evaluator_sha256()
        if body.mode == "replay":
            src = ledger.study(body.replay_of or "")
            if src is None:
                raise HTTPException(422, "mode=replay requires replay_of: an existing study id")
            if not any(e["artifact_path"] for e in ledger.experiments(src["id"])):
                raise HTTPException(422, "replay source has no recorded runs")
            frozen = src["frozen_eval_hash"]  # replayed artifacts were scored by the source study's evaluator
        elif body.replay_of:
            raise HTTPException(422, "replay_of is only valid with mode=replay")
        study = ledger.create_study(goal=body.goal.strip(), template_id=body.template_id, mode=body.mode,
                                    policy_arm=body.policy_arm, max_trials=body.budget.max_trials,
                                    approved_spend_usd=body.budget.approved_spend_usd,
                                    max_wall_seconds=body.budget.max_wall_seconds, frozen_eval_hash=frozen,
                                    replay_of=body.replay_of, notes=body.notes)
        return study_out(study)

    @api.get("/studies", response_model=list[StudyOut])
    def list_studies() -> list[dict]:
        return [study_out(s) for s in ledger.studies()]

    @api.get("/studies/{sid}", response_model=StudyOut)
    def read_study(sid: str) -> dict:
        return study_out(get_study(sid))

    @api.post("/studies/{sid}/approve-budget", response_model=StudyOut)
    def approve_budget(sid: str) -> dict:
        study = get_study(sid)
        if study["state"] not in ("DRAFT", "PAUSED"):
            raise HTTPException(409, "budget can only be approved while the study is DRAFT or PAUSED")
        ledger.approve_budget(sid)
        ledger.emit(sid, "budget.updated", {"approved_spend_usd": study["approved_spend_usd"], "approved": True})
        return study_out(get_study(sid))

    def _run(sid: str) -> dict:
        study = get_study(sid)
        if study["state"] == "RUNNING":
            orch.launch(sid)  # idempotent: no second loop, no second charge
            return study_out(get_study(sid))
        blockers = start_blockers(study)
        if blockers:
            raise HTTPException(409, {"message": "study cannot start", "blockers": blockers})
        try:
            ledger.set_study_state(sid, "RUNNING")
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        ledger.emit(sid, "study.started", {})
        orch.launch(sid)
        return study_out(get_study(sid))

    api.post("/studies/{sid}/start", response_model=StudyOut)(_run)
    api.post("/studies/{sid}/resume", response_model=StudyOut)(_run)

    @api.post("/studies/{sid}/pause", response_model=StudyOut)
    def pause(sid: str) -> dict:
        get_study(sid)
        try:
            if ledger.set_study_state(sid, "PAUSED"):
                ledger.emit(sid, "study.paused", {"reason": "operator; the run in flight finishes, no new run starts"})
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        return study_out(get_study(sid))

    @api.post("/studies/{sid}/cancel", response_model=StudyOut)
    def cancel(sid: str) -> dict:
        get_study(sid)
        try:
            if ledger.set_study_state(sid, "CANCELLED"):
                orch.cancel(sid)
                ledger.emit(sid, "study.completed", {"reason": "cancelled by operator"})
        except InvalidTransition as exc:
            raise HTTPException(409, str(exc)) from exc
        return study_out(get_study(sid))

    @api.get("/studies/{sid}/experiments", response_model=list[ExperimentOut])
    def list_experiments(sid: str, include_cancelled: bool = False) -> list[dict]:
        get_study(sid)
        return ledger.experiments(sid, include_cancelled)

    @api.get("/studies/{sid}/decisions", response_model=list[DecisionOut])
    def list_decisions(sid: str) -> list[dict]:
        get_study(sid)
        return ledger.decisions(sid)

    @api.get("/experiments/{eid}", response_model=ExperimentDetail)
    def read_experiment(eid: str) -> dict:
        exp = ledger.experiment(eid)
        if exp is None:
            raise HTTPException(404, f"experiment {eid} not found")
        out = {**exp, "decisions": [d for d in ledger.decisions(exp["study_id"])
                                    if d["experiment_id"] == eid or eid in d["evidence_ids"]]}
        if exp["artifact_path"] and Path(exp["artifact_path"]).is_file():
            artifact = json.loads(Path(exp["artifact_path"]).read_text())
            strip = lambda rows: rows and [{k: v for k, v in r.items() if k != "token_ids"} for r in rows]  # noqa: E731
            out["requests"], out["control_requests"] = strip(artifact.get("requests")), strip(artifact.get("control_requests"))
        return out

    @api.get("/studies/{sid}/events")
    async def events(sid: str, request: Request, after_event_id: int = Query(0, ge=0),
                     last_event_id: str | None = Header(None)) -> StreamingResponse:
        get_study(sid)
        cursor = int(last_event_id) if last_event_id and last_event_id.isdigit() else after_event_id

        async def stream():
            nonlocal cursor
            idle = 0
            while not await request.is_disconnected():
                rows = ledger.events_after(sid, cursor)
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {row['id']}\nevent: {row['type']}\ndata: {json.dumps(row['data'])}\n\n"
                idle = 0 if rows else idle + 1
                if idle % 20 == 19:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.get("/studies/{sid}/events.json")
    def events_json(sid: str, after_event_id: int = Query(0, ge=0)) -> list[dict]:
        get_study(sid)
        return ledger.events_after(sid, after_event_id)

    @api.get("/studies/{sid}/report", response_class=PlainTextResponse)
    def study_report(sid: str) -> PlainTextResponse:
        return PlainTextResponse(report.markdown(ledger, get_study(sid)), media_type="text/markdown; charset=utf-8")

    @api.get("/studies/{sid}/export.json")
    def study_export(sid: str) -> dict:
        return report.export(ledger, get_study(sid))

    @api.post("/import/autolab")
    def import_autolab(body: AutolabImport) -> dict:
        study = get_study(body.study_id)
        if study["mode"] != "autolab":
            raise HTTPException(409, "imports attach only to studies created with mode=autolab")
        if body.hill_verdict == "pending":
            raise HTTPException(409, "run has no Hill verdict yet; a submission is not a completed evaluation")
        if body.hill_verdict == "scored" and body.artifact is None:
            raise HTTPException(422, "a scored run must include the evaluator artifact")
        if body.kind == "candidate":
            try:
                ev.validate_config(body.config)
            except ev.EvalError as exc:
                raise HTTPException(422, f"config rejected by pinned allowlist: {exc}") from exc
        elif body.config is not None:
            raise HTTPException(422, "baseline runs carry no candidate config")
        payload_hash = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        seen = ledger.claim_import("autolab", body.external_run_id, payload_hash)
        if seen:
            if seen["payload_hash"] != payload_hash:
                raise HTTPException(409, "this external run id was already imported with a different payload")
            return {"duplicate": True, "experiment": ledger.experiment(seen["experiment_id"])}
        try:
            exp = ledger.reserve_experiment(
                study, kind=body.kind, action="run_baseline" if body.kind == "baseline" else "propose_new",
                config=body.config, split=body.split, hypothesis=None, rationale="Imported Autolab-scored run.",
                parent_id=None, evidence_class="observed", estimated_cost_usd=body.cost_usd or 0.0,
                source={"provider": "autolab", "run_id": body.external_run_id, "run_url": body.run_url,
                        "hill_hash": body.hill_hash, "proposer_origin": body.proposer_origin})
        except BudgetExceeded as exc:
            ledger.link_import("autolab", body.external_run_id, None)
            raise HTTPException(409, f"budget gate refused the import: {exc}") from exc
        ledger.link_import("autolab", body.external_run_id, exp["id"])
        ledger.transition(exp["id"], "RUNNING")
        result = (RunResult("ok", artifact=body.artifact) if body.hill_verdict == "scored"
                  else RunResult("failed", error=body.failure or "Hill reported failure"))
        exp = orch.record_result(study, exp, result, None, actual_cost={"autolab_run": body.cost_usd},
                                 cost_source=f"autolab run {body.external_run_id}"
                                             + ("" if body.cost_usd is not None else " (amount not reported)"))
        # judge what was imported; never dispatch new work from an import
        while True:
            try:
                action = policies.decide(orch._state(get_study(study["id"])))
            except adapters.BridgeError:
                break
            if action.kind not in policies.VERDICT_KINDS:
                break
            ledger.set_verdict(action.target_experiment_id, action.kind)
            orch._decide(study["id"], action, action.target_experiment_id)
        return {"duplicate": False, "experiment": ledger.experiment(exp["id"])}

    app.include_router(api)
    return app


app = create_app() if os.environ.get("MOBIUS_LAB_NO_AUTOAPP") != "1" else None
