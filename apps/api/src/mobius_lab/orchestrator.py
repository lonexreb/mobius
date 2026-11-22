"""Study loop: decide -> reserve budget -> execute -> validate -> record -> decide again.

Metrics only ever enter the ledger through `validate`, from an evaluator artifact.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from . import policies
from .adapters import REPO, BridgeError, ExecutionAdapter, LocalAdapter, MobiusBridge, ReplayAdapter, RunResult
from .persistence import BudgetExceeded, DuplicateRun, Ledger

log = logging.getLogger("mobius_lab")
PER_RUN_TIMEOUT = 900.0
REQUIRED_METRICS = ("throughput_tokens_per_sec", "p50_latency_ms", "p95_latency_ms", "total_output_tokens")
CANDIDATE_METRICS = ("speedup_vs_target", "speedup_ci_low", "speedup_ci_high")


def git_sha() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, timeout=10)
        return {"git_sha": sha.stdout.strip() or None, "git_dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def validate(artifact: dict, exp: dict, study: dict, split_hashes: dict[str, str]) -> tuple[bool, dict, dict, dict]:
    """Apply the validity gates. Returns (valid, metrics, validation, repro). Never fills missing values."""
    passed, failed, notes = [], [], {}

    def gate(name: str, ok: bool, note: str | None = None) -> None:
        (passed if ok else failed).append(name)
        if note and not ok:
            notes[name] = note

    metrics = artifact.get("metrics") or {}
    needed = REQUIRED_METRICS + (CANDIDATE_METRICS if exp["kind"] == "candidate" else ())
    bad = [k for k in needed if not isinstance(metrics.get(k), (int, float)) or not math.isfinite(metrics[k])]
    gate("metrics_complete_and_finite", not bad, f"missing or non-finite: {bad}")
    gate("matches_request", artifact.get("config") == exp["config"] and artifact.get("split") == exp["split"],
         "artifact config/split differ from the requested experiment")
    gate("evaluator_unchanged", artifact.get("evaluator_sha256") == study["frozen_eval_hash"],
         "evaluator hash differs from the hash frozen at study creation")
    gate("dataset_split_pinned", artifact.get("dataset_split_sha256") == split_hashes.get(exp["split"]),
         "prompt split hash does not match the registry")
    engine = artifact.get("engine") or {}
    gate("provenance_complete", bool(engine.get("target_model_sha256") and engine.get("engine_version")),
         "model hash or engine version missing")
    correctness = artifact.get("correctness") or {}
    if exp["kind"] == "candidate":
        gate("control_reproduces_reference", correctness.get("control_reproduces_reference") is True,
             "in-run target-only control did not reproduce the stored reference tokens")
        truncated = len(correctness.get("truncated_prefix_prompt_ids") or [])
        diverged = len(correctness.get("diverged") or [])
        gate("exact_token_match", correctness.get("exact_token_match") is True,
             f"{correctness.get('matched')}/{correctness.get('compared')} prompts matched; "
             f"{truncated} stopped early (prefix equal), {diverged} diverged")
    validation = {"gates_passed": passed, "gates_failed": failed, "notes": notes,
                  "exact_token_match": correctness.get("exact_token_match"),
                  "output_token_match_rate": correctness.get("output_token_match_rate"),
                  "truncated_prefix": len(correctness.get("truncated_prefix_prompt_ids") or []),
                  "diverged": len(correctness.get("diverged") or []),
                  "compared": correctness.get("compared"), "heldout_accessed_online": False}
    repro = {**git_sha(), "evaluator_sha256": artifact.get("evaluator_sha256"),
             "dataset_split_sha256": artifact.get("dataset_split_sha256"), "engine": engine.get("name"),
             "engine_version": engine.get("engine_version"), "engine_argv": engine.get("argv"),
             "target_model": engine.get("target_model"), "target_model_sha256": engine.get("target_model_sha256"),
             "draft_model": engine.get("draft_model"), "draft_model_sha256": engine.get("draft_model_sha256"),
             "hardware": artifact.get("hardware"), "generation": artifact.get("generation"),
             "design": artifact.get("design"), "started_at": artifact.get("started_at"),
             "replayed_from": artifact.get("replayed_from")}
    extra = {"server_rss_gb": artifact.get("server_rss_gb"), "warmup_seconds": artifact.get("warmup_seconds"),
             "wall_seconds": artifact.get("wall_seconds")}
    return not failed, {**metrics, **extra}, validation, repro


class Orchestrator:
    def __init__(self, ledger: Ledger, var_dir: Path, evaluator, run_inline: bool = False) -> None:
        self.ledger, self.var, self.ev, self.run_inline = ledger, var_dir, evaluator, run_inline
        self.manifest = evaluator.MANIFEST
        self.bridge = MobiusBridge(var_dir / "mobius_state", self.manifest["search_space"])
        self.adapter_factory: Callable[[dict], ExecutionAdapter] = self._default_adapter
        self._threads: dict[str, threading.Thread] = {}
        self._adapters: dict[str, ExecutionAdapter] = {}
        self._lock = threading.Lock()

    def _default_adapter(self, study: dict) -> ExecutionAdapter:
        if study["mode"] == "replay":
            return ReplayAdapter(self.ledger, study["replay_of"], study["id"])
        return LocalAdapter()

    def split_hashes(self) -> dict[str, str]:
        import json
        registry = json.loads(self.ev.REGISTRY_PATH.read_text())
        return {name: meta["sha256"] for name, meta in registry["splits"].items() if name in self.ev.ONLINE_SPLITS}

    # ---- lifecycle
    def launch(self, study_id: str) -> None:
        """Start the loop for a RUNNING study unless one is already alive (idempotent)."""
        if self.run_inline:
            self.run_study(study_id)
            return
        with self._lock:
            alive = self._threads.get(study_id)
            if alive and alive.is_alive():
                return
            thread = threading.Thread(target=self.run_study, args=(study_id,), daemon=True, name=f"study-{study_id}")
            self._threads[study_id] = thread
            thread.start()

    def cancel(self, study_id: str) -> None:
        adapter = self._adapters.get(study_id)
        if adapter:
            adapter.cancel()

    # ---- loop
    def _state(self, study: dict) -> dict:
        exps = self.ledger.experiments(study["id"])
        return {
            "study_id": study["id"], "arm": study["policy_arm"], "max_trials": study["max_trials"],
            "seed": self.manifest["generation"]["seed"], "search_space": self.manifest["search_space"],
            "fixed_spec": self.manifest["fixed_spec"], "experiments": exps,
            "dev_candidates": [e for e in exps if e["kind"] == "candidate" and e["split"] == "dev"],
            "tpe_ask": lambda: self.bridge.ask(study["id"], study["policy_arm"]),
        }

    def _decide(self, sid: str, action: policies.Action, experiment_id: str | None) -> None:
        d = self.ledger.add_decision(sid, experiment_id, action.kind, action.rationale, list(action.evidence_ids),
                                     action.uncertainty, policies.CONTROLLER_VERSION)
        self.ledger.emit(sid, "research.decision", {"decision_id": d["id"], "action": action.kind,
                                                     "experiment_id": experiment_id})

    def _finish(self, sid: str, state: str, reason: str) -> None:
        if self.ledger.set_study_state(sid, state):
            self.ledger.emit(sid, "study.completed" if state == "COMPLETED" else "source.error", {"reason": reason})

    def run_study(self, sid: str) -> None:
        try:
            while True:
                study = self.ledger.study(sid)
                if study["state"] != "RUNNING":
                    return
                try:
                    action = policies.decide(self._state(study))
                except BridgeError as exc:
                    self._finish(sid, "FAILED", f"Möbius Rust bridge unavailable: {exc}")
                    return
                if action.kind in policies.VERDICT_KINDS:
                    self.ledger.set_verdict(action.target_experiment_id, action.kind)
                    self._decide(sid, action, action.target_experiment_id)
                    continue
                if action.kind in policies.TERMINAL_KINDS:
                    if action.kind == "promote":
                        self.ledger.set_verdict(action.target_experiment_id, "promoted")
                    self._decide(sid, action, action.target_experiment_id)
                    self._finish(sid, "COMPLETED", action.kind)
                    return
                adapter = self.adapter_factory(study)
                try:
                    exp = self.ledger.reserve_experiment(
                        study, kind="baseline" if action.kind == "run_baseline" else "candidate", action=action.kind,
                        config=action.config, split=action.split, hypothesis=action.hypothesis,
                        rationale=action.rationale, parent_id=action.parent_id,
                        evidence_class=adapter.evidence_class, estimated_cost_usd=0.0)
                except BudgetExceeded as exc:
                    self._decide(sid, policies.Action("stop", f"Budget gate refused the next run: {exc}."), None)
                    self._finish(sid, "COMPLETED", "budget")
                    return
                except DuplicateRun:
                    return  # another loop already owns this step
                self._decide(sid, action, exp["id"])
                self.ledger.emit(sid, "experiment.queued", {"experiment_id": exp["id"], "action": action.kind})
                self._execute(study, exp, adapter)
        except Exception as exc:  # last-resort guard: a crashed loop must be visible, never silent
            log.exception("study loop crashed: %s", sid)
            try:
                self._finish(sid, "FAILED", f"internal error: {type(exc).__name__}: {exc}")
            except Exception:
                log.exception("could not mark study failed: %s", sid)

    def _execute(self, study: dict, exp: dict, adapter: ExecutionAdapter) -> None:
        sid, eid = study["id"], exp["id"]
        self.ledger.transition(eid, "RUNNING")
        self.ledger.emit(sid, "experiment.started", {"experiment_id": eid})
        out_dir = self.var / "artifacts" / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{eid}.json"
        reference = None
        if exp["kind"] == "candidate":
            base = next(e for e in self.ledger.experiments(sid)
                        if e["kind"] == "baseline" and e["split"] == exp["split"] and e["state"] == "VALID")
            reference = Path(base["artifact_path"])
        used = self.ledger.budget(study)["wall_seconds_used"]
        timeout = max(1.0, min(PER_RUN_TIMEOUT, study["max_wall_seconds"] - used))
        self._adapters[sid] = adapter
        try:
            result = adapter.evaluate(exp, reference, out_path, timeout)
        except Exception as exc:  # adapter bugs become a recorded failure, not a lost run
            log.exception("adapter crashed: %s", eid)
            result = RunResult("failed", error=f"adapter error: {type(exc).__name__}: {exc}")
        finally:
            self._adapters.pop(sid, None)
        self.record_result(study, exp, result, out_path)

    def record_result(self, study: dict, exp: dict, result: RunResult, out_path: Path | None,
                      actual_cost: dict[str, float | None] | None = None, cost_source: str | None = None) -> dict:
        """Single ingestion path for local, replay and imported runs."""
        sid, eid = study["id"], exp["id"]
        if result.status != "ok":
            state = {"timeout": "TIMED_OUT", "cancelled": "CANCELLED"}.get(result.status, "FAILED")
            exp = self.ledger.transition(eid, state, failure=result.error)
            self.ledger.emit(sid, "source.error", {"experiment_id": eid, "state": state, "error": result.error})
        else:
            valid, metrics, validation, repro = validate(result.artifact, exp, study, self.split_hashes())
            exp = self.ledger.transition(
                eid, "VALID" if valid else "INVALID", metrics=metrics, validation=validation, repro=repro,
                artifact_path=str(out_path) if out_path else None, wall_seconds=metrics.get("wall_seconds"),
                failure=None if valid else "; ".join(f"{g}: {validation['notes'].get(g, '')}"
                                                     for g in validation["gates_failed"]))
            self.ledger.emit(sid, "experiment.metrics", {"experiment_id": eid})
            self.ledger.emit(sid, "experiment.validated", {"experiment_id": eid, "valid": valid,
                                                           "source": exp["evidence_class"]})
            if valid and exp["kind"] == "candidate" and exp["split"] == "dev":
                self._tell(study, exp)
        if actual_cost is None:  # local / replay: no billed service was called
            actual_cost = {"rented_gpu_compute": 0.0, "llm_inference": 0.0, "autolab_agent": 0.0}
            cost_source = "no billed service used (local process, rule-based proposer)"
        self.ledger.settle(exp, actual_cost, cost_source or "unspecified")
        self.ledger.emit(sid, "budget.updated", {"experiment_id": eid})
        return exp

    def _tell(self, study: dict, exp: dict) -> None:
        """Mirror a VALID dev result into the Rust optimizer history, exactly once. Best effort."""
        if not self.ledger.mark_told(exp["id"]):
            return
        try:
            tell_id = self.bridge.tell(study["id"], study["policy_arm"], exp["config"],
                                       exp["metrics"]["speedup_vs_target"])
            self.ledger.emit(study["id"], "experiment.metrics", {"experiment_id": exp["id"], "mobius_tell_id": tell_id})
        except BridgeError as exc:
            with self.ledger.lock:
                self.ledger.conn.execute("UPDATE experiments SET rust_told=0 WHERE id=?", (exp["id"],))
            log.warning("mobius tell skipped for %s: %s", exp["id"], exc)
            if study["policy_arm"] == "tpe":
                raise
