"""Study loop: decide -> reserve budget -> execute -> validate -> record -> decide again.

Metrics only ever enter the ledger through `validate`, from an evaluator artifact.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import policies
from .adapters import (REPO, BridgeError, ExecutionAdapter, LocalAdapter, MobiusBridge, ModalAdapter,
                       ReplayAdapter, RunResult)
from .persistence import BudgetExceeded, DuplicateRun, InvalidTransition, Ledger

log = logging.getLogger("mobius_lab")
PER_RUN_TIMEOUT = 900.0
REQUIRED_METRICS = ("throughput_tokens_per_sec", "p50_latency_ms", "p95_latency_ms", "total_output_tokens")
CANDIDATE_METRICS = ("speedup_vs_target", "speedup_ci_low", "speedup_ci_high")


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _dict(x: object) -> dict:
    return x if isinstance(x, dict) else {}


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

    metrics = _dict(artifact.get("metrics"))
    needed = REQUIRED_METRICS + (CANDIDATE_METRICS if exp["kind"] == "candidate" else ())
    bad = [k for k in needed if not _finite(metrics.get(k))]
    if exp["kind"] == "candidate":
        domains = metrics.get("speedup_by_domain")
        if not (isinstance(domains, dict) and domains and all(_finite(v) for v in domains.values())):
            bad.append("speedup_by_domain")
    gate("metrics_complete_and_finite", not bad, f"missing or non-finite: {bad}")
    gate("matches_request", artifact.get("config") == exp["config"] and artifact.get("split") == exp["split"],
         "artifact config/split differ from the requested experiment")
    gate("evaluator_unchanged", artifact.get("evaluator_sha256") == study["frozen_eval_hash"],
         "evaluator hash differs from the hash frozen at study creation")
    gate("dataset_split_pinned", artifact.get("dataset_split_sha256") == split_hashes.get(exp["split"]),
         "prompt split hash does not match the registry")
    engine = _dict(artifact.get("engine"))
    gate("provenance_complete", bool(engine.get("target_model_sha256") and engine.get("engine_version")),
         "model hash or engine version missing")
    correctness = _dict(artifact.get("correctness"))
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
        if study["mode"] == "modal":
            return ModalAdapter(self.manifest)
        return LocalAdapter()

    @staticmethod
    def engine_for(study: dict) -> str:
        """The template fixes the engine, and with it the search space."""
        return "vllm" if study["template_id"] == "qwen3_speculative_ood" else "llama"

    def split_hashes(self) -> dict[str, str]:
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
                alive.join(timeout=2)   # a loop that just saw PAUSED may still be exiting
                if alive.is_alive():
                    return              # genuinely running: it re-reads study state every step
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
            "seed": self.manifest["generation"]["seed"], "search_space": self.ev.space_for(self.engine_for(study)),
            "fixed_spec": self.ev.fixed_spec_for(self.engine_for(study)), "experiments": exps,
            "dev_candidates": [e for e in exps if e["kind"] == "candidate" and e["split"] == "dev"],
            "tpe_ask": lambda: self.bridge.ask(study["id"], study["policy_arm"],
                                               self.ev.space_for(self.engine_for(study))),
        }

    def _decide(self, sid: str, action: policies.Action, experiment_id: str | None) -> None:
        d = self.ledger.add_decision(sid, experiment_id, action.kind, action.rationale, list(action.evidence_ids),
                                     action.uncertainty, policies.CONTROLLER_VERSION)
        self.ledger.emit(sid, "research.decision", {"decision_id": d["id"], "action": action.kind,
                                                     "experiment_id": experiment_id})

    def _complete(self, sid: str, reason: str) -> bool:
        """Close the study. False if it was paused/cancelled meanwhile: the caller must not record a
        terminal decision then, because it will be decided again on resume."""
        try:
            self.ledger.set_study_state(sid, "COMPLETED")
        except InvalidTransition:
            return False
        self.ledger.emit(sid, "study.completed", {"reason": reason})
        return True

    def _fail(self, sid: str, reason: str) -> None:
        self.ledger.force_study_state(sid, "FAILED")
        self.ledger.emit(sid, "source.error", {"reason": reason})

    def _retell(self, study: dict) -> None:
        for exp in self.ledger.untold(study["id"]):
            self._tell(study, exp)

    def run_study(self, sid: str) -> None:
        try:
            while True:
                study = self.ledger.study(sid)
                if study["state"] != "RUNNING":
                    return
                self._retell(study)
                try:
                    action = policies.decide(self._state(study))
                except BridgeError as exc:
                    self._fail(sid, f"Möbius Rust bridge unavailable: {exc}")
                    return
                if action.kind in policies.VERDICT_KINDS:
                    # a verdict on a replicate applies to every run of that config it cites
                    self.ledger.set_verdict(list(action.evidence_ids) or action.target_experiment_id, action.kind)
                    self._decide(sid, action, action.target_experiment_id)
                    continue
                if action.kind in policies.TERMINAL_KINDS:
                    if not self._complete(sid, action.kind):
                        return
                    if action.kind == "promote":
                        self.ledger.set_verdict(action.target_experiment_id, "promoted")
                    self._decide(sid, action, action.target_experiment_id)
                    return
                adapter = self.adapter_factory(study)
                self._adapters[sid] = adapter   # registered before reserving so a cancel can always reach it
                try:
                    exp = self.ledger.reserve_experiment(
                        study, kind="baseline" if action.kind == "run_baseline" else "candidate", action=action.kind,
                        config=action.config, split=action.split, hypothesis=action.hypothesis,
                        rationale=action.rationale, parent_id=action.parent_id,
                        evidence_class=adapter.evidence_class, estimated_cost_usd=adapter.estimated_cost_usd)
                except BudgetExceeded as exc:
                    if self._complete(sid, "budget"):
                        self._decide(sid, policies.Action("stop", f"Budget gate refused the next run: {exc}."), None)
                    return
                except DuplicateRun:
                    return  # another loop already owns this step
                self._decide(sid, action, exp["id"])
                self.ledger.emit(sid, "experiment.queued", {"experiment_id": exp["id"], "action": action.kind})
                self._execute(study, exp, adapter)
        except Exception as exc:  # last-resort guard: a crashed loop must be visible, never silent
            log.exception("study loop crashed: %s", sid)
            try:
                self._fail(sid, f"internal error: {type(exc).__name__}: {exc}")
            except Exception:
                log.exception("could not mark study failed: %s", sid)
        finally:
            self._adapters.pop(sid, None)

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
        started = time.monotonic()
        try:
            if self.ledger.study(sid)["state"] == "CANCELLED":
                result = RunResult("cancelled", error="cancelled by operator before dispatch")
            else:
                result = adapter.evaluate(exp, reference, out_path, timeout)
        except Exception as exc:  # adapter bugs become a recorded failure, not a lost run
            log.exception("adapter crashed: %s", eid)
            result = RunResult("failed", error=f"adapter error: {type(exc).__name__}: {exc}")
        if self.ledger.study(sid)["state"] == "CANCELLED" and result.status != "ok":
            result = RunResult("cancelled", error="cancelled by operator")
        if result.source:
            with self.ledger.lock:
                self.ledger.conn.execute("UPDATE experiments SET source=? WHERE id=?",
                                         (json.dumps(result.source), eid))
        self.record_result(study, exp, result, out_path, actual_cost=result.cost, cost_source=result.cost_source,
                           elapsed=time.monotonic() - started)

    def record_result(self, study: dict, exp: dict, result: RunResult, out_path: Path | None,
                      actual_cost: dict[str, float | None] | None = None, cost_source: str | None = None,
                      elapsed: float = 0.0) -> dict:
        """Single ingestion path for local, replay and imported runs.

        `elapsed` is the orchestrator's own clock around the run. Wall time is charged for every
        outcome, including failures and timeouts, so a failing evaluator cannot outrun the wall cap.
        """
        sid, eid = study["id"], exp["id"]
        if result.status != "ok":
            state = {"timeout": "TIMED_OUT", "cancelled": "CANCELLED"}.get(result.status, "FAILED")
            exp = self.ledger.transition(eid, state, failure=result.error, wall_seconds=elapsed)
            self.ledger.emit(sid, "source.error", {"experiment_id": eid, "state": state, "error": result.error})
        else:
            valid, metrics, validation, repro = validate(result.artifact, exp, study, self.split_hashes())
            exp = self.ledger.transition(
                eid, "VALID" if valid else "INVALID", metrics=metrics, validation=validation, repro=repro,
                artifact_path=str(out_path) if out_path else None,
                wall_seconds=max(elapsed, metrics["wall_seconds"] if _finite(metrics.get("wall_seconds")) else 0.0),
                failure=None if valid else "; ".join(f"{g}: {validation['notes'].get(g, '')}"
                                                     for g in validation["gates_failed"]))
            self.ledger.emit(sid, "experiment.metrics", {"experiment_id": eid})
            self.ledger.emit(sid, "experiment.validated", {"experiment_id": eid, "valid": valid,
                                                           "source": exp["evidence_class"]})
        if actual_cost is None:  # local / replay: no billed service was called
            actual_cost = {"rented_gpu_compute": 0.0, "llm_inference": 0.0, "autolab_agent": 0.0}
            cost_source = "no billed service used (local process, rule-based proposer)"
        self.ledger.settle(exp, actual_cost, cost_source or "unspecified")
        self.ledger.emit(sid, "budget.updated", {"experiment_id": eid})
        if exp["state"] == "VALID" and exp["kind"] == "candidate" and exp["split"] == "dev":
            self._tell(study, exp)
        return exp

    def _tell(self, study: dict, exp: dict) -> None:
        """Mirror a VALID dev result into the Rust optimizer history, exactly once.

        Never raises: the ledger is authoritative. A failed tell is left unmarked and retried at the
        top of the next loop step.
        """
        if not self.ledger.mark_told(exp["id"]):
            return
        try:
            tell_id = self.bridge.tell(study["id"], study["policy_arm"], exp["config"],
                                       exp["metrics"]["speedup_vs_target"])
            self.ledger.emit(study["id"], "experiment.metrics", {"experiment_id": exp["id"], "mobius_tell_id": tell_id})
        except BridgeError as exc:
            self.ledger.unmark_told(exp["id"])
            log.warning("mobius tell deferred for %s: %s", exp["id"], exc)
