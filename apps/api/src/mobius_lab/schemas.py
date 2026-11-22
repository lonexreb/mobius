"""Typed API contract (v1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["local", "autolab", "replay"]
Arm = Literal["target_only", "fixed_spec", "random", "tpe", "metric_greedy", "evidence_aware"]
Split = Literal["dev", "audit"]  # "sealed" is deliberately not representable in this API
EvidenceClass = Literal["local", "observed", "replay", "fixture"]
TemplateId = Literal["local_speculative_qwen25", "qwen3_speculative_ood"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BudgetIn(Strict):
    max_trials: int = Field(8, ge=1, le=24, description="Scored runs, including baselines, replicates and audits")
    approved_spend_usd: float = Field(0.0, ge=0, le=1000)
    max_wall_seconds: int = Field(3600, ge=60, le=86400)


class StudyCreate(Strict):
    template_id: TemplateId
    goal: str = Field(min_length=5, max_length=500)
    mode: Mode
    policy_arm: Arm
    budget: BudgetIn = BudgetIn()
    replay_of: str | None = Field(None, description="Source study id; required for mode=replay")
    notes: str | None = Field(None, max_length=1000)


class BudgetOut(BaseModel):
    approved_spend_usd: float
    budget_approved: bool
    reserved_usd: float
    observed_spend_usd: float
    unknown_cost_events: int
    by_category: dict[str, dict[str, float | int]]
    max_trials: int
    trials_used: int
    max_wall_seconds: int
    wall_seconds_used: float


class DecisionOut(BaseModel):
    id: int
    study_id: str
    experiment_id: str | None
    action: str
    rationale: str
    evidence_ids: list[str]
    uncertainty: str | None
    controller_version: str
    created_at: str


class ExperimentOut(BaseModel):
    id: str
    study_id: str
    seq: int
    policy_arm: str
    kind: Literal["baseline", "candidate"]
    action: str
    config: dict[str, Any] | None
    split: str
    hypothesis: str | None
    rationale: str | None
    parent_id: str | None
    state: str
    verdict: str | None
    mode: str
    evidence_class: EvidenceClass
    source: dict[str, Any]
    repro: dict[str, Any]
    metrics: dict[str, Any]
    validation: dict[str, Any]
    failure: str | None
    wall_seconds: float | None
    created_at: str
    finished_at: str | None


class ExperimentDetail(ExperimentOut):
    requests: list[dict[str, Any]] | None = None
    control_requests: list[dict[str, Any]] | None = None
    decisions: list[DecisionOut] = []


class StudyOut(BaseModel):
    id: str
    goal: str
    template_id: str
    mode: Mode
    policy_arm: str
    state: str
    frozen_eval_hash: str | None
    replay_of: str | None
    notes: str | None
    created_at: str
    budget: BudgetOut
    evidence_label: str
    best_valid: ExperimentOut | None
    promoted: ExperimentOut | None
    latest_decision: DecisionOut | None


class AutolabImport(Strict):
    """Ingestion contract for an Autolab-scored run.

    `artifact` must be the fixed evaluator's artifact as produced inside the Hill. Mapping
    from raw `autolab metrics --format json` output is intentionally not implemented: no real
    payload has been captured yet, and guessing its shape would fabricate provenance.
    """
    study_id: str
    external_run_id: str = Field(min_length=1, max_length=200)
    run_url: str | None = Field(None, max_length=500, pattern=r"^https://")
    hill_hash: str = Field(min_length=7, max_length=128)
    hill_verdict: Literal["scored", "pending", "failed"]
    kind: Literal["baseline", "candidate"]
    config: dict[str, Any] | None
    split: Split
    artifact: dict[str, Any] | None
    cost_usd: float | None = Field(None, ge=0, description="Billed amount if known; null means unknown, not zero")
    failure: str | None = None
    proposer_origin: Literal["mobius", "autolab_agent"] = "mobius"
