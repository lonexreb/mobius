"""Shared test fixtures. Everything produced here is FIXTURE data, never a measurement."""

import os

os.environ["MOBIUS_LAB_NO_AUTOAPP"] = "1"

import json  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mobius_lab import adapters, policies  # noqa: E402
from mobius_lab.adapters import RunResult  # noqa: E402
from mobius_lab.main import create_app  # noqa: E402

EV = adapters.load_evaluator()
SPLITS = {k: v["sha256"] for k, v in json.loads(EV.REGISTRY_PATH.read_text())["splits"].items()}
C44 = {"draft_p_min": 0.75, "lookahead_tokens": 4}


def make_artifact(config, split="dev", speedup=1.2, ci_low=None, domains=None, exact=True, truncated=0, diverged=0,
                  control_ok=True, drop_metric=None, wall=1.0):
    """Evaluator-shaped artifact with chosen outcomes."""
    metrics = {"throughput_tokens_per_sec": 60.0, "throughput_by_domain": {"chat": 60.0, "code": 60.0, "math": 60.0},
               "p50_latency_ms": 1500.0, "p95_latency_ms": 1900.0, "time_to_first_token_ms": 120.0,
               "draft_acceptance_rate": 0.8 if config else None, "total_output_tokens": 2304,
               "total_prompt_tokens": 2989, "prompts": 24}
    correctness = None
    if config is not None:
        d = domains or {"chat": speedup, "code": speedup, "math": speedup}
        metrics.update({"speedup_vs_target": speedup, "speedup_by_domain": d,
                        "speedup_ci_low": speedup - 0.05 if ci_low is None else ci_low,
                        "speedup_ci_high": speedup + 0.05, "paired_prompts": 24})
        bad = truncated + diverged
        correctness = {"compared": 24, "matched": 24 - bad, "output_token_match_rate": (24 - bad) / 24,
                       "truncated_prefix_prompt_ids": [f"t{i}" for i in range(truncated)],
                       "diverged": [{"prompt_id": f"d{i}", "first_divergent_index": 3} for i in range(diverged)],
                       "missing_prompt_ids": [], "exact_token_match": exact and not bad,
                       "control_reproduces_reference": control_ok}
    if drop_metric:
        metrics.pop(drop_metric)
    return {"schema_version": 2, "kind": "candidate" if config else "target_only", "config": config, "split": split,
            "evaluator_sha256": EV.evaluator_sha256(), "dataset_split_sha256": SPLITS[split],
            "generation": {"max_output_tokens": 96}, "design": "fixture",
            "engine": {"name": "fixture-engine", "engine_version": "fixture", "target_model": "fixture",
                       "target_model_sha256": "f" * 64, "draft_model_sha256": "d" * 64 if config else None},
            "hardware": {"cpu": "fixture"}, "started_at": "2026-01-01T00:00:00+00:00", "wall_seconds": wall,
            "warmup_seconds": 0.1, "server_rss_gb": None, "metrics": metrics, "correctness": correctness,
            "requests": [], "control_requests": [] if config else None}


class FixtureAdapter:
    """Scripted outcomes keyed by (config, split); each key may list several results for replicates."""

    evidence_class = "fixture"

    def __init__(self, script=None, default=None, evidence_class="fixture"):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default or {}
        self.evidence_class = evidence_class
        self.calls = []

    def evaluate(self, exp, reference_path, out_path, timeout):
        self.calls.append((exp["config"], exp["split"]))
        if exp["config"] is not None and reference_path is None:
            return RunResult("failed", error="no reference")
        queue = self.script.get((policies.key(exp["config"]), exp["split"]))
        outcome = queue.pop(0) if queue else dict(self.default)
        status = outcome.pop("status", "ok")
        if status != "ok":
            return RunResult(status, error=outcome.get("error", "scripted failure"))
        artifact = make_artifact(exp["config"], exp["split"], **outcome)
        out_path.write_text(json.dumps(artifact))
        return RunResult("ok", artifact=artifact)

    def cancel(self):
        pass


@pytest.fixture
def make_client(tmp_path):
    apps = []

    def build(adapter=None):
        for old in apps:  # simulate the previous process exiting: it releases the ledger lock
            old.state.lock_file.close()
        apps.clear()
        app = create_app(db_path=tmp_path / "lab.sqlite", var_dir=tmp_path / "var", run_inline=True)
        adapter = adapter or FixtureAdapter()
        app.state.orchestrator.adapter_factory = lambda study: adapter
        apps.append(app)
        return TestClient(app), adapter
    yield build
    for old in apps:
        old.state.lock_file.close()


@pytest.fixture
def client(make_client):
    return make_client()[0]


def new_study(client, arm="evidence_aware", mode="local", trials=8, **extra):
    body = {"template_id": "local_speculative_qwen25", "goal": "Improve throughput, keep outputs identical",
            "mode": mode, "policy_arm": arm, "budget": {"max_trials": trials, **extra.pop("budget", {})}, **extra}
    r = client.post("/api/v1/studies", json=body)
    assert r.status_code == 201, r.text
    return r.json()
