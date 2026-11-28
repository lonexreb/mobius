"""API contract, lifecycle, budget gates, idempotency, imports, restart persistence."""

import json
import threading

import pytest
from conftest import C44, FixtureAdapter, make_artifact, new_study
from mobius_lab import policies
from mobius_lab.adapters import MobiusBridge
from mobius_lab.persistence import BudgetExceeded, InvalidTransition, Ledger

K44 = policies.key(C44)
V1 = "/api/v1"


def run(client, sid):
    r = client.post(f"{V1}/studies/{sid}/start")
    assert r.status_code == 200, r.text
    return r.json()


# ---------- contract ----------

def test_health_reports_real_version_and_evaluator_hash(client):
    h = client.get(f"{V1}/health").json()
    assert h["status"] == "ok" and h["version"] == "0.1.0" and len(h["evaluator_sha256"]) == 64


def test_templates_expose_pinned_sources_and_flagship_status(client):
    t = client.get(f"{V1}/templates").json()["templates"]
    assert all(len(s["revision"]) == 40 for s in t[0]["sources"])
    assert t[1]["runnable_modes"] == ["modal", "replay"] and "paid" in t[1]["label"]
    assert t[1]["search_space"] == {"num_speculative_tokens": [2, 3, 4, 5, 7]}
    assert 0 < t[1]["estimated_max_usd_per_run"] < 5 and "modal.com/pricing" in t[1]["pricing_source"]


def test_connections_never_claim_unverified_services(client):
    c = client.get(f"{V1}/connections").json()
    assert c["autolab"]["ready"] is False and c["gpu"]["ready"] is False and c["egbe"]["ready"] is False
    assert "token" not in json.dumps(c).lower() or "tokens" in json.dumps(c).lower()


@pytest.mark.parametrize("patch", [
    {"mode": "cloud"}, {"policy_arm": "yolo"}, {"goal": "x"}, {"budget": {"max_trials": 0}},
    {"budget": {"approved_spend_usd": -1}}, {"shell": "rm -rf /"}, {"template_id": "qwen3_speculative_ood"},
    {"mode": "replay"}, {"replay_of": "study_x"},
])
def test_create_rejects_invalid_input(client, patch):
    body = {"template_id": "local_speculative_qwen25", "goal": "Improve throughput", "mode": "local",
            "policy_arm": "evidence_aware", **patch}
    assert client.post(f"{V1}/studies", json=body).status_code == 422


def test_unknown_ids_are_404(client):
    assert client.get(f"{V1}/studies/nope").status_code == 404
    assert client.get(f"{V1}/experiments/nope").status_code == 404


# ---------- lifecycle ----------

def test_study_is_draft_with_empty_state_and_survives_restart(make_client):
    client, _ = make_client()
    s = new_study(client)
    assert s["state"] == "DRAFT" and s["best_valid"] is None and s["budget"]["trials_used"] == 0
    assert s["budget"]["approved_spend_usd"] == 0
    client2, _ = make_client()   # new app instance over the same database file
    again = client2.get(f"{V1}/studies/{s['id']}").json()
    assert again["id"] == s["id"] and again["state"] == "DRAFT" and again["frozen_eval_hash"] == s["frozen_eval_hash"]


def test_end_to_end_evidence_aware_loop_promotes_only_after_audit(make_client):
    adapter = FixtureAdapter(script={(K44, "dev"): [{"speedup": 1.3, "ci_low": 1.2}],
                                     (K44, "audit"): [{"speedup": 1.25}]}, default={"speedup": 0.9})
    client, _ = make_client(adapter)
    s = run(client, new_study(client, trials=6)["id"])
    assert s["state"] == "COMPLETED" and s["latest_decision"]["action"] == "promote"
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    actions = [e["action"] for e in exps]
    assert actions[0] == "run_baseline" and actions[-2:] == ["run_baseline", "audit"]
    assert s["promoted"]["split"] == "audit" and s["promoted"]["config"] == C44
    decisions = client.get(f"{V1}/studies/{s['id']}/decisions").json()
    promote = decisions[-1]
    assert set(promote["evidence_ids"]) <= {e["id"] for e in exps} and len(promote["evidence_ids"]) == 2
    assert s["budget"]["trials_used"] == len(exps) <= 6


def test_metric_greedy_promotes_unaudited_dev_winner_on_same_evidence(make_client):
    adapter = FixtureAdapter(script={(K44, "dev"): [{"speedup": 1.3}]}, default={"speedup": 0.9})
    client, _ = make_client(adapter)
    s = run(client, new_study(client, arm="metric_greedy", trials=3)["id"])
    assert s["latest_decision"]["action"] == "promote" and s["promoted"]["split"] == "dev"
    assert all(split == "dev" for _, split in adapter.calls)


def test_token_mismatch_never_becomes_best_valid_or_promoted(make_client):
    adapter = FixtureAdapter(default={"speedup": 2.5, "truncated": 10, "diverged": 2})
    client, _ = make_client(adapter)
    s = run(client, new_study(client, trials=8)["id"])
    assert s["best_valid"] is None and s["promoted"] is None
    assert s["latest_decision"]["action"] == "investigate"
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    cands = [e for e in exps if e["kind"] == "candidate"]
    assert len(cands) == 3 and all(e["state"] == "INVALID" and e["verdict"] == "reject" for e in cands)
    assert "exact_token_match" in cands[0]["validation"]["gates_failed"]
    assert s["budget"]["trials_used"] == 4  # stopped early with budget unspent


def test_failed_and_timed_out_runs_are_first_class_and_carry_no_metrics(make_client):
    adapter = FixtureAdapter(script={(K44, "dev"): [{"status": "timeout", "error": "evaluator exceeded 900s"}]},
                             default={"status": "failed", "error": "llama-server exited"})
    client, _ = make_client(adapter)
    adapter.script[("null", "dev")] = [{}]  # baseline succeeds
    s = run(client, new_study(client, arm="metric_greedy", trials=3)["id"])
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    assert [e["state"] for e in exps] == ["VALID", "TIMED_OUT", "FAILED"]
    assert all(e["metrics"] == {} for e in exps[1:]) and s["promoted"] is None
    assert s["latest_decision"]["action"] == "stop"


def test_missing_metric_is_invalid_not_zero_filled(make_client):
    adapter = FixtureAdapter(default={"speedup": 1.5, "drop_metric": "speedup_ci_low"})
    client, _ = make_client(adapter)
    adapter.script[("null", "dev")] = [{}]
    s = run(client, new_study(client, arm="fixed_spec", trials=2)["id"])
    cand = client.get(f"{V1}/studies/{s['id']}/experiments").json()[1]
    assert cand["state"] == "INVALID" and "metrics_complete_and_finite" in cand["validation"]["gates_failed"]


def test_evaluator_change_after_freeze_invalidates_runs(make_client):
    client, adapter = make_client()
    s = new_study(client, arm="target_only")
    led = client.app.state.ledger
    led.conn.execute("UPDATE studies SET frozen_eval_hash='0' WHERE id=?", (s["id"],))
    run(client, s["id"])
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    assert all("evaluator_unchanged" in e["validation"]["gates_failed"] for e in exps)


def test_fixture_evidence_is_never_relabelled(make_client):
    client, _ = make_client(FixtureAdapter(default={"speedup": 1.3, "ci_low": 1.2}))
    s = run(client, new_study(client, trials=5)["id"])
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    assert {e["evidence_class"] for e in exps} == {"fixture"}
    md = client.get(f"{V1}/studies/{s['id']}/report").text
    assert "| FIXTURE |" in md and "| LOCAL |" not in md and "| OBSERVED |" not in md


def test_start_is_idempotent_and_illegal_transitions_are_refused(make_client):
    client, adapter = make_client()
    s = new_study(client, arm="target_only")
    assert client.post(f"{V1}/studies/{s['id']}/pause").status_code == 409   # DRAFT cannot pause
    run(client, s["id"])
    n = len(adapter.calls)
    assert client.post(f"{V1}/studies/{s['id']}/start").status_code == 409   # COMPLETED cannot restart
    assert len(adapter.calls) == n == 1
    led = client.app.state.ledger
    with pytest.raises(InvalidTransition):
        led.transition(led.experiments(s["id"])[0]["id"], "RUNNING")       # VALID is terminal


def test_pause_stops_new_work_and_resume_continues(make_client):
    client, adapter = make_client(FixtureAdapter(default={"speedup": 0.95}))
    s = new_study(client, arm="metric_greedy", trials=4)
    orig = adapter.evaluate

    def pausing(exp, *a):
        if len(adapter.calls) == 1:
            client.post(f"{V1}/studies/{s['id']}/pause")
        return orig(exp, *a)

    adapter.evaluate = pausing
    paused = run(client, s["id"])
    assert paused["state"] == "PAUSED" and paused["budget"]["trials_used"] == 2
    adapter.evaluate = orig
    done = client.post(f"{V1}/studies/{s['id']}/resume").json()
    assert done["state"] == "COMPLETED" and done["budget"]["trials_used"] == 4


def test_restart_cancels_inflight_runs_and_pauses_study(make_client):
    client, _ = make_client()
    s = new_study(client)
    led = client.app.state.ledger
    led.set_study_state(s["id"], "RUNNING")
    exp = led.reserve_experiment(led.study(s["id"]), kind="baseline", action="run_baseline", config=None, split="dev",
                                 hypothesis=None, rationale="r", parent_id=None, evidence_class="fixture",
                                 estimated_cost_usd=0.0)
    led.transition(exp["id"], "RUNNING")
    client2, _ = make_client()
    assert client2.get(f"{V1}/health").json()["recovered_interrupted_runs"] == 1
    after = client2.get(f"{V1}/studies/{s['id']}").json()
    assert after["state"] == "PAUSED" and after["budget"]["trials_used"] == 0 and after["budget"]["reserved_usd"] == 0
    resumed = client2.post(f"{V1}/studies/{s['id']}/resume").json()
    assert resumed["budget"]["trials_used"] >= 1


def test_results_decisions_and_export_survive_restart_unchanged(make_client):
    client, _ = make_client(FixtureAdapter(default={"speedup": 1.3, "ci_low": 1.2}))
    s = run(client, new_study(client, trials=5)["id"])
    before = client.get(f"{V1}/studies/{s['id']}/export.json").json()
    md = client.get(f"{V1}/studies/{s['id']}/report").text
    client2, _ = make_client()
    assert client2.get(f"{V1}/studies/{s['id']}/export.json").json() == before
    assert client2.get(f"{V1}/studies/{s['id']}/report").text == md
    assert before["decisions"] and before["experiments"] and "Limitations" in md
    detail = client2.get(f"{V1}/experiments/{before['experiments'][1]['id']}").json()
    assert detail["decisions"] and detail["requests"] == []


def test_events_are_monotonic_and_replayable(make_client):
    client, _ = make_client()
    s = run(client, new_study(client, arm="fixed_spec", trials=2)["id"])
    ev = client.get(f"{V1}/studies/{s['id']}/events.json").json()
    ids = [e["id"] for e in ev]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    types = [e["type"] for e in ev]
    assert types[0] == "study.created" and "experiment.validated" in types
    assert types[-2:] == ["study.completed", "research.decision"]   # state closes first, then the decision is logged
    tail = client.get(f"{V1}/studies/{s['id']}/events.json", params={"after_event_id": ids[3]}).json()
    assert [e["id"] for e in tail] == ids[4:]


# ---------- budget ----------

def test_trial_cap_is_enforced(make_client):
    client, adapter = make_client(FixtureAdapter(default={"speedup": 0.95}))
    s = run(client, new_study(client, arm="metric_greedy", trials=3)["id"])
    assert s["budget"]["trials_used"] == 3 == len(adapter.calls) and s["state"] == "COMPLETED"


def test_wall_time_cap_stops_scheduling(make_client):
    client, adapter = make_client(FixtureAdapter(default={"speedup": 0.95, "wall": 50.0}))
    s = run(client, new_study(client, arm="metric_greedy", trials=8, budget={"max_wall_seconds": 60})["id"])
    assert len(adapter.calls) == 2 and "wall-time" in s["latest_decision"]["rationale"]


def _reserve(led, study, cost=0.0):
    return led.reserve_experiment(study, kind="baseline", action="run_baseline", config=None, split="dev",
                                  hypothesis=None, rationale="r", parent_id=None, evidence_class="fixture",
                                  estimated_cost_usd=cost)


def test_paid_reservation_beyond_approved_cap_is_refused(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    study = led.create_study(goal="g", template_id="t", mode="autolab", policy_arm="evidence_aware", max_trials=5,
                             approved_spend_usd=10.0, max_wall_seconds=600, frozen_eval_hash="h")
    a = _reserve(led, study, 6.0)
    with pytest.raises(BudgetExceeded, match="exceed"):
        _reserve(led, study, 6.0)                       # 6 reserved + 6 > 10
    with pytest.raises(BudgetExceeded, match="unknown"):
        _reserve(led, study, None)                      # unknown cost is never treated as free
    led.transition(a["id"], "RUNNING")
    led.transition(a["id"], "VALID")
    led.settle(led.experiment(a["id"]), {"rented_gpu_compute": 4.0, "llm_inference": None}, "invoice")
    led.settle(led.experiment(a["id"]), {"rented_gpu_compute": 4.0}, "retry")   # settling twice charges once
    b = led.budget(led.study(study["id"]))
    assert b["reserved_usd"] == 0 and b["observed_spend_usd"] == 4.0 and b["unknown_cost_events"] == 1
    _reserve(led, study, 6.0)                           # 4 spent + 6 == 10 fits
    with pytest.raises(BudgetExceeded):
        _reserve(led, study, 0.01)


def test_concurrent_reservations_cannot_exceed_trial_cap(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    study = led.create_study(goal="g", template_id="t", mode="local", policy_arm="random", max_trials=3,
                             approved_spend_usd=0.0, max_wall_seconds=600, frozen_eval_hash="h")
    ok, refused = [], []

    def worker():
        try:
            ok.append(_reserve(led, study))
        except BudgetExceeded:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 3 and len(refused) == 9
    assert len({e["idempotency_key"] for e in ok}) == 3


def test_remote_start_is_blocked_and_nonzero_cap_needs_approval(make_client):
    client, adapter = make_client()
    remote = new_study(client, mode="autolab", budget={"approved_spend_usd": 10})
    r = client.post(f"{V1}/studies/{remote['id']}/start")
    assert r.status_code == 409 and adapter.calls == []
    blockers = " ".join(r.json()["detail"]["blockers"])
    assert "approved" in blockers and "autolab" in blockers and "gpu" in blockers
    local = new_study(client, arm="target_only", budget={"approved_spend_usd": 5})
    assert client.post(f"{V1}/studies/{local['id']}/start").status_code == 409
    assert client.post(f"{V1}/studies/{local['id']}/approve-budget").json()["budget"]["budget_approved"] is True
    assert run(client, local["id"])["state"] == "COMPLETED"


def test_local_runs_record_zero_billed_spend_by_category(make_client):
    client, _ = make_client()
    s = run(client, new_study(client, arm="fixed_spec", trials=2)["id"])
    b = s["budget"]
    assert b["observed_spend_usd"] == 0 and b["unknown_cost_events"] == 0 and b["reserved_usd"] == 0
    assert set(b["by_category"]) == {"rented_gpu_compute", "llm_inference", "autolab_agent"}


# ---------- autolab import (ingestion contract) ----------

def payload(sid, run_id="alab-run-1", **over):
    base = {"study_id": sid, "external_run_id": run_id, "run_url": "https://app.autolab.ai/runs/alab-run-1",
            "hill_hash": "abc1234", "hill_verdict": "scored", "kind": "baseline", "config": None, "split": "dev",
            "artifact": make_artifact(None), "cost_usd": 0.0}
    return {**base, **over}


def test_import_is_observed_idempotent_and_charged_once(client):
    s = new_study(client, mode="autolab", budget={"approved_spend_usd": 5})
    client.post(f"{V1}/studies/{s['id']}/approve-budget")
    first = client.post(f"{V1}/import/autolab", json=payload(s["id"], cost_usd=1.5))
    assert first.status_code == 200 and first.json()["duplicate"] is False
    exp = first.json()["experiment"]
    assert exp["evidence_class"] == "observed" and exp["state"] == "VALID" and exp["source"]["run_id"] == "alab-run-1"
    again = client.post(f"{V1}/import/autolab", json=payload(s["id"], cost_usd=1.5)).json()
    assert again["duplicate"] is True and again["experiment"]["id"] == exp["id"]
    study = client.get(f"{V1}/studies/{s['id']}").json()
    assert study["budget"]["trials_used"] == 1 and study["budget"]["observed_spend_usd"] == 1.5
    conflict = client.post(f"{V1}/import/autolab", json=payload(s["id"], cost_usd=9.0))
    assert conflict.status_code == 409


def test_import_candidate_is_judged_but_never_dispatches_work(make_client):
    client, adapter = make_client()
    s = new_study(client, mode="autolab")
    client.post(f"{V1}/import/autolab", json=payload(s["id"]))
    cand = payload(s["id"], "alab-run-2", kind="candidate", config=C44, cost_usd=None,
                   artifact=make_artifact(C44, speedup=1.4, truncated=5))
    exp = client.post(f"{V1}/import/autolab", json=cand).json()["experiment"]
    assert exp["state"] == "INVALID" and exp["verdict"] == "reject" and adapter.calls == []
    study = client.get(f"{V1}/studies/{s['id']}").json()
    assert study["budget"]["unknown_cost_events"] == 1 and study["state"] == "DRAFT"


@pytest.mark.parametrize("over,code", [
    ({"hill_verdict": "pending"}, 409),
    ({"split": "sealed"}, 422),
    ({"artifact": None}, 422),
    ({"kind": "candidate", "config": {"lookahead_tokens": 99, "draft_p_min": 0.75}}, 422),
    ({"kind": "candidate", "config": {"cmd": "curl evil"}}, 422),
    ({"run_url": "javascript:alert(1)"}, 422),
    ({"cost_usd": 50.0}, 409),
])
def test_import_rejects_bad_payloads(client, over, code):
    s = new_study(client, mode="autolab")
    assert client.post(f"{V1}/import/autolab", json=payload(s["id"], **over)).status_code == code
    assert client.get(f"{V1}/studies/{s['id']}").json()["budget"]["trials_used"] == 0


def test_import_failed_run_and_wrong_mode(client):
    s = new_study(client, mode="autolab")
    exp = client.post(f"{V1}/import/autolab", json=payload(s["id"], hill_verdict="failed", artifact=None,
                                                           failure="OOM")).json()["experiment"]
    assert exp["state"] == "FAILED" and exp["metrics"] == {} and exp["failure"] == "OOM"
    local = new_study(client)
    assert client.post(f"{V1}/import/autolab", json=payload(local["id"], "r9")).status_code == 409


# ---------- replay ----------

def test_replay_reserves_recorded_runs_and_labels_them(make_client):
    src_adapter = FixtureAdapter(script={(K44, "dev"): [{"speedup": 1.3, "ci_low": 1.2}],
                                         (K44, "audit"): [{"speedup": 1.25}]}, default={"speedup": 0.9},
                                 evidence_class="local")  # stands in for a real local study in this test
    client, _ = make_client(src_adapter)
    src = run(client, new_study(client, trials=6)["id"])
    client.app.state.orchestrator.adapter_factory = client.app.state.orchestrator._default_adapter
    rep = new_study(client, mode="replay", trials=6, replay_of=src["id"])
    done = run(client, rep["id"])
    assert "REPLAY" in done["evidence_label"] and done["latest_decision"]["action"] == "promote"
    exps = client.get(f"{V1}/studies/{rep['id']}/experiments").json()
    assert {e["evidence_class"] for e in exps} == {"replay"}
    assert all(e["repro"]["replayed_from"]["study_id"] == src["id"] for e in exps)
    # a different arm may ask for runs that were never recorded: that fails, it is not invented
    other = run(client, new_study(client, arm="random", mode="replay", trials=4, replay_of=src["id"])["id"])
    states = [e["state"] for e in client.get(f"{V1}/studies/{other['id']}/experiments").json()]
    assert "FAILED" in states


def test_fixture_runs_cannot_be_replayed_as_evidence(make_client):
    client, _ = make_client()
    src = run(client, new_study(client, arm="fixed_spec", trials=2)["id"])
    client.app.state.orchestrator.adapter_factory = client.app.state.orchestrator._default_adapter
    rep = run(client, new_study(client, arm="fixed_spec", mode="replay", trials=2, replay_of=src["id"])["id"])
    assert rep["latest_decision"]["action"] == "stop" and rep["promoted"] is None


# ---------- Rust Möbius bridge (real binary) ----------

needs_mobius = pytest.mark.skipif(MobiusBridge.binary() is None, reason="mobius CLI not built")


@needs_mobius
def test_bridge_roundtrip_with_named_metric_and_per_arm_isolation(tmp_path):
    from conftest import EV
    bridge = MobiusBridge(tmp_path, EV.MANIFEST["search_space"])
    cfg, why = bridge.ask("s1", "tpe")
    assert set(cfg) == set(EV.MANIFEST["search_space"]) and why
    assert bridge.tell("s1", "tpe", cfg, 1.1) == "tell-0001"
    assert bridge.tell("s1", "tpe", cfg, 1.2) == "tell-0002"
    assert bridge.tell("s1", "evidence_aware", cfg, 1.0) == "tell-0001"   # separate history per arm
    assert bridge.tell("s2", "tpe", cfg, 1.0) == "tell-0001"              # and per study
    assert (tmp_path / "s1" / "tpe" / ".mobius" / "history.db").is_file()


@needs_mobius
def test_only_valid_dev_results_reach_rust_history_exactly_once(make_client):
    adapter = FixtureAdapter(script={(K44, "dev"): [{"speedup": 1.3, "truncated": 4}]}, default={"speedup": 0.95})
    client, _ = make_client(adapter)
    s = run(client, new_study(client, arm="tpe", trials=4)["id"])
    led, orch = client.app.state.ledger, client.app.state.orchestrator
    exps = led.experiments(s["id"])
    told = [e for e in exps if e["rust_told"]]
    assert told and all(e["state"] == "VALID" and e["kind"] == "candidate" for e in told)
    assert not any(e["rust_told"] for e in exps if e["state"] != "VALID" or e["kind"] == "baseline")
    orch._tell(led.study(s["id"]), told[0])                                # re-import attempt is a no-op
    next_id = orch.bridge.tell(s["id"], "tpe", told[0]["config"], 1.0)
    assert next_id == f"tell-{len(told) + 1:04d}"
    assert any("TPE" in d["rationale"] for d in led.decisions(s["id"]))


# ---------- regressions from review ----------

def test_second_instance_on_same_ledger_is_refused_and_cannot_cancel_live_runs(tmp_path):
    from mobius_lab.main import create_app
    first = create_app(db_path=tmp_path / "lab.sqlite", var_dir=tmp_path / "var", run_inline=True)
    led = first.state.ledger
    study = led.create_study(goal="g", template_id="t", mode="local", policy_arm="random", max_trials=3,
                             approved_spend_usd=0.0, max_wall_seconds=600, frozen_eval_hash="h")
    led.set_study_state(study["id"], "RUNNING")
    live = _reserve(led, led.study(study["id"]))
    led.transition(live["id"], "RUNNING")
    with pytest.raises(RuntimeError, match="already owns"):
        create_app(db_path=tmp_path / "lab.sqlite", var_dir=tmp_path / "var")
    assert led.experiment(live["id"])["state"] == "RUNNING" and led.study(study["id"])["state"] == "RUNNING"
    first.state.lock_file.close()


def test_failed_and_timed_out_runs_are_charged_wall_time(make_client):
    import time as _time
    adapter = FixtureAdapter(default={"status": "timeout", "error": "slow"})
    adapter.script[("null", "dev")] = [{}]
    client, _ = make_client(adapter)
    orig = adapter.evaluate

    def slow(exp, *a):
        if exp["config"] is not None:
            _time.sleep(0.6)
        return orig(exp, *a)

    adapter.evaluate = slow
    study = new_study(client, arm="metric_greedy", trials=8, budget={"max_wall_seconds": 60})
    led = client.app.state.ledger
    led.conn.execute("UPDATE studies SET max_wall_seconds=2 WHERE id=?", (study["id"],))
    s = run(client, study["id"])
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    timed_out = [e for e in exps if e["state"] == "TIMED_OUT"]
    assert timed_out and all(e["wall_seconds"] >= 0.5 for e in timed_out)
    assert len(exps) < 8 and "wall-time" in s["latest_decision"]["rationale"]


def test_replicate_does_not_spend_an_existing_contenders_audit_budget():
    from test_policy import C88, exp, state
    from mobius_lab.policies import decide
    contender = exp(1, C44, verdict="keep", speedup=1.2, ci_low=1.1)
    unsure = exp(2, C88, verdict="request_replicate", speedup=1.05, ci_low=0.98)
    a = decide(state([exp(0), contender, unsure], max_trials=5))   # 2 left = audit baseline + audit
    assert a.kind == "run_baseline" and a.split == "audit"
    assert decide(state([exp(0), contender, unsure], max_trials=6)).kind == "replicate"


def test_best_valid_excludes_gains_the_controller_rejected_after_replication(make_client):
    adapter = FixtureAdapter(script={(K44, "dev"): [{"speedup": 1.10, "ci_low": 0.98}, {"speedup": 0.90}]},
                             default={"speedup": 0.95})
    client, _ = make_client(adapter)
    s = run(client, new_study(client, trials=4)["id"])
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    runs44 = [e for e in exps if e["config"] == C44]
    assert [e["verdict"] for e in runs44] == ["flag_investigate", "flag_investigate"]   # verdict covers both runs
    assert s["best_valid"] is None and s["promoted"] is None


def test_pause_during_terminal_decision_cannot_double_promote(make_client):
    from mobius_lab import policies as pol
    adapter = FixtureAdapter(default={"speedup": 1.3})
    client, _ = make_client(adapter)
    study = new_study(client, arm="metric_greedy", trials=3)
    led, real = client.app.state.ledger, pol.decide

    def pausing(state):
        action = real(state)
        if action.kind == "promote" and led.study(study["id"])["state"] == "RUNNING":
            led.set_study_state(study["id"], "PAUSED")       # operator pause lands mid-decision
        return action

    pol.decide = pausing
    try:
        paused = run(client, study["id"])
    finally:
        pol.decide = real
    assert paused["state"] == "PAUSED" and paused["promoted"] is None
    done = client.post(f"{V1}/studies/{study['id']}/resume").json()
    exps = client.get(f"{V1}/studies/{study['id']}/experiments").json()
    assert done["state"] == "COMPLETED" and [e["verdict"] for e in exps].count("promoted") == 1


def test_malformed_import_is_recorded_failed_and_does_not_wedge_the_study(client):
    s = new_study(client, mode="autolab")
    bad = make_artifact(None)
    bad["engine"] = "llama"                                   # wrong type
    r = client.post(f"{V1}/import/autolab", json=payload(s["id"], artifact=bad))
    assert r.status_code == 200 and r.json()["experiment"]["state"] in ("INVALID", "FAILED")
    ok = client.post(f"{V1}/import/autolab", json=payload(s["id"], "alab-run-2"))
    assert ok.json()["experiment"]["state"] == "VALID"
    no_domains = make_artifact(C44, speedup=1.4)
    del no_domains["metrics"]["speedup_by_domain"]
    cand = client.post(f"{V1}/import/autolab", json=payload(s["id"], "alab-run-3", kind="candidate", config=C44,
                                                            artifact=no_domains)).json()["experiment"]
    assert cand["state"] == "INVALID" and "metrics_complete_and_finite" in cand["validation"]["gates_failed"]
    assert all(e["state"] != "RUNNING" for e in client.get(f"{V1}/studies/{s['id']}/experiments").json())


def test_import_requires_approval_of_a_nonzero_cap(client):
    s = new_study(client, mode="autolab", budget={"approved_spend_usd": 50})
    assert client.post(f"{V1}/import/autolab", json=payload(s["id"], cost_usd=40.0)).status_code == 409
    assert client.get(f"{V1}/studies/{s['id']}").json()["budget"]["observed_spend_usd"] == 0


@needs_mobius
def test_failed_tell_never_breaks_recording_and_is_retried(make_client):
    from mobius_lab.adapters import BridgeError
    client, _ = make_client(FixtureAdapter(default={"speedup": 0.95}))
    orch = client.app.state.orchestrator
    real, calls = orch.bridge.tell, []

    def flaky(*a):
        calls.append(a)
        if len(calls) == 1:
            raise BridgeError("mobius tell timed out")
        return real(*a)

    orch.bridge.tell = flaky
    s = run(client, new_study(client, arm="tpe", trials=4)["id"])
    exps = client.app.state.ledger.experiments(s["id"])
    cands = [e for e in exps if e["kind"] == "candidate"]
    assert s["state"] == "COMPLETED" and s["budget"]["reserved_usd"] == 0
    assert sum(e["rust_told"] for e in cands) >= len(cands) - 1 and cands[0]["rust_told"] == 1   # first one retried


# ---------- paid Modal mode (no network: the remote call is a fixture) ----------

class PaidFixture(FixtureAdapter):
    """Stands in for the Modal adapter: reserves an estimate, reports a cost and a source per run."""

    evidence_class = "observed"
    estimated_cost_usd = 1.0

    def evaluate(self, exp, reference_path, out_path, timeout):
        from dataclasses import replace
        result = super().evaluate(exp, reference_path, out_path, timeout)
        return replace(result, cost={"rented_gpu_compute": 0.6}, cost_source="estimate: fixture",
                       source={"provider": "modal", "call_id": f"fc-{len(self.calls)}", "gpu": "A100-80GB"})


def flagship(client, cap=25.0, trials=8, arm="evidence_aware"):
    body = {"template_id": "qwen3_speculative_ood", "goal": "Improve Qwen3-8B throughput, keep outputs identical",
            "mode": "modal", "policy_arm": arm, "budget": {"max_trials": trials, "approved_spend_usd": cap}}
    r = client.post(f"{V1}/studies", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_modal_mode_rules_at_creation(client):
    base = {"goal": "Improve throughput", "policy_arm": "evidence_aware"}
    zero = client.post(f"{V1}/studies", json={**base, "template_id": "qwen3_speculative_ood", "mode": "modal"})
    assert zero.status_code == 422 and "cap" in zero.text                      # paid mode needs a cap above $0
    wrong = client.post(f"{V1}/studies", json={**base, "template_id": "local_speculative_qwen25", "mode": "modal",
                                               "budget": {"approved_spend_usd": 5}})
    assert wrong.status_code == 422
    assert client.post(f"{V1}/studies", json={**base, "template_id": "qwen3_speculative_ood",
                                              "mode": "local"}).status_code == 422


def test_modal_start_needs_approval_and_a_reachable_app(make_client):
    client, _ = make_client()
    client.app.state.orchestrator.adapter_factory = client.app.state.orchestrator._default_adapter
    s = flagship(client)
    r = client.post(f"{V1}/studies/{s['id']}/start")
    blockers = " ".join(r.json()["detail"]["blockers"])
    assert r.status_code == 409 and "approved" in blockers and "modal:" in blockers
    tiny = flagship(client, cap=0.5)
    assert "above the $0.50 cap" in " ".join(client.post(f"{V1}/studies/{tiny['id']}/start").json()["detail"]["blockers"])


def test_modal_runs_use_flagship_space_record_source_and_charge_every_run(make_client):
    k3 = policies.key({"num_speculative_tokens": 3})
    adapter = PaidFixture(script={(k3, "dev"): [{"status": "failed", "error": "preempted"}]}, default={"speedup": 0.95},
                          evidence_class="observed")
    client, _ = make_client(adapter)
    s = flagship(client, trials=4, arm="metric_greedy")
    client.post(f"{V1}/studies/{s['id']}/approve-budget")
    done = run(client, s["id"])
    exps = client.get(f"{V1}/studies/{s['id']}/experiments").json()
    assert exps[1]["config"] == {"num_speculative_tokens": 3} and exps[1]["state"] == "FAILED"
    assert all(set(e["config"]) == {"num_speculative_tokens"} for e in exps[1:])
    assert {e["evidence_class"] for e in exps} == {"observed"} and exps[0]["source"]["call_id"] == "fc-1"
    b = done["budget"]
    assert b["observed_spend_usd"] == pytest.approx(0.6 * 4) and b["reserved_usd"] == 0   # the failed run is charged too
    assert "OBSERVED" in done["evidence_label"] and "estimate" in done["evidence_label"]


def test_modal_spend_cap_stops_the_study_before_exceeding_it(make_client):
    client, adapter = make_client(PaidFixture(default={"speedup": 0.95}))
    s = flagship(client, cap=2.0, trials=8, arm="metric_greedy")
    client.post(f"{V1}/studies/{s['id']}/approve-budget")
    done = run(client, s["id"])
    # each run reserves $1.00 then settles at $0.60: 0.6+1, 1.2+1 > 2 -> the third run is refused
    assert len(adapter.calls) == 2 and done["budget"]["observed_spend_usd"] == pytest.approx(1.2)
    assert done["budget"]["observed_spend_usd"] <= 2.0 and "exceed" in done["latest_decision"]["rationale"]
