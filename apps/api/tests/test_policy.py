"""Controller decisions from fixed observed inputs. Pure: no I/O, no adapter."""

from conftest import C44, EV, make_artifact
from mobius_lab import policies
from mobius_lab.policies import decide, key

SPACE = EV.MANIFEST["search_space"]
C88 = {"draft_p_min": 0.75, "lookahead_tokens": 8}


def exp(i, config=None, split="dev", state="VALID", verdict=None, gates_failed=(), **outcome):
    art = make_artifact(config, split, **outcome)
    return {"id": f"run_{i}", "seq": i, "kind": "candidate" if config else "baseline", "config": config,
            "split": split, "state": state, "verdict": verdict, "metrics": art["metrics"] if state == "VALID" or config else {},
            "validation": {"gates_failed": list(gates_failed)}, "failure": "boom" if state == "FAILED" else None}


def state(exps, arm="evidence_aware", max_trials=8, ask=None):
    return {"study_id": "s", "arm": arm, "max_trials": max_trials, "seed": 2026, "search_space": SPACE,
            "fixed_spec": C44, "experiments": exps, "tpe_ask": ask or (lambda: (C88, "tpe says")),
            "dev_candidates": [e for e in exps if e["kind"] == "candidate" and e["split"] == "dev"]}


def test_baseline_first_and_deterministic():
    a, b = decide(state([])), decide(state([]))
    assert a == b and a.kind == "run_baseline" and a.split == "dev"


def test_baseline_failing_twice_stops():
    exps = [exp(0, state="FAILED"), exp(1, state="INVALID", gates_failed=["evaluator_unchanged"])]
    assert decide(state(exps)).kind == "stop"


def test_first_proposal_is_preregistered_fixed_spec():
    a = decide(state([exp(0)]))
    assert a.kind == "propose_new" and a.config == C44


def test_invalid_candidate_rejected_despite_high_throughput():
    exps = [exp(0), exp(1, C44, state="INVALID", gates_failed=["exact_token_match"], speedup=3.0)]
    a = decide(state(exps))
    assert a.kind == "reject" and "exact_token_match" in a.rationale and a.target_experiment_id == "run_1"
    assert decide(state(exps, arm="metric_greedy")).kind == "reject"  # validity is mandatory in every arm


def test_failed_run_rejected_without_metrics():
    a = decide(state([exp(0), exp(1, C44, state="FAILED")]))
    assert a.kind == "reject" and "boom" in a.rationale


def test_borderline_gain_requests_replication_then_replicates():
    borderline = exp(1, C44, speedup=1.05, ci_low=0.98)
    a = decide(state([exp(0), borderline]))
    assert a.kind == "request_replicate"
    borderline["verdict"] = "request_replicate"
    b = decide(state([exp(0), borderline]))
    assert b.kind == "replicate" and b.config == C44 and b.parent_id == "run_1"
    assert decide(state([exp(0), {**borderline, "verdict": None}], arm="metric_greedy")).kind == "keep"


def test_replicate_outcomes():
    first = exp(1, C44, verdict="request_replicate", speedup=1.05, ci_low=0.98)
    held = decide(state([exp(0), first, exp(2, C44, speedup=1.06)]))
    assert held.kind == "keep" and set(held.evidence_ids) == {"run_1", "run_2"}
    faded = decide(state([exp(0), first, exp(2, C44, speedup=1.01)]))
    assert faded.kind == "reject" and "replication" in faded.rationale
    flipped = decide(state([exp(0), first, exp(2, C44, speedup=0.9)]))
    assert flipped.kind == "flag_investigate"


def test_clear_gain_is_kept_then_audited_before_promotion():
    clear = exp(1, C44, speedup=1.3, ci_low=1.2)
    assert decide(state([exp(0), clear])).kind == "keep"
    clear["verdict"] = "keep"
    s = state([exp(0), clear], max_trials=4)          # 2 left = exactly the audit reserve
    a = decide(s)
    assert a.kind == "run_baseline" and a.split == "audit"
    b = decide(state([exp(0), clear, exp(2, split="audit")], max_trials=4))
    assert b.kind == "audit" and b.config == C44 and b.split == "audit"


def test_audit_pass_promotes_and_audit_fail_rejects():
    clear = exp(1, C44, verdict="keep", speedup=1.3, ci_low=1.2)
    base = [exp(0), clear, exp(2, split="audit")]
    ok = exp(3, C44, split="audit", speedup=1.2)
    assert decide(state(base + [ok])).kind == "keep"
    ok["verdict"] = "keep"
    p = decide(state(base + [ok]))
    assert p.kind == "promote" and p.target_experiment_id == "run_3" and set(p.evidence_ids) == {"run_1", "run_3"}
    bad = exp(3, C44, split="audit", speedup=1.1, domains={"chat": 1.4, "code": 0.8, "math": 1.2})
    r = decide(state(base + [bad]))
    assert r.kind == "reject" and "code" in r.rationale


def test_evidence_aware_holds_back_audit_budget_but_greedy_spends_it():
    clear = exp(1, C44, verdict="keep", speedup=1.3, ci_low=1.2)
    assert decide(state([exp(0), clear], max_trials=4)).kind == "run_baseline"      # audit, not a new proposal
    assert decide(state([exp(0), clear], max_trials=4, arm="metric_greedy")).kind == "propose_new"
    assert decide(state([exp(0), clear], max_trials=5)).kind == "propose_new"       # 3 left: one spare


def test_greedy_promotes_on_dev_score_alone():
    a, b = exp(1, C44, verdict="keep", speedup=1.1), exp(2, C88, verdict="keep", speedup=1.3)
    p = decide(state([exp(0), a, b], arm="metric_greedy", max_trials=3))
    assert p.kind == "promote" and p.target_experiment_id == "run_2" and "Not replicated" in p.uncertainty


def test_systematic_gate_failure_triggers_investigate_only_for_evidence_aware():
    cfgs = policies.grid(SPACE)[:3]
    exps = [exp(0)] + [exp(i + 1, c, state="INVALID", verdict="reject", gates_failed=["exact_token_match"])
                       for i, c in enumerate(cfgs)]
    a = decide(state(exps))
    assert a.kind == "investigate" and "exact_token_match" in a.rationale and len(a.evidence_ids) == 3
    assert decide(state(exps, arm="metric_greedy")).kind == "propose_new"


def test_futility_stop():
    cfgs = policies.grid(SPACE)[:3]
    exps = [exp(0)] + [exp(i + 1, c, verdict="reject", speedup=0.8) for i, c in enumerate(cfgs)]
    a = decide(state(exps))
    assert a.kind == "stop" and "unspent" in a.rationale


def test_neighbour_proposer_moves_one_step_from_best_and_never_repeats():
    exps = [exp(0), exp(1, C44, verdict="reject", speedup=0.9)]
    a = decide(state(exps))
    assert a.kind == "propose_new" and a.parent_id == "run_1" and key(a.config) != key(C44)
    steps = sum(abs(SPACE[k].index(a.config[k]) - SPACE[k].index(C44[k])) for k in SPACE)
    assert steps == 1


def test_random_arm_is_seeded_and_tpe_substitutes_duplicates():
    r1, r2 = decide(state([exp(0)], arm="random")), decide(state([exp(0)], arm="random"))
    assert r1.config == r2.config and r1.config in policies.grid(SPACE)
    dup = decide(state([exp(0), exp(1, C88, verdict="reject", speedup=0.9)], arm="tpe"))
    assert dup.kind == "propose_new" and key(dup.config) != key(C88) and "substituted" in dup.rationale


def test_control_arms():
    assert decide(state([exp(0)], arm="target_only")).kind == "stop"
    assert decide(state([exp(0)], arm="fixed_spec")).config == C44
    assert decide(state([exp(0), exp(1, C44, verdict="reject", speedup=0.9)], arm="fixed_spec")).kind == "stop"


def test_search_exhaustion_terminates():
    exps = [exp(0)] + [exp(i + 1, c, verdict="reject", speedup=0.95) for i, c in enumerate(policies.grid(SPACE))]
    assert decide(state(exps, max_trials=24, arm="metric_greedy")).kind == "stop"
    assert decide(state(exps, max_trials=24)).kind == "stop"
