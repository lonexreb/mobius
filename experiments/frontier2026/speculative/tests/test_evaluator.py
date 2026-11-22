"""Science tests for the fixed evaluator. No model needed: FakeEngine is a test fixture."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import evaluator as ev  # noqa: E402

REGISTRY = json.loads(ev.REGISTRY_PATH.read_text())
needs_prompts = pytest.mark.skipif(not (ev.HERE / "data" / "prompts_dev.jsonl").exists(),
                                   reason="prompt cache not fetched")


class FakeEngine:
    """FIXTURE engine: deterministic tokens, configurable speed. Never used outside tests."""

    def __init__(self, tps=50.0, spec_tps=60.0, corrupt=False, nan=False):
        self.tps, self.spec_tps, self.corrupt, self.nan = tps, spec_tps, corrupt, nan

    def start(self, config):
        self.config = config

    def complete(self, prompt, speculative):
        toks = [len(prompt) % 97, 7, 9]
        if speculative and self.corrupt and "Say hello" not in prompt:
            toks = toks[:2] if len(prompt) % 2 else [toks[0], 8, 9]
        tps = float("nan") if self.nan else (self.spec_tps if speculative else self.tps)
        return {"token_ids": toks, "decode_tps": tps, "output_tokens": len(toks), "prompt_tokens": 5, "ttft_ms": 10.0,
                "stop_type": "limit", "draft_n": 10 if speculative else None, "draft_n_accepted": 8 if speculative else None}

    def describe(self):
        return {"name": "fixture", "target_model_sha256": "t", "draft_model_sha256": "d"}

    def stop(self):
        pass


CFG = {"lookahead_tokens": 4, "draft_p_min": 0.75}


def test_no_prompt_overlap_and_stable_split_hashes():
    hashes = [p["sha256"] for p in REGISTRY["prompts"]]
    assert len(hashes) == len(set(hashes)) == 72
    for split, meta in REGISTRY["splits"].items():
        prompts = [p for p in REGISTRY["prompts"] if p["split"] == split]
        expected = ev.hashlib.sha256("\n".join(sorted(f"{p['prompt_id']}:{p['sha256']}" for p in prompts)).encode())
        assert meta["sha256"] == expected.hexdigest()
        assert {p["domain"] for p in prompts} == {"code", "math", "chat"}


def test_all_source_revisions_pinned():
    assert all(len(s["revision"]) == 40 for s in REGISTRY["sources"])


@pytest.mark.parametrize("bad", [
    {"lookahead_tokens": 3, "draft_p_min": 0.75},          # value outside pinned space
    {"lookahead_tokens": 4},                                # missing knob
    {"lookahead_tokens": 4, "draft_p_min": 0.75, "cmd": "rm -rf /"},  # extra key
    {"lookahead_tokens": True, "draft_p_min": 0.75},
    "lookahead=4",
])
def test_config_allowlist_rejects(bad):
    with pytest.raises(ev.EvalError):
        ev.validate_config(bad)


def test_sealed_split_refused_without_explicit_unseal():
    with pytest.raises(ev.EvalError, match="unseal"):
        ev.load_prompts("sealed")


def test_token_mismatch_is_detected():
    ref = [{"prompt_id": "a", "token_ids": [1, 2, 3]}, {"prompt_id": "b", "token_ids": [4]}]
    cand = [{"prompt_id": "a", "token_ids": [1, 2, 3]}, {"prompt_id": "b", "token_ids": [4, 5]}]
    out = ev.compare_tokens(ref, cand)
    assert out["exact_token_match"] is False
    assert out["diverged"] == [{"prompt_id": "b", "first_divergent_index": 1}]
    short = ev.compare_tokens(ref, [{"prompt_id": "a", "token_ids": [1, 2]}, ref[1]])
    assert short["exact_token_match"] is False and short["truncated_prefix_prompt_ids"] == ["a"]
    assert ev.compare_tokens(ref, cand[:1])["exact_token_match"] is False  # incomplete generation set
    assert ev.compare_tokens(ref, ref)["exact_token_match"] is True


def test_speedup_is_deterministic_and_ci_brackets_point():
    ref = [{"prompt_id": f"{d}{i}", "domain": d, "decode_tps": 50 + i} for d in "xyz" for i in range(8)]
    cand = [{**r, "decode_tps": r["decode_tps"] * (1.2 if r["domain"] == "x" else 1.0)} for r in ref]
    a, b = ev.speedup_stats(ref, cand), ev.speedup_stats(ref, cand)
    assert a == b
    assert a["speedup_by_domain"]["x"] == pytest.approx(1.2) and a["speedup_by_domain"]["y"] == pytest.approx(1.0)
    assert a["speedup_ci_low"] <= a["speedup_vs_target"] <= a["speedup_ci_high"]
    assert a["speedup_vs_target"] == pytest.approx(1.2 ** (1 / 3))


@needs_prompts
def test_run_scores_candidate_against_reference():
    base = ev.run(FakeEngine(50), None, "dev", None)
    cand = ev.run(FakeEngine(50, 60), CFG, "dev", base)
    assert base["metrics"]["draft_acceptance_rate"] is None  # not applicable is null, not zero
    assert cand["correctness"]["exact_token_match"] is True
    assert cand["metrics"]["speedup_vs_target"] == pytest.approx(1.2)
    assert cand["metrics"]["draft_acceptance_rate"] == pytest.approx(0.8)
    assert cand["correctness"]["control_reproduces_reference"] is True
    assert len(cand["control_requests"]) == len(cand["requests"]) == 24
    assert cand["evaluator_sha256"] == ev.evaluator_sha256()


@needs_prompts
def test_corrupted_output_fails_correctness_but_still_reports():
    base = ev.run(FakeEngine(50), None, "dev", None)
    cand = ev.run(FakeEngine(50, 500, corrupt=True), CFG, "dev", base)
    assert cand["correctness"]["exact_token_match"] is False
    assert cand["correctness"]["matched"] == 0
    assert cand["correctness"]["truncated_prefix_prompt_ids"] and cand["correctness"]["diverged"]


@needs_prompts
def test_non_finite_metrics_raise_instead_of_zero_fill():
    with pytest.raises(ev.EvalError, match="non-finite"):
        ev.run(FakeEngine(nan=True), None, "dev", None)


@needs_prompts
def test_candidate_needs_matched_reference():
    base = ev.run(FakeEngine(), None, "dev", None)
    with pytest.raises(ev.EvalError, match="requires"):
        ev.run(FakeEngine(), CFG, "dev", None)
    with pytest.raises(ev.EvalError, match="dataset_split_sha256"):
        ev.run(FakeEngine(), CFG, "audit", base)
    with pytest.raises(ev.EvalError, match="evaluator_sha256"):
        ev.run(FakeEngine(), CFG, "dev", {**base, "evaluator_sha256": "tampered"})
