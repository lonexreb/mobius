from pathlib import Path
from hills import run_evaluator

HILL = Path(__file__).resolve().parents[1]


def test_reference_passes():
    result = run_evaluator(HILL, HILL / "examples/baseline")
    assert result["passed"]
    assert result["metrics"][0]["value"] == 1


def test_observed_draft_fails_exactness():
    result = run_evaluator(HILL, HILL / "examples/draft")
    assert not result["passed"]
    assert result["details"]["reason"] == "token mismatch"


def test_final_declines_to_score_the_public_dev_pilot():
    result = run_evaluator(HILL, HILL / "examples/baseline", final=True)
    assert not result["passed"]
