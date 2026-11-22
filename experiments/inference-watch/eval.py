"""Autolab hill for replaying paired inference measurements on one machine."""
import json
from pathlib import Path

PILOT = Path(__file__).with_name("pilot.json")


def assess(pilot, workload, configuration):
    if workload not in pilot:
        raise ValueError("Unknown workload")
    if configuration == "target-only":
        return True, 1.0, []
    if configuration not in pilot[workload]:
        raise ValueError("Configuration has no matched recordings")
    records = pilot[workload][configuration]
    passed = all(row["matched"] == row["prompts"] for row in records)
    return passed, min(row["ratio"] for row in records), records


def eval(submission: Path, *, final: bool = False) -> dict:
    result = {"passed": False, "metrics": [], "config": [], "details": {}}
    if final:
        result["details"]["error"] = "The public pilot contains development recordings only."
        return result
    try:
        choice = json.loads((submission / "solution.json").read_text())
        workload, configuration = choice["workload"], choice["configuration"]
        passed, ratio, records = assess(json.loads(PILOT.read_text()), workload, configuration)
    except (OSError, TypeError, ValueError, KeyError) as error:
        result["details"]["error"] = str(error)
        return result
    result["passed"] = passed
    result["config"] = [{"name": "workload", "value": workload, "primary": True}]
    result["details"] = {"evidence": "recorded local pilot", "records": records,
                         "minimum_paired_ratio": ratio,
                         "reason": "exact target reference" if not records else
                                   ("all outputs match" if passed else "token mismatch")}
    if passed:
        result["metrics"] = [{"name": "paired_speedup", "value": ratio, "direction": "max"}]
    return result
