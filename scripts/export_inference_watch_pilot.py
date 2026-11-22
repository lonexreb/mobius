"""Rebuild the portable benchmark catalog from the two original local studies."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

STUDIES = ("study_3919d763ab", "study_68dddde577")
SETTINGS = {(4, 0.75), (4, 0.5), (2, 0.5)}
WORKLOADS = ("code", "math", "chat")


def build_catalog(directory):
    catalog = {workload: {} for workload in WORKLOADS}
    for study in STUDIES:
        records = [
            (path, json.loads(path.read_text()))
            for path in sorted((directory / study).glob("*.json"))
        ]
        (baseline,) = [
            record for _, record in records if record["kind"] == "target_only"
        ]
        reference = {row["prompt_id"]: row["token_ids"] for row in baseline["requests"]}
        for path, candidate in records:
            if candidate["kind"] != "candidate":
                continue
            config = candidate["config"]
            lookahead, probability = config["lookahead_tokens"], config["draft_p_min"]
            if (lookahead, probability) not in SETTINGS:
                continue
            control = {row["prompt_id"]: row for row in candidate["control_requests"]}
            if set(control) != set(reference) or any(
                row["token_ids"] != reference[pid] for pid, row in control.items()
            ):
                raise ValueError(f"Unmatched target reference for {path.name}")
            key = f"d{lookahead}-p{round(100 * probability)}"
            for workload in WORKLOADS:
                rows = [
                    row for row in candidate["requests"] if row["domain"] == workload
                ]
                if len(rows) != 8 or len({row["prompt_id"] for row in rows}) != 8:
                    raise ValueError(f"Incomplete {workload} sample for {path.name}")
                speedup = statistics.median(
                    row["decode_tps"] for row in rows
                ) / statistics.median(
                    control[row["prompt_id"]]["decode_tps"] for row in rows
                )
                if not math.isclose(
                    speedup, candidate["metrics"]["speedup_by_domain"][workload]
                ):
                    raise ValueError(f"Stored speedup differs for {path.name}")
                catalog[workload].setdefault(key, []).append(
                    {
                        "study": study,
                        "run": path.stem,
                        "artifact_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "evaluator_sha256": candidate["evaluator_sha256"],
                        "split_sha256": candidate["dataset_split_sha256"],
                        "matched": sum(
                            row["token_ids"] == reference[row["prompt_id"]]
                            for row in rows
                        ),
                        "prompts": len(rows),
                        "ratio": round(speedup, 6),
                    }
                )
    for workload, configs in catalog.items():
        if len(configs) != len(SETTINGS) or any(
            [row["study"] for row in rows] != list(STUDIES) for rows in configs.values()
        ):
            raise ValueError(f"Incomplete replication for {workload}")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    result = build_catalog(args.artifact_directory)
    args.output_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
