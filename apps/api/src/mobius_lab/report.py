"""Factual Markdown report and machine-readable export, built only from ledger records."""

from __future__ import annotations

from .persistence import Ledger
from .policies import label

EVIDENCE_LABELS = {
    "local": "LOCAL: measured on this machine with llama.cpp. Not comparable to the Qwen3-8B / EAGLE-3 CUDA flagship.",
    "observed": "OBSERVED: measured on external compute by the fixed evaluator, with a source run ID.",
    "replay": "REPLAY: recorded runs re-served from the ledger. Nothing was re-measured.",
    "fixture": "FIXTURE: test data. Not a measurement; must not support any performance claim.",
}
MODE_LABELS = {"local": EVIDENCE_LABELS["local"], "replay": EVIDENCE_LABELS["replay"],
               "modal": "OBSERVED: Qwen3-8B + EAGLE-3 on vLLM, measured on a rented NVIDIA GPU via Modal. "
                        "Paid compute; cost shown is an estimate until reconciled with the Modal bill.",
               "autolab": "AUTOLAB: only imported, externally scored runs appear. No connection is verified yet."}


def _n(x: float | None, fmt: str = "{:.3f}") -> str:
    return "not available" if x is None else fmt.format(x)


def export(ledger: Ledger, study: dict) -> dict:
    exps = ledger.experiments(study["id"], include_cancelled=True)
    return {
        "schema_version": 1,
        "study": study,
        "evidence_label": MODE_LABELS[study["mode"]],
        "budget": ledger.budget(study),
        "experiments": [{k: v for k, v in e.items() if k != "artifact_path"} for e in exps],
        "decisions": ledger.decisions(study["id"]),
        "limitations": LIMITATIONS,
    }


LIMITATIONS = [
    "Pilot scale: 8 prompts per domain per split. No statistical significance is claimed.",
    "Bootstrap CIs cover prompt-to-prompt variation within one run, not run-to-run drift.",
    "The sealed split has not been evaluated and is not reachable through the API.",
    "Local results use Qwen2.5-7B with a Qwen2.5-1.5B draft on llama.cpp/Metal. They say nothing about "
    "Qwen3-8B with EAGLE-3 on vLLM/CUDA, which has not been run.",
    "Decision rules are transparent V0 heuristics, not a calibrated value-of-information policy.",
    "One policy arm per study; no matched multi-arm comparison has been run, so no claim that one policy is better.",
]


def markdown(ledger: Ledger, study: dict) -> str:
    data = export(ledger, study)
    exps, decisions, b = data["experiments"], data["decisions"], data["budget"]
    promoted = next((e for e in exps if e["verdict"] == "promoted"), None)
    final = decisions[-1] if decisions else None
    lines = [
        f"# Möbius Lab research report: {study['id']}",
        "",
        f"**Evidence class:** {data['evidence_label']}",
        "",
        f"- Goal: {study['goal']}",
        f"- Policy arm: `{study['policy_arm']}`  |  Mode: `{study['mode']}`  |  State: `{study['state']}`",
        f"- Evaluator hash (frozen at creation): `{study['frozen_eval_hash']}`",
        f"- Created: {study['created_at']}",
        "",
        "## Outcome",
        "",
    ]
    if promoted:
        m = promoted["metrics"]
        lines.append(f"Promoted `{label(promoted['config'])}` ({promoted['id']}): speedup {_n(m.get('speedup_vs_target'))}x "
                     f"on `{promoted['split']}`, 95% CI [{_n(m.get('speedup_ci_low'))}, {_n(m.get('speedup_ci_high'))}].")
    else:
        lines.append("**No candidate was promoted.**")
    if final:
        done = study["state"] in ("COMPLETED", "CANCELLED")
        lines += ["", f"{'Final' if done else 'Latest (study not finished)'} decision: "
                      f"**{final['action'].upper()}**. {final['rationale']}"]
        if final["uncertainty"]:
            lines.append(f"Uncertainty: {final['uncertainty']}")
    lines += ["", "## Experiments", "",
              "| # | Run | Action | Split | Config | State | Verdict | Speedup (95% CI) | Tokens matched | tok/s | Evidence |",
              "|---|-----|--------|-------|--------|-------|---------|------------------|----------------|-------|----------|"]
    for e in exps:
        m, v = e["metrics"], e["validation"]
        speed = (f"{_n(m.get('speedup_vs_target'))}x [{_n(m.get('speedup_ci_low'))}, {_n(m.get('speedup_ci_high'))}]"
                 if m.get("speedup_vs_target") is not None else "n/a")
        match = f"{v.get('compared', 0) - v.get('truncated_prefix', 0) - v.get('diverged', 0)}/{v['compared']}" \
            if v.get("compared") else "n/a"
        lines.append(f"| {e['seq']} | {e['id']} | {e['action']} | {e['split']} | `{label(e['config'])}` | "
                     f"{e['state']} | {e['verdict'] or '-'} | {speed} | {match} | "
                     f"{_n(m.get('throughput_tokens_per_sec'), '{:.1f}')} | {e['evidence_class'].upper()} |")
    lines += ["", "## Decision trace", ""]
    for d in decisions:
        cites = f" (evidence: {', '.join(d['evidence_ids'])})" if d["evidence_ids"] else ""
        lines.append(f"{d['id']}. **{d['action']}**" + (f" on {d['experiment_id']}" if d["experiment_id"] else "")
                     + f": {d['rationale']}{cites}")
    lines += ["", "## Budget", "",
              f"- Trials: {b['trials_used']} / {b['max_trials']}; wall time {b['wall_seconds_used']}s / {b['max_wall_seconds']}s",
              f"- Approved spend: ${b['approved_spend_usd']:.2f}; reserved ${b['reserved_usd']:.2f}; "
              f"observed ${b['observed_spend_usd']:.2f}; cost events with unknown amount: {b['unknown_cost_events']}"]
    for cat, c in sorted(b["by_category"].items()):
        lines.append(f"  - {cat}: ${c['actual_usd']:.2f} observed" +
                     (f", {c['unknown_events']} unknown" if c["unknown_events"] else ""))
    lines.append("- Local electricity and hardware wear are not metered (unknown, not zero).")
    ref = next((e for e in exps if e["repro"]), None)
    if ref:
        r = ref["repro"]
        hw = r.get("hardware") or {}
        lines += ["", "## Reproducibility", "",
                  f"- Code: `{r.get('git_sha')}`" + (" (uncommitted changes present)" if r.get("git_dirty") else ""),
                  f"- Engine: {r.get('engine')} ({r.get('engine_version')})",
                  f"- Target model: {r.get('target_model')} sha256 `{r.get('target_model_sha256')}`",
                  f"- Hardware: {hw.get('cpu')}, {_n(hw.get('memory_gb'), '{:.0f}')} GB, {hw.get('os')}",
                  f"- Generation: `{r.get('generation')}`",
                  "- Per-run dataset split hashes, draft model hashes and engine argv are in the JSON export."]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    return "\n".join(lines)
