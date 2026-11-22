"""Research controller: a pure function from observed study state to the next action.

Transparent V0 heuristics, not a value-of-information estimator. Every arm enforces the
validity gates; arms differ only in how they propose and in how much evidence they
demand before promoting. The state passed in contains dev and audit runs only: the
sealed split never reaches the API, so no policy can read it.

Arms
  target_only     baseline, then stop
  fixed_spec      baseline, the pre-registered fixed config, promote on dev score
  random          seeded random configs, promote best dev score
  tpe             configs from the Rust Möbius `ask --strategy tpe`, promote best dev score
  metric_greedy   neighbourhood proposer, promote best dev score
  evidence_aware  same proposer; replicates borderline gains, audits before promoting,
                  stops early on systematic gate failure or futility
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CONTROLLER_VERSION = "evidence-v0.1"
MIN_EFFECT = 0.02            # a gain must clear 1 + MIN_EFFECT to count
AUDIT_DOMAIN_FLOOR = 0.95    # no domain may regress below this on the audit split
SYSTEMATIC_INVALID = 3       # distinct configs failing validity, with none valid -> investigate
FUTILITY_CONFIGS = 3         # distinct valid configs tried ...
FUTILITY_SPEEDUP = 0.90      # ... all below this -> stop
MAX_BASELINE_ATTEMPTS = 2

RUN_KINDS = {"run_baseline", "propose_new", "replicate", "audit"}
VERDICT_KINDS = {"keep", "reject", "request_replicate", "flag_investigate"}
TERMINAL_KINDS = {"promote", "stop", "investigate"}
ARMS = ("target_only", "fixed_spec", "random", "tpe", "metric_greedy", "evidence_aware")


@dataclass(frozen=True)
class Action:
    kind: str
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    target_experiment_id: str | None = None
    config: dict | None = None
    split: str = "dev"
    hypothesis: str | None = None
    parent_id: str | None = None
    uncertainty: str | None = None


def key(config: dict | None) -> str:
    return json.dumps(config, sort_keys=True)


def grid(space: dict[str, list]) -> list[dict]:
    knobs = sorted(space)
    out: list[dict] = [{}]
    for k in knobs:
        out = [{**c, k: v} for c in out for v in space[k]]
    return out


def _speedup(e: dict) -> float | None:
    return e["metrics"].get("speedup_vs_target")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}x"


def _failed_gates(e: dict) -> str:
    return ", ".join(e["validation"].get("gates_failed", [])) or (e.get("failure") or "unknown")


# ---------- proposers ----------

def _neighbour_proposal(state: dict, tried: set[str]) -> Action | None:
    space, cands = state["search_space"], state["dev_candidates"]
    untried = [c for c in grid(space) if key(c) not in tried]
    if not untried:
        return None
    if not cands:
        cfg = state["fixed_spec"] if key(state["fixed_spec"]) not in tried else untried[0]
        return Action("propose_new", "Start from the pre-registered fixed configuration.", config=cfg,
                      hypothesis=f"The pre-registered config {cfg} decodes faster than target-only while "
                                 "reproducing its tokens exactly.")
    valid = [e for e in cands if e["state"] == "VALID"]
    anchor = max(valid, key=_speedup) if valid else cands[-1]

    def dist(c: dict) -> int:
        return sum(abs(space[k].index(c[k]) - space[k].index(anchor["config"][k])) for k in space)

    cfg = min(untried, key=dist)  # ties resolved by fixed grid order
    changed = {k: (anchor["config"][k], cfg[k]) for k in sorted(space) if cfg[k] != anchor["config"][k]}
    delta = "; ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items())
    basis = (f"best valid dev result so far ({_fmt(_speedup(anchor))})" if valid
             else f"the last run, which failed validity ({_failed_gates(anchor)})")
    return Action("propose_new", f"Nearest untried config to {basis}.", config=cfg, parent_id=anchor["id"],
                  evidence_ids=(anchor["id"],),
                  hypothesis=f"Changing {delta} improves on {anchor['id']}: valid tokens and higher paired throughput.",
                  uncertainty="Direction of the effect is not known in advance; this is a local search step.")


def _random_proposal(state: dict, tried: set[str]) -> Action | None:
    order = grid(state["search_space"])
    random.Random(f"{state['seed']}:{state['study_id']}").shuffle(order)
    cfg = next((c for c in order if key(c) not in tried), None)
    return cfg and Action("propose_new", "Seeded uniform sample from the pinned search space.", config=cfg,
                          hypothesis=f"Control arm: random config {cfg}.")


def _fixed_proposal(state: dict, tried: set[str]) -> Action | None:
    cfg = state["fixed_spec"]
    if key(cfg) in tried:
        return None
    return Action("propose_new", "Pre-registered fixed speculative config (control arm).", config=cfg,
                  hypothesis=f"The fixed config {cfg} is faster than target-only and lossless.")


def _tpe_proposal(state: dict, tried: set[str]) -> Action | None:
    untried = [c for c in grid(state["search_space"]) if key(c) not in tried]
    if not untried:
        return None
    cfg, why = state["tpe_ask"]()
    note = f"Möbius Rust TPE: {why}"
    if key(cfg) in tried or cfg not in grid(state["search_space"]):
        note += f" It suggested {cfg}, already evaluated or outside the grid; substituted the next untried config."
        cfg = untried[0]
    return Action("propose_new", note, config=cfg, hypothesis=f"TPE arm: config {cfg} suggested from validated history.")


PROPOSERS: dict[str, Callable[[dict, set[str]], Action | None]] = {
    "fixed_spec": _fixed_proposal, "random": _random_proposal, "tpe": _tpe_proposal,
    "metric_greedy": _neighbour_proposal, "evidence_aware": _neighbour_proposal,
}


# ---------- judging a finished run ----------

def _judge(state: dict, e: dict) -> Action:
    eid, arm = e["id"], state["arm"]
    if e["state"] in ("FAILED", "TIMED_OUT"):
        return Action("reject", f"Run {e['state'].lower()}: {e.get('failure')}. No metrics were recorded, so there "
                                "is nothing to promote.", (eid,), eid)
    if e["state"] == "INVALID":
        return Action("reject", f"Failed validity gate(s): {_failed_gates(e)}. Throughput "
                                f"({_fmt(_speedup(e))}) is not considered for an invalid run.", (eid,), eid)
    s, m = _speedup(e), e["metrics"]
    if e["split"] == "audit":
        worst_domain, worst = min(m["speedup_by_domain"].items(), key=lambda kv: kv[1])
        if s >= 1 + MIN_EFFECT and worst >= AUDIT_DOMAIN_FLOOR:
            return Action("keep", f"Audit passed: {_fmt(s)} on the independent longer-prompt split, weakest domain "
                                  f"{worst_domain} at {_fmt(worst)}.", (eid,), eid)
        return Action("reject", f"Dev gain did not survive the audit split: {_fmt(s)} overall, {worst_domain} at "
                                f"{_fmt(worst)} (needs >= {1 + MIN_EFFECT:.2f}x and every domain >= "
                                f"{AUDIT_DOMAIN_FLOOR:.2f}x).", (eid,), eid)
    runs = [r for r in state["dev_candidates"] if key(r["config"]) == key(e["config"]) and r["state"] == "VALID"]
    ids = tuple(r["id"] for r in runs)
    replicated = arm == "evidence_aware" and len(runs) > 1   # a replicate is judged against its earlier runs
    if s <= 1.0 and not replicated:
        return Action("reject", f"Valid but no gain: {_fmt(s)} vs the in-run target-only control.", (eid,), eid)
    if arm != "evidence_aware":
        return Action("keep", f"Valid with dev speedup {_fmt(s)}. This arm ranks on the dev score alone.", (eid,), eid,
                      uncertainty="Single run; not replicated or audited.")
    if len(runs) == 1:
        if m["speedup_ci_low"] > 1 + MIN_EFFECT:
            return Action("keep", f"Gain is clear of within-run variation: {_fmt(s)}, CI low "
                                  f"{_fmt(m['speedup_ci_low'])} > {1 + MIN_EFFECT:.2f}x. Contender, pending audit.",
                          ids, eid, uncertainty="CI covers prompt variation within one run, not run-to-run drift.")
        return Action("request_replicate", f"Apparent gain {_fmt(s)} is within measured variation (CI low "
                                           f"{_fmt(m['speedup_ci_low'])} <= {1 + MIN_EFFECT:.2f}x). Replicate before "
                                           "believing it.", ids, eid)
    speeds = [_speedup(r) for r in runs]
    spread = f"replicates: {', '.join(_fmt(x) for x in speeds)}"
    if all(x > 1 + MIN_EFFECT for x in speeds):
        return Action("keep", f"Gain held across {len(runs)} runs ({spread}). Contender, pending audit.", ids, eid)
    if max(speeds) > 1 + MIN_EFFECT and min(speeds) < 1.0:
        return Action("flag_investigate", f"Replicates disagree in direction ({spread}). Not promotable; the "
                                          "measurement is unstable for this config.", ids, eid)
    return Action("reject", f"Gain did not hold on replication ({spread}); within measured variation.", ids, eid)


# ---------- choosing the next step ----------

def decide(state: dict[str, Any]) -> Action:
    """Next action for a study. Deterministic for a given state (TPE arm delegates to the Rust optimizer)."""
    exps, arm = state["experiments"], state["arm"]
    finished = lambda e: e["state"] in ("VALID", "INVALID", "FAILED", "TIMED_OUT")  # noqa: E731

    def baselines(split: str) -> list[dict]:
        return [e for e in exps if e["kind"] == "baseline" and e["split"] == split]

    base = next((e for e in baselines("dev") if e["state"] == "VALID"), None)
    if base is None:
        failed = [e for e in baselines("dev") if finished(e)]
        if len(failed) >= MAX_BASELINE_ATTEMPTS:
            return Action("stop", f"No valid target-only baseline after {len(failed)} attempts "
                                  f"({_failed_gates(failed[-1])}). Nothing can be compared.",
                          tuple(e["id"] for e in failed))
        return Action("run_baseline", "No comparable evidence yet: measure the target-only baseline first.",
                      hypothesis="Target-only greedy decoding is deterministic and gives the reference tokens.")

    for e in exps:
        if e["kind"] == "candidate" and finished(e) and e["verdict"] is None:
            return _judge(state, e)

    if arm == "target_only":
        return Action("stop", "Control arm: target-only baseline recorded.", (base["id"],))

    remaining = state["max_trials"] - len(exps)
    cands = state["dev_candidates"]
    tried = {key(e["config"]) for e in exps if e["kind"] == "candidate"}
    latest: dict[str, dict] = {key(e["config"]): e for e in cands}   # latest dev run per config
    audits = {key(e["config"]): e for e in exps if e["kind"] == "candidate" and e["split"] == "audit"}
    contenders = sorted((e for k, e in latest.items() if e["verdict"] == "keep" and k not in audits),
                        key=_speedup, reverse=True)

    if arm != "evidence_aware":
        nxt = PROPOSERS[arm](state, tried) if remaining >= 1 else None
        if nxt:
            return nxt
        if contenders:
            best = contenders[0]
            return Action("promote", f"Highest valid dev speedup {_fmt(_speedup(best))} among "
                                     f"{len(tried)} configs. Promoted on the dev score alone.",
                          tuple(e["id"] for e in contenders), best["id"],
                          uncertainty="Not replicated and not checked on the audit split.")
        return Action("stop", f"Budget spent or search space exhausted after {len(tried)} configs; none was valid "
                              "and faster than target-only.", tuple(e["id"] for e in cands))

    # ----- evidence_aware -----
    passed = [e for e in audits.values() if e["verdict"] == "keep"]
    if passed:
        best = max(passed, key=_speedup)
        dev = latest[key(best["config"])]
        return Action("promote", f"Config {best['config']} is valid, gained on dev ({_fmt(_speedup(dev))}) and "
                                 f"survived the independent audit ({_fmt(_speedup(best))}).", (dev["id"], best["id"]),
                      best["id"], uncertainty="Pilot-sized prompt sets; sealed split not evaluated. Not a "
                                              "statistical significance claim.")

    valid = [e for e in cands if e["state"] == "VALID"]
    invalid_cfgs = {key(e["config"]): e for e in cands if e["state"] == "INVALID"}
    if not valid and len(invalid_cfgs) >= SYSTEMATIC_INVALID:
        gates = sorted({g for e in invalid_cfgs.values() for g in e["validation"].get("gates_failed", [])})
        return Action("investigate", f"{len(invalid_cfgs)} distinct configs failed validity ({', '.join(gates)}) "
                                     "and none passed. The failure does not look config-specific, so more knob "
                                     f"search is unlikely to help. {remaining} trial(s) left unspent. Investigate the "
                                     "engine/correctness issue before searching further.",
                      tuple(e["id"] for e in invalid_cfgs.values()),
                      uncertainty=f"Heuristic threshold ({SYSTEMATIC_INVALID} configs); untried configs could behave "
                                  "differently.")
    valid_cfgs = {key(e["config"]) for e in valid}
    if not contenders and len(valid_cfgs) >= FUTILITY_CONFIGS and max(map(_speedup, valid)) < FUTILITY_SPEEDUP:
        return Action("stop", f"{len(valid_cfgs)} valid configs tried, best {_fmt(max(map(_speedup, valid)))}: all "
                              f"more than {1 - FUTILITY_SPEEDUP:.0%} slower than target-only. Stopping with "
                              f"{remaining} trial(s) unspent.", tuple(e["id"] for e in valid),
                      uncertainty="Futility heuristic, not a value-of-information estimate.")

    pending_rep = [e for e in latest.values() if e["verdict"] == "request_replicate"]
    if pending_rep and remaining >= 1:
        e = max(pending_rep, key=_speedup)
        return Action("replicate", f"Re-run {e['config']} to see whether {_fmt(_speedup(e))} holds.", (e["id"],),
                      config=e["config"], parent_id=e["id"],
                      hypothesis=f"The gain measured in {e['id']} reproduces on an independent run.")

    audit_base = next((e for e in baselines("audit") if e["state"] == "VALID"), None)
    audit_base_failed = len([e for e in baselines("audit") if finished(e)]) >= MAX_BASELINE_ATTEMPTS
    audit_cost = 1 if audit_base else 2   # candidate on audit split (+ audit-split baseline once)
    if remaining - audit_cost >= 1:       # always hold back enough budget to audit before promoting
        nxt = _neighbour_proposal(state, tried)
        if nxt:
            return nxt
    if contenders and remaining >= audit_cost and not audit_base_failed:
        best = contenders[0]
        if audit_base is None:
            return Action("run_baseline", f"Reserved budget for verification: measure target-only on the audit "
                                          f"split so {best['id']} can be audited.", (best["id"],), split="audit",
                          hypothesis="Target-only reference tokens for the independent audit split.")
        worst_domain, worst = min(best["metrics"]["speedup_by_domain"].items(), key=lambda kv: kv[1])
        return Action("audit", f"Before promoting {best['config']} ({_fmt(_speedup(best))} on dev, weakest domain "
                               f"{worst_domain} at {_fmt(worst)}), check it on the independent longer-prompt split.",
                      (best["id"],), config=best["config"], split="audit", parent_id=best["id"],
                      hypothesis=f"The dev gain of {best['id']} generalises to longer prompts in all domains.")
    if contenders:
        return Action("stop", f"Best contender {contenders[0]['id']} ({_fmt(_speedup(contenders[0]))} on dev) could "
                              "not be audited within the remaining budget. Not promoted.",
                      tuple(e["id"] for e in contenders), uncertainty="Unaudited dev gain.")
    return Action("stop", f"No config survived: {len(tried)} tried, none both valid and reliably faster than "
                          "target-only.", tuple(e["id"] for e in cands))
