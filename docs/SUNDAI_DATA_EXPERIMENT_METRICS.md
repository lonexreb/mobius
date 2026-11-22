> **Superseded for frontier research (2026-11-22):** The UCI Online Shoppers task and scikit-learn digits experiments in this document are retained **only as smoke tests** for data ingestion, deterministic splits, and metrics. The approved frontier experimental research program is [MOBIUS_FRONTIER_RESEARCH_NOV_2026.md](MOBIUS_FRONTIER_RESEARCH_NOV_2026.md). Do not run UCI shopping as the Sundai flagship or cite digits smoke tests as evidence of autonomous ML research.

# Sundai: Data, Hypothesis, Experiment, and Metrics Contract

**Target project:** lonexreb/mobius • **Date:** 2026-11-22  
**Status:** Public-source research and reproducible local smoke-tested harness; no Autolab account-run or UCI model result yet.

## Primary scientific questions

**Model question:** Given UCI purchase-intent session records, can controlled model changes improve purchase ranking quality under time/budget constraints?

**Meta-optimizer question:** Given identical candidate search spaces and 10 evaluator calls, does an AI hypothesis proposer improve best validation Average Precision faster than Möbius TPE or random search?

Do not conflate improvements in a fitted model with evidence of a superior autonomous research policy.

## Data layer (official source)

- Source: UCI Online Shoppers Purchasing Intention Dataset, ID 468, DOI 10.24432/C5F88Q, CC BY 4.0; https://archive.ics.uci.edu/dataset/468/online
- 12,330 sessions, 10,422 nonbuyers / 1,908 buyers (~15.47% positive).
- 17 original predictor variables and Revenue label; exclude PageValues in initial benchmark because it depends on historical transactions and may overstate prospective prediction power.
- Note: other full-session aggregates do not establish genuine real-time deployability.
- Stratified 60% train / 20% validation / 20% held-out test, seed 2026; store deterministic split indices, dataset hash, split hash.
- Fit preprocessing on training only. Record exact feature column allowlist and software dependencies.
- Test must remain unobserved until model/search policy frozen. In Autolab Hill hold it inside private/, not a user-editable candidate directory.

## Experiment layer

### Model controls (E0–E3)

- E0: Dummy majority/prior classifier; ranking AP should be at approximate prevalence.
- E1: Logistic regression; standardized numerical and one-hot categorical features.
- E2: RandomForest/ExtraTrees with bounded tree counts and depth.
- E3: Histogram gradient boosting with bounded learning rate and iterations.

### Search policy controls (E4–E5)

- E4a: Random config proposer on fixed allowed search space.
- E4b: Möbius TPE on same allowed space.
- E5a: LLM proposer restricted to **exact same** allowed parameter/config space. The direct sample-efficiency comparison must use this arm.
- E5b: LLM proposer permitted to edit a restricted candidate file and feature preprocessing. This is an **extended-capability demonstration**, not an apples-to-apples comparison against parameter-only TPE.

Fixed per-policy budget: 10 scored attempts and a declared compute/token cap. Include failed evaluations in resource accounting. Record proposal cost separately from model training cost. Run three random seeds if feasible; a single seed is preliminary.

### Run contract (requested additions, NOT current built-in API)

```json
{
  "experiment_id": "uuid",
  "dataset_id": "uci-468",
  "dataset_sha256": "sha256",
  "split_sha256": "sha256",
  "hill_version": "pinned-hill-hash",
  "policy": "random|tpe|llm-config|llm-code",
  "hypothesis": "testable claim",
  "parent_id": null,
  "candidate_git_sha": "sha",
  "seed": 2026,
  "status": "pending|success|error|timeout|rejected",
  "val_average_precision": null,
  "roc_auc": null,
  "precision_at_10pct": null,
  "recall_at_10pct": null,
  "lift_at_10pct": null,
  "brier": null,
  "train_seconds": null,
  "predict_ms_per_1000": null,
  "llm_cost_usd": null,
  "compute_cost_usd": null,
  "accepted": null
}
```

- No invented cost values; null if not available, never silently equate missing cost to free.
- Validate numerical finiteness; invalid/missing score marks failure.
- Idempotently sync from Autolab run ID, preserve external run and source URLs.
- Include candidate patch diff and rejection reason in append-only provenance record.
- Make stop conditions explicit: budget, plateau, number of trials, or valid target.

## Metrics layer

**Primary:** validation Average Precision (maximize). Appropriate for imbalanced positives.

**Secondary:** ROC AUC; precision@top10%; recall@top10%; lift@top10% = precision@top10% / positive prevalence; Brier score; macro-F1; training/inference time; memory; success rate.

**Meta-optimizer:** best validation AP after K=10, normalized area under best-so-far AP vs trials, best AP at matched cost, attempts until first improvement beyond baseline, failed trial fraction.

**Promotion rule:** only valid evaluation on the same pinned Hill, same split and metric definition, within declared constraints. Proposals may be rejected; a validation gain need not automatically mean robust or generalizable improvement. Frozen winner is scored on held-out test exactly once.

**Evidence:** differences versus matched baselines, paired repeated seeds, confidence intervals where feasible, detailed run trace, and explicit uncertainty. Do not claim significance from one small hackathon run.

## Sponsor connections

### Autolab

- Official Hill documentation: https://docs.autolab.ai/guides/hills/
- Pinned Hill evaluator and held-out private data.
- CLI: `autolab hills new mobius-shoppers` → implement Hill → `autolab hills check mobius-shoppers` → commit/push → configure compute and max-cost → run Climbs.
- Official score import: `autolab log`, `autolab metrics <RUN_ID> --format json`.
- The agent must not edit evaluator or its private data. Explicitly distinguish Möbius-suggested hypotheses from Autolab's own agent-generated ideas.

### Egbe

- Source: https://docs.egbe.ai/infrastructure/analytics-and-email/
- Separate business experiment; funnel events: verified acquisition visit → CTA click → qualified signup → activation.
- Primary exploratory metric: qualified signups / eligible unique visitors; also visitor count, cost per signup, attribution coverage, failed events.
- Track unique users, not raw event count; preserve a consistent time window, filter test traffic, require approvals before spend/outbound.
- Do **not** treat the UCI dataset as evidence of effects on actual Egbe campaigns.

## Local harness already exercised

An independent ZIP contains `experiment.py`, `report.py`, `test_experiment.py`, README and pinned-range requirements. It runs a scikit-learn *digits* dataset smoke test offline and accepts downloaded UCI CSV at `data/online_shoppers_intention.csv`. Three deterministic seeds × five models were run, all local validation only. Three unit tests pass. No code or artifact in this repo branch should be described as Autolab-executed.

Sample local commands after copying/extracting the harness into a workspace:

```bash
python -m pip install -r requirements.txt
python experiment.py --dataset digits --suite --seed 2026
# Download official UCI dataset CSV into data/online_shoppers_intention.csv
python experiment.py --dataset shoppers --suite --seed 2026
python report.py --dataset shoppers
```

## Claude Code next task

Read this file, README, CLAUDE.md, crates/mobius-claw/src/agent.rs, and the current ask/tell CLI source. Verify existing Rust tests and reproduce the local harness. Then:

1. Place approved harness source under `examples/sundai-study/` and commit it.
2. Acquire UCI CSV directly from its official source, pin SHA, audit schema, and log the actual E0–E3 validation numbers.
3. Create a frozen `mobius-shoppers` Autolab Hill. Enforce candidate/evaluator separation and run `autolab hills check`.
4. Run and import at least one official sponsor experiment and its provenance.
5. Implement matched config-space random/TPE/LLM-search arms.
6. Compare results honestly; do not call synthetic/offline tests sponsor experiments.
7. Finally connect Egbe analytics for a separate verified product funnel.

Keep unrelated refactors out of this branch. Implement behavioral changes with failing tests first. Preserve main.
