# Möbius 2.0 — Frontier ML Research Program (as of 2026-11-20)

**Repository:** https://github.com/lonexreb/mobius  
**Working branch:** `revival/sundai-recursive-experiments-2026`  
**Status:** Research proposal and Claude Code execution handoff; NOT a claim of achieved benchmark gains.

## Decision: Retire the UCI shopper classifier as flagship

The former UCI shopping-intent benchmark and local digits study remain **offline smoke tests only** for data hashing, score ingestion and experiment histories. They are insufficient evidence for frontier ML research.

**New product/research thesis:** Möbius is a **proxy-aware autonomous research controller**. Research agents can already edit code, train models and improve visible metrics. The harder unresolved question is whether they select the right experiments, calibrate *imperfect proxies* against true objectives, replicate uncertain findings, and allocate fixed compute budgets toward *out-of-distribution* improvements.

We propose a research question, not claim of novel SOTA:
> Under a fixed, total GPU + inference token + verifier budget, can an evidence-calibrated controller improve held-out task quality and reduce false experiment promotion versus identical research agents guided by visible validation score alone?

### Dates and prior art: important not to misrepresent novelty

- HF standalone `huggingface/ml-intern` is archived (Sept 2026). ML Intern continues **in HuggingChat** via https://huggingface.co/chat/?mode=ml-intern; reads papers, runs Jobs, builds datasets, trains/fine-tunes models, evaluates, and deploys demos. https://huggingface.co/docs/chat-ui/ml-intern/getting-started
- HF case study (Oct 8 2026): specialized Qwen3.5-2B citrus VLM, teacher-student 9B→0.8B prompt rewriter, image LoRAs and data construction. https://huggingface.co/blog/building-with-ml-intern
- HF (Nov 20 2026) **harness adapters at scale**: Qwen3.8-27B harness-specific GRPO LoRAs with held-out evaluations and reduced token/tool usage. https://huggingface.co/blog/harness-adapters-at-scale
- Induction Labs **FrontierML** (Oct 13 2026): 21 open-ended ML tasks across training, inference, vision, audio, computer use, etc., at $10/$50/$200 experiment budgets. https://www.frontierml.org/blog/ and source https://github.com/induction-labs/frontier-ml
- Synthetic Sciences **FrontierML** (Nov 18 2026, a distinct benchmark with coincidentally the same name): 25 hidden-final-objective empirical ML tasks, 15 available on Harbor; agents must design local proxies and decide whether to trust them. https://hub.harborframework.com/datasets/synthetic-sciences/frontierml/latest
- METR (Nov 20 2026): strong automated metric optimization, weaker judgment under limited/expensive feedback. https://metr.org/blog/2026-11-20-q4-2026-capability-update/
- **AIDE²** (Sep 22 2026): prior work already demonstrates self-editing research agents and held-out transfers. We must **not** claim generic RSI or agent self-rewrite is novel. https://arxiv.org/abs/2609.26457
- Microsoft Research (Sep 2026): reward hacking and independent measurement concerns in research agents. https://www.microsoft.com/en-us/research/publication/reward-hacking-challenges-oversight-of-autonomous-research-agents/
- Dynamic speculative decoding is already studied (e.g. LTD https://arxiv.org/abs/2603.01639, SpecKV https://arxiv.org/abs/2605.02888): do **not** claim adaptive draft length by itself is novel.

## Ranked experimental workloads

### Track A — fastest real GPU testbed: cross-domain speculative decoding efficiency

This is a **testbed for research decisions** rather than claiming to invent speculative decoding. It is one of Induction Labs FrontierML's published task families.

- Target: `Qwen/Qwen3-8B` or verified compatible 2026 Qwen target model; candidate draft: compatible sub-1B model (validate architecture/tokenizer/sampling correctness first).
- Runtime: vLLM/SGLang or another verified engine supporting this *specific* target/draft combination. Use Autolab GPU or Hugging Face Jobs; local Apple Silicon for parsing, unit tests and limited baseline only.
- Data **input prompts**, not fine-tune labels: 
  - `livecodebench/code_generation_lite`, selected temporal code prompts (do not train on held-out eval split);
  - `HuggingFaceH4/MATH-500`, math prompts for inference workload only; this benchmark is old and not used to claim novel math capability;
  - `HuggingFaceH4/ultrachat_200k`, disjoint chat prompts.
- Partition prompts **by domain and source**: development code+math+chat strata; final sealed OOD mix must never be visible to the hypothesis agent. Pin full revisions, timestamps, prompt hashes and sampling settings.
- Knobs: draft model, `gamma`/lookahead length, verification batch size, decoding engine flags, workload composition, optional drafter adaptation once base path is stable. Do not alter target weights or output semantics when claiming lossless acceleration.
- Baselines: ordinary target autoregressive generation (no speculation); best hand-tuned *fixed* lookahead; random or TPE config optimization; off-the-shelf adaptive controller if available.
- Metrics: **exact greedy output equality** against target-only decoding, decode tokens/s, p50/p95 end-to-end latency, time to first token, accepted tokens per verification, verification overhead, peak VRAM, GPU-seconds and total dollars. Exact match is mandatory for deterministic lossless configuration. If sampling, distribution-preserving verification requires separate probabilistic test protocol.
- Research signal: the selected config may look strong on development chat prompts but regress on held-out code/math or under batch-size shift. Möbius should allocate budget to targeted replication/OOD checks instead of blindly promoting a high proxy score.

### Track B — frontier deep research: transfer-aware agent post-training

- Train/eval source: `harness-hub/agentic-coding-envs` with 871 documented tasks, 171 reported qualified for GRPO in HF Nov 20 article. Respect license and source task data. The label *qualified* is model-specific; remeasure pass variance with chosen base model.
- Method: HF TRL `AsyncGRPOTrainer`, LoRA/PEFT, OpenEnv/Harbor sandboxes, one isolated reproducible environment per rollout. Benchmark/inference code is not altered by candidate.
- Start with a small feasible open model after confirming tool use and non-degenerate outcomes; do NOT assume Qwen3.5-0.8B can solve enough SWE tasks to supply GRPO learning signal. Raise model size/compute only after qualified trials exist.
- Baselines: frozen base model, fixed random curriculum, difficulty-based curriculum, reward-only RL, and **identical proposer without Möbius evidence controller**.
- Independent evaluation: `terminal-bench/terminal-bench@4.1.0` (Nov 3 2026, 66 tasks), not the stale 2.1 alone; plus repo-disjoint evaluation and, if funded, SWE-bench Pro.
- Metrics: pass@1; solved tasks per total dollar; token and tool calls per task; success variance across seeds; cross-harness / cross-repo OOD generalization; fraction of GRPO rollouts all-pass/all-fail; reward hacking or verifier tampering rate.
- Research hypothesis: a controller that selects useful training environments and *replicates unexpected reward gains* outperforms fixed difficulty sampling at matched budget. This is distinct from just retraining the published harness adapters.

### Track C — longer horizon: weight-space / compression experiments

- Candidate: ternary quantization of `Qwen3.5-4B` (explicit Induction Labs FrontierML task) and/or output-aware pruning/quantization studied separately.
- User's Jacobian-Lens depth pruning research is relevant prior experience; treat extensions as hypotheses, not guaranteed improvements.
- Measure downstream reasoning/generalization, perplexity, memory footprint, tokenizer-specific serving compatibility, quality at fixed latency, and calibrated proxy-to-hidden metric correlation.
- Cost and time greater than Track A; do not try all tracks during one hackathon.

## Möbius's novel *proposed* mechanism: evidence-calibrated allocation

Current Mobius 0.1 implements fixed parameter search/agent loops; it does NOT implement the following.

State: candidate hypothesis, claimed mechanism, experiment revision, visible proxy score, uncertainty, estimated cost, source OOD risks, prior outcomes and failure modes.

Actions: `propose`, `pilot`, `replicate`, `run stronger verifier`, `inspect data leak`, `scale`, `revert`, `stop`.

At each step choose action `a` using estimated value-of-information:
```
VOI(a) = E[decision quality gain | currently available evidence]
         - lambda * (GPU + agent + verification cost)
         - mu * (risk of invalid or non-generalizing claim)
```
The true hidden score is **never used** in online selection; estimate gains from prior task families and calibrate them retrospectively using other held-out tasks, not the active task's final test.

Three controllers, same proposer and budget:
1. `metric_greedy`: maximize current cheap proxy.
2. `tpe`: standard Möbius TPE or random search on same knobs.
3. `evidence_aware`: select replication / stronger eval / new proposal based on uncertainty and expected decision value.

The first two serve as essential controls. AIDE² and unconstrained Claude agents are secondary agent-level comparison arms, *not* necessarily budget-matched to TPE.

## Data, artifact and metrics contracts

### 1. Data registry (immutable source of truth)

Each dataset in `dataset_registry.yaml` must record: Hub ID and revision SHA, license, original release date, split eligibility, data hash, source publication, leakage/canary policy, and exact `train/dev/ood_sealed` role. Keep eval canaries and test solutions strictly out of all candidate training contexts and prompts.

### 2. Experiment ledger

Required fields:
- `experiment_id`, `parent_id`, `research_task`, `hypothesis`, `action`, `proposer_model`, `controller_arm`;
- code and model revision hashes, eval-hill hash, dataset revisions/split hashes, seed, runtime engine, GPU type;
- machine-recomputed metric estimates, uncertainty/replication counts, failure/timeout/censoring flags;
- trial budget reserved/spent: GPU compute, model-agent inference tokens, verifier spend, total wall time;
- external Autolab/HF job ID, source URLs, immutable artifacts, decision and rationale.
Keep numeric metrics compatible with `mobius tell` but persist non-numeric provenance **separately**, since current ask/tell interfaces are not a full research ledger.

### 3. Scientific scoring

**Primary across workloads:** `best valid held-out task score at fixed TOTAL cost budget`, and `area under best-valid-held-out-quality vs spend curve` **for evaluation only**, never online. For ongoing optimization decisions use only affordable development/proxy metrics.

**Secondary:** Spearman correlation of development/heldout deltas across candidates (retrospective), false-promotion rate (fraction of adopted candidates regressing on sealed OOD), replicated-effect confidence intervals, regret at matched budgets, cost/time to first genuine improvement, failure/hallucination/leak rates.

Any quality-per-dollar comparison needs comparable denominator; do not silently assume missing cost is zero. Report wall time and GPU seconds separately.

### 4. Evaluation integrity

- Isolated ephemeral candidate working tree; separate immutable evaluator/hill; no hidden data access or internet at final scoring.
- Identical benchmark versions, task order, max tool calls, GPU constraints and random seeds across matched policies.
- Train/development/hidden test across **task families**, not just random examples from the same repository.
- At least 3 independent stochastic seeds before strong conclusions. A Sundai demo with 1–3 short iterations is a pilot, not proof of significance or a new SOTA.
- Invalidate claims if the agent edits test scripts, leaks labels, swaps target model, changes score definition, modifies golden answers or fabricates logs.
- Safety and costs: hard cap spending and CPU/GPU time, explicit opt-in before public deployments or costly jobs.

## Sponsors and Hugging Face roles

| Component | Responsibility |
| --- | --- |
| Möbius Rust core | hypothesis priority, evidence store, budget, stop/revert policy, experiment decisions |
| Claude Code | proposes candidate source changes and research hypotheses; one edit worktree per candidate |
| Autolab | frozen Hill evaluation, agent-controlled Climb with actual scored executions, inspect `autolab log`/`autolab metrics` |
| Hugging Face ML Intern / Jobs | model/data research, baseline training, fine-tuning and model/dataset artifacts; optional direct Jobs compute |
| Egbe | audience research/distribution of resulting research artifacts, NOT a substitute for independent ML evaluator |

Autolab docs: https://docs.autolab.ai/guides/experiments/  
HF Jobs: https://huggingface.co/docs/hub/jobs-training

## Hackathon delivery ladder (AI-first; estimates, not promises)

**~60–120 minutes with coding agents:** Verify existing Rust test baseline. Instrument Track A; download pinned small prompt subsets; run target-only and fixed-lookahead baseline with a verified compatible inference engine. If unavailable, record exact hardware/API blocker instead of fabricating metrics.

**Further 1–3 hours of live experiments if compute is connected:** Create/pin Autolab Hill. Run 4–8 candidate policies with visible quality/speed/cost traces and at least one deliberate OOD confirmation. Compute a reproducible difference, or honestly show no win.

**Longer work (one or more GPU sessions, depending on environment):** Train Track B small LoRA using valid reward variance, run repo-disjoint eval, then full Terminal-Bench 4.1 if funded. This is serious independent research, not a guaranteed 2-hour training result.

**Research paper target after replication:** Compare evidence-aware and score-greedy autonomous research policies across 3+ different real task families, matched total budget and task-family heldouts. Stress test proxy gaming and claim validity. Publish negative results if necessary.

## First Claude Code instruction

> Read `CLAUDE.md`, `crates/mobius-claw/src/agent.rs`, `crates/mobius-cli/src/commands/ask.rs`, `tell.rs`, `docs/SUNDAI_SPONSOR_INTEGRATION.md`, and this document. Begin with read-only discovery. Check latest `cargo fmt --check`, clippy and test results; do not assume historic counts. Produce a tiny design review listing the exact functions/interfaces you will reuse.
>
> For the first milestone, implement `experiments/frontier2026/speculative/{README.md,manifest.yaml,fetch_data.py,eval.py,tests/}` with data revision hashes, deterministic prompt subsets, target-only baseline and fixed-gamma control, and an immutable JSON metric record. Test on one verified local/remote backend (don't assume GPU access). The evaluator must check exact greedy output equivalence, throughput and GPU/time cost. Do not edit `mobius-core` or `mobius-claw` without a failing contract test.
>
> Then design an Autolab Hill and wire machine-verified scored results into the existing ask/tell interface plus a separate provenance file. Build a tiny `evidence_aware` proposal-vs-replication controller against the same candidate slate and budget as `metric_greedy`. Give me one actual before/after measurement and a reproducibility command, or state why you couldn't run it.
>
> **Do not**: claim SOTA, change evaluator to improve the score, access held-out data during search, invent GPU metrics, hardcode credential values, or silently exceed any budget.

## Success criteria for Möbius 2.0

A credible version-0.2 claim is **not** “the model scored higher on a benchmark”. It is “under the same research budget and task conditions, Möbius chose better *experiments* and avoided unreliable progress claims, with independently checkable outcomes.” That makes it an experiment-level scientific contribution worth testing.
