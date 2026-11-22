# ♾ Möbius Lab — Claude Code End-to-End Implementation Handoff

**Product:** Möbius Lab, the evidence-calibrated autonomous ML research director  
**Repository:** https://github.com/lonexreb/mobius  
**Implementation branch:** revival/sundai-recursive-experiments-2026  
**Prepared:** November 22, 2026 (Pacific)  
**Reference research cutoff:** November 20, 2026  
**Priority:** Sundai live demo, then a reusable research platform  
**Status:** Implementation specification, NOT implemented product or validated research result  
**Role:** You are Claude Code, lead engineer, ML research engineer, agent-systems engineer, and test owner.

---

## 0. Read This First: The One-Sentence Goal

**Build Möbius Lab: a web app where a user gives an ML research objective and a budget; an evidence-aware controller proposes experiments, runs them through Autolab, independently verifies the outcomes, stores an audit trail, and decides whether to promote, replicate, investigate, or reject each candidate.**

Do NOT create a generic chatbot. Do NOT rebuild the existing Rust Möbius optimizer. Do NOT claim a speedup before observing one on matched hardware and held-out prompts. The product is **experiment judgment**, not code generation.

**Pitch:** "Anyone can automate an experiment. Möbius automates the decision about what evidence to trust and which experiment to run next."

### Immediate working style and priorities

1. First inspect existing code in read-only mode; report exact file:line references. Run existing tests before edits.
2. Work only on the existing revival branch, or a narrower feature branch off it; never modify main directly.
3. Implement **one thin, vertical slice**: create study → define hypothesis → evaluate candidate → persist result → decide → display evidence. Make it work with a real local evaluator before adding cloud integrations.
4. Use AI-first implementation and small clean commits. Frontend/API scaffolding can be rapid; actual GPU measurements, statistical validation, and authentication are separate, nontrivial phases.
5. After the first vertical slice, integrate official Autolab commands and optionally Egbe; use authentic data and real source IDs.
6. Do not call something "running", "passed", "deployed", "connected", or "improved" unless you just verified it.
7. Before paid compute, credential setup, outbound communications, or public deployment, obtain explicit human authorization. Defaults are zero spending and read-only.
8. Do not ask the user to choose every small implementation detail. Select sensible defaults, record assumptions, and keep moving. Ask only for blocked external credentials/approvals or genuinely irreversible decisions.

### Canonical sources and precedence

Read these checked-in files:
- CLAUDE.md — original Rust project conventions.
- Cargo.toml — five-crate Rust workspace.
- crates/mobius-claw/src/agent.rs — real agent loop.
- crates/mobius-core/src/compute.rs — current compute and metric parser.
- crates/mobius-cli/src/main.rs — exact command flags.
- crates/mobius-cli/src/commands/ask.rs and tell.rs — metric handoff.
- docs/MOBIUS_FRONTIER_RESEARCH_NOV_2026.md — canonical frontier research thesis.
- experiments/frontier2026/research_plan.yaml — research planning metadata, NOT runnable API.
- docs/SUNDAI_SPONSOR_INTEGRATION.md — sponsor setup background.
- docs/SUNDAI_DATA_EXPERIMENT_METRICS.md — **superseded for flagship**; UCI/digits are smoke tests only.

**This file is the canonical implementation plan for the Sundai Möbius Lab PRODUCT.** If an older document suggests that UCI shoppers or handwritten digits should be the headline demo, this newer file overrides that. No documented historical test count proves the current environment passes tests.

---

## 1. What Already Exists vs. What Must Be Built

### Verified existing repository as of this handoff

Five Rust crates:
- mobius-core: experiment schema, JSONL/SQLite storage, budget guard, local subprocess compute traits.
- mobius-bench: F1, classification accuracy, timestamp MAE scoring and evaluation.
- mobius-claw: strategies, agent loop, learning, pruning, constraints, heuristics.
- mobius-cli: init, run, sweep, agent, ask, tell, compare, status, history, export, dashboard, etc.
- mobius-mcp: ten documented MCP tools and three resources.

The README documents ten optimization strategies and 161 tests. Most recent checked GitHub Actions run was April 4, 2026, successful. **Re-run locally**.

Important direct code observations:
- AgentLoop has a **RESEARCH no-op**. It does NOT perform LLM literature review or autonomous code patches yet.
- The existing CLI "agent" command sets its primary metric to "f1" in its configuration.
- **mobius ask --metric NAME --strategy tpe --store sqlite** supports named numeric metrics without changing core code.
- **mobius tell --config JSON --metrics JSON --store sqlite** appends successful numeric observations; it is NOT a provenance store, does NOT represent failed jobs, and does NOT protect against repeated imports.
- "ask" and "tell" use ~/.mobius/history.db or ~/.mobius/history.jsonl. Isolation across studies/arms is NOT yet built in.
- The current generic OutputParser JSON path expects a JSON line containing the literal "f1". Do not route arbitrary sponsor metric payloads through it and assume they parse.
- SubprocessBackend splits commands at whitespace. Do not pass untrusted/user-generated shell commands through this helper. For external commands use Python subprocess.run with an explicit argument array and timeouts, or an appropriate sandboxed backend.
- The latest checked-in MEMORY.md is historically stale; it describes earlier unfinished phases.

**Build as separate adapters around this core**. A narrow core patch is allowed only when supported by failing tests. Do not refactor the five crates to match a brand-new architecture.

### Build now

- A clean UI where goals, hypotheses, budgets, runs, evidence and decisions are visible.
- A FastAPI orchestration service with per-study state, idempotent run ingestion, provenance, and budgets.
- A small typed research controller: baseline, metric-greedy, TPE, evidence-aware.
- A fixed evaluator + deterministic prompt registry for inference efficiency.
- An Autolab adapter that uses documented CLI operations, snapshots authentic results, and records source IDs.
- An import into Möbius numeric ask/tell history **only for actual valid results**, isolated by policy arm.
- A replay mode for previously captured genuine experimental traces, clearly labeled.
- A local offline mode that exercises plumbing and safety but is NOT misrepresented as real GPU research.
- Optional Egbe marketing/growth connection after the research demo is solid.
- Exportable machine-readable and human-readable research report.

### Not now

- General-purpose no-code agent builder, multi-tenant SaaS billing, broad RBAC.
- Kubernetes autoscaling, distributed GPU fleet, Modal/RunPod/Slurm adapters simultaneously.
- Full RL fine-tuning with 27B models, end-to-end TRL/GRPO setup, or long 8xH200 jobs.
- Train a new EAGLE speculator; use a verified pretrained one first.
- Unrestricted self-editing of the evaluator, benchmarks, cost guard, or policy tests.
- Make an "AI scientist" claim with just three cherry-picked trials.
- Build a second autonomous loop duplicating Autolab or an outbound marketing engine duplicating Egbe.

---

## 2. Demo User Story and Definition of a Delightful Product

### Primary user

Research engineers and ML platform engineers trying to improve real models without repeatedly babysitting experiments.

### User asks

"Improve inference throughput for Qwen3-8B without changing the outputs, under my research budget. Test math, code, and chat workloads. Tell me which results you trust and what to test next."

### Product behavior

1. User selects the "LLM inference efficiency" research template; sees target model, prompt sources and evaluation checks.
2. User chooses one of: **local validation**, **verified Autolab project**, or **replay recorded results**. Mode is conspicuous and persistent.
3. User sets a limit (trial count, approximate GPU/agent token spend, time cap). Start button is disabled for paid remote mode until connected and approved.
4. UI displays a baseline and a prioritized hypothesis with rationale ("try lookahead=3", or "replicate a surprising speedup on held-out math prompts").
5. Evaluation executes on a fixed Hill or independent local benchmark, not by reading the candidate's self-reported score.
6. Results arrive with source IDs, model hashes, evaluator hash, device, metric directions and pass/fail gates.
7. The research controller selects KEEP, REJECT, REPLICATE, AUDIT, PROPOSE_NEXT, or STOP.
8. User can click any candidate to inspect the actual configuration, why it was tried, source run, raw metrics, comparison, failures and evidence quality.
9. UI shows best VALID result vs spend, a hypothesis tree, and a researcher-style written findings report.
10. Egbe may later research who wants this product and help publish it with explicit user approval.

### Product success

A judge can understand, without reading code, the **difference between a fast-looking result and a trustworthy result**. The most memorable screen is a candidate with attractive development throughput, flagged "needs independent verification" or rejected after measured OOD regression. If data does not contain such a case, show the actual results and explain the uncertainty; do not engineer fake surprises.

### Home screen wireframe (design target, not implemented UI)

~~~text
┌──────────────────────────────────────────────────────────────────────────┐
│  ♾ MOBIUS LAB        Research Projects      Compare      Reports          │
│  Local / Autolab / Replay          Compute: Disconnected       $0 / $10    │
├──────────────────────────────────────────────────────────────────────────┤
│  Research objective: Improve Qwen3-8B throughput, preserve outputs       │
│  [ Qwen3-8B + EAGLE3 ]  [ Code | Math | Chat ]  [ Budget ] [▶ Run]        │
├──────────────────────────────────────────────────────────────────────────┤
│  Best VERIFIED speedup      Correctness       Spend        Active trials │
│        -- (not yet run)        -- / --        $0.00          0             │
├─────────────────────────────┬────────────────────────────────────────────┤
│ EXPERIMENT TIMELINE         │ RESEARCHER'S NEXT DECISION                 │
│ baseline   [pending]        │ "First get target-only baseline."          │
│ lookahead3 [pending]        │ Why: no comparable evidence yet           │
│ lookahead5 [pending]        │ [Approve] [Change constraint] [Stop]       │
├─────────────────────────────┴────────────────────────────────────────────┤
│ Best valid quality / cost curve | Domain bars: code, math, chat          │
├──────────────────────────────────────────────────────────────────────────┤
│ Evidence and provenance: model SHA | data SHA | Hill hash | source link  │
│ [Open source run]  [Compare candidates]  [Export research report]         │
└──────────────────────────────────────────────────────────────────────────┘
~~~

No made-up KPIs or decorative live charts. Before runs exist show real empty states, not sampled measurements.

---

## 3. Architecture and Ownership Boundaries

~~~mermaid
flowchart TD
    UI[React/Vite UI] -->|REST + SSE| API[FastAPI Orchestrator]
    API --> DB[(lab.sqlite / append-only audit)]
    API --> Controller[Decision Policy]
    Controller --> Mobius[Möbius Rust CLI ask/tell]
    Controller --> Proposer[Claude Code Hypothesis Proposer]
    API --> Adapters[Execution Adapter Interface]
    Adapters --> Local[Local Evaluator]
    Adapters --> Autolab[Autolab CLI / Climb]
    Autolab --> Hill[Frozen Hill / Independent Scorer]
    Local --> Evaluator[Immutable Local Scorer]
    Hill --> Ingest[Validated Results Import]
    Evaluator --> Ingest
    Ingest --> DB
    Ingest --> Controller
    DB --> Report[Report + Repro Manifest]
    Report --> UI
    API -->|optional, after approval| Egbe[Egbe Connected Claude]
~~~

### Responsibility map

| Component | Owns | Must not own |
| --- | --- | --- |
| Frontend | goal input, read-only run views, intentional controls | scientific scoring or credential access |
| FastAPI | study/run lifecycle, budget reservation, SSE, idempotency, validation | self-assigned experiment metrics |
| Möbius Rust | numeric search strategies, history-informed config proposals | UI, sponsor credentials or hidden eval data |
| Proposer | hypothesis, structured candidate config, rationale | modifying scorer, private labels, budget |
| Autolab | official experiment run, fixed Hill evaluation, source run tracking | cross-task research allocation policy |
| Local evaluator | deterministic metrics, correctness gates | selecting favorable candidates |
| Research controller | next action, audit/replication policy, stop decisions | overriding an invalid experimental result |
| Egbe | later research/distribution/analytics with approval | ML truth metric or cost authorization |

**Single-source-of-truth decision:** The new app SQLite database is authoritative for studies, attempts, approval, provenance, spend and decisions. Existing Rust Möbius SQLite/JSONL is a **derived optimization history** per study/policy, not the product system of record. Import into Rust after an authoritative validated result is committed.

### Suggested tech choices

- **Frontend:** React + TypeScript + Vite; Tailwind CSS; shadcn/ui or simple accessible primitives; Recharts for charts; TanStack Query for server state. Use existing project conventions if already present.
- **Backend:** Python 3.11+, FastAPI, Pydantic v2, SQLite WAL, standard sqlite3 or SQLAlchemy; choose a minimal consistent persistence approach. SSE via StreamingResponse. pytest/httpx for API tests.
- **Rust integration:** build mobius-cli from source; use subprocess with list argv and per-study HOME for legacy isolation. Check this isolation on macOS/Linux. Avoid "agent" hardcoded f1 path for V0.
- **Autolab:** official CLI; first use documented CLI commands, no invented private endpoints. Capture structured outputs from the live installed CLI.
- **Egbe:** official "Connect your agent" per business; installed skills/playbooks only. No imagined generic egbe command.
- **Deploy:** local API + frontend for the demo. Public read-only hosting is optional and requires confirmation and authentication design.

Pin versions only after verifying package compatibility and successful install. Do not paste a future-looking version number from memory into production requirements.

---

## 4. Repository Layout — New Files Only by Default

Create under existing repo:

~~~text
mobius/
├── Cargo.toml                       # existing, do not rewrite
├── CLAUDE.md                        # existing, obey rules
├── crates/                          # existing 5-crate Rust workspace
├── docs/
│   ├── MOBIUS_LAB_CLAUDE_CODE_HANDOFF.md  # this handoff
│   ├── MOBIUS_FRONTIER_RESEARCH_NOV_2026.md
│   ├── SUNDAI_SPONSOR_INTEGRATION.md
│   ├── MOBIUS_LAB_ARCHITECTURE.md         # generated ADR / constraints
│   ├── MOBIUS_LAB_RUNBOOK.md              # commands and troubleshooting
│   └── MOBIUS_LAB_DEMO_SCRIPT.md
├── apps/
│   ├── api/
│   │   ├── pyproject.toml
│   │   ├── .env.example                   # fake placeholders only
│   │   ├── src/mobius_lab/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── schemas.py
│   │   │   ├── persistence.py
│   │   │   ├── events.py
│   │   │   ├── budget.py
│   │   │   ├── routes/
│   │   │   │   ├── health.py
│   │   │   │   ├── studies.py
│   │   │   │   ├── runs.py
│   │   │   │   ├── experiments.py
│   │   │   │   └── reports.py
│   │   │   ├── policies/
│   │   │   │   ├── base.py
│   │   │   │   ├── greedy.py
│   │   │   │   ├── tpe_bridge.py
│   │   │   │   └── evidence_aware.py
│   │   │   ├── adapters/
│   │   │   │   ├── base.py
│   │   │   │   ├── local.py
│   │   │   │   ├── autolab_cli.py
│   │   │   │   ├── mobius_cli.py
│   │   │   │   └── replay.py
│   │   │   ├── metrics/
│   │   │   │   ├── validate.py
│   │   │   │   ├── normalize.py
│   │   │   │   ├── decision.py
│   │   │   │   └── report.py
│   │   │   └── services/
│   │   │       ├── orchestrator.py
│   │   │       ├── auth_gate.py
│   │   │       └── artifact_store.py
│   │   └── tests/
│   │       ├── test_studies.py
│   │       ├── test_policy.py
│   │       ├── test_budget.py
│   │       ├── test_sponsor_import.py
│   │       ├── test_mode_labels.py
│   │       └── fixtures/
│   └── web/
│       ├── package.json
│       ├── src/
│       │   ├── App.tsx
│       │   ├── api/client.ts
│       │   ├── hooks/useStudyEvents.ts
│       │   ├── components/
│       │   │   ├── TopNav.tsx
│       │   │   ├── GoalComposer.tsx
│       │   │   ├── ConnectionPanel.tsx
│       │   │   ├── BudgetGauge.tsx
│       │   │   ├── PolicySelector.tsx
│       │   │   ├── ResearchTimeline.tsx
│       │   │   ├── ExperimentGraph.tsx
│       │   │   ├── CandidateComparison.tsx
│       │   │   ├── EvidenceDrawer.tsx
│       │   │   ├── DecisionReason.tsx
│       │   │   ├── ProvenanceStrip.tsx
│       │   │   └── ReportExport.tsx
│       │   ├── pages/
│       │   │   ├── ResearchHome.tsx
│       │   │   ├── StudyPage.tsx
│       │   │   └── ReportPage.tsx
│       │   └── styles/
│       └── e2e/                          # Playwright
├── experiments/
│   └── frontier2026/
│       ├── research_plan.yaml            # existing
│       └── speculative/
│           ├── README.md
│           ├── manifest.yaml
│           ├── dataset_registry.yaml
│           ├── fetch_prompts.py
│           ├── split_registry.py
│           ├── bench_runner.py
│           ├── correctness.py
│           ├── evaluator.py
│           ├── fixed_baselines.py
│           ├── analysis.py
│           ├── tests/
│           └── hills/                    # or a scoped .autolab project
│               └── mobius-inference/
├── integrations/
│   ├── autolab/
│   │   ├── README.md
│   │   ├── import_run.py
│   │   └── fixtures/                     # sanitized, genuine CLI captures only
│   └── egbe/
│       └── README.md
├── scripts/
│   ├── demo_local.sh
│   ├── demo_autolab.sh
│   ├── verify_repo.sh
│   └── export_report.sh
└── .gitignore                            # ensure credentials, caches, run data ignored
~~~

The file layout is a target, not a mandate to create every empty file. Begin with the smallest vertical subset and add modules only as real complexity appears. Prefer a clean API over scattered helpers.

---

## 5. Research Data Layer: Real Inputs, No Toy Headline Dataset

### Flagship task

**Cross-domain speculative decoding efficiency on an open-weight Qwen model.**

Primary target: Qwen/Qwen3-8B.  
Pretrained draft/speculator: RedHatAI/Qwen3-8B-speculator.eagle3.  
Inference runtime: verified compatible vLLM build on a suitable Linux/CUDA machine.  
The draft and target must be proven compatible before use; record model revisions and runtime version. The Qwen3-8B target-only run is always the first baseline.

Official vLLM docs:
- https://docs.vllm.ai/projects/speculators/en/stable/user_guide/getting_started/
- https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/serve_vllm/

### Prompt workload sources

1. Code: https://huggingface.co/datasets/livecodebench/code_generation_lite
2. Math: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
3. Conversation: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

These are **inference prompt workloads**, not training data to claim a state-of-the-art reasoning result. Check each source's actual schema, license, allowed split, repository revision and current size before extraction. Do not guess field names.

### Data registry

For each source record:
- source URL and repo ID
- Hugging Face revision SHA and original split
- license / access restrictions
- extraction script commit
- allowed prompt fields and normalization rule
- full source checksum and selected prompt ID + content hashes
- prompt count, token count, prompt length buckets and domain
- train/dev/audit/sealed assignment
- cutoff / contamination considerations
- any rejected examples and why

### Three levels of evidence — essential

**Development:** visible to proposer and scoring policy. Use this for generating and tuning candidate settings.  
**Audit:** fixed independent subset, optionally queried by evidence-aware policy at a *counted* verification cost. Metric-greedy arm can use the same budget on another proposal; neither may peek at sealed final.  
**Sealed final:** not accessible to proposer, optimization agent or online decision policy; evaluated only after all controller arms and configurations are frozen.

Avoid random samples alone masquerading as distribution shift. Pre-register domain and length stress: e.g., development contains shorter conversation prompts; audit targets long conversations and reasoning; sealed contains code-heavy and length-shifted prompts. Document any overlap and run dedupe checks on text hashes.

**Tiny hackathon pilot:** 8–16 prompts per domain per visible split, 128 output-token cap where feasible. The small sample is an instrumentation demo, not a statistically supported frontier claim. Increase to substantial prompts and multiple seeds for paper-grade results.

**Data integrity assertions:**
- No prompt text hash occurs in two mutually exclusive splits.
- All source revisions pinned.
- All prompt token lengths recorded.
- No hidden answers or evaluator internals in model inputs.
- Sealed final cannot be requested through any normal API route.
- Audit calls are recorded and charged to the correct controller arm.
- Compare same ordered prompt set, model generation settings, warmup and hardware.

### License and sensitive data

Keep local cache and Hub downloads out of Git unless permitted. Commit only sample IDs, hashes, scripts and permitted minimal fixture examples. Do not publish full private evaluation examples, access tokens, user emails, or live leads.

---

## 6. Scientific Hypotheses: What Exactly Is Being Tested?

### Model-level hypothesis H1

With compatible EAGLE-3 speculation, some decode settings improve throughput relative to standard Qwen3-8B generation **while preserving exact outputs in matched deterministic greedy decoding**.

This is an engineering baseline, not an original discovery by itself.

### Research-policy hypothesis H2 — the flagship claim to test

A controller allowed to decide between **proposing a fresh configuration, replicating an uncertain run, or spending effort on an independent audit** can produce more reliable improvements at the SAME total research budget than a controller that simply promotes the highest visible development score.

This is testable even if speculative decoding offers no speedup on the selected hardware; honest negative results remain useful.

### Required control arms

| Arm | Decision rule | Allowed actions |
| --- | --- | --- |
| target_only | Fixed target autoregressive decoding | one baseline evaluation |
| fixed_spec | A preregistered fixed lookahead | one baseline evaluation |
| random | Sample allowed config from fixed search space | propose, run |
| tpe | Existing Möbius TPE over identical configs | propose, run |
| metric_greedy | Same candidate proposer as evidence-aware; maximize dev metric | propose, run |
| evidence_aware | Same proposer; additionally can replicate/audit to reduce decision risk | propose, run, replicate, audit |

Do not compare a code-editing AI arm to parameter-only TPE as if the search spaces were identical. Any code-editing arm must be labeled as *extended capability*, secondary to the core matched search-policy test.

### Allowed candidate knobs (start with supported ones only)

- speculative lookahead count, validated from vLLM installed flags;
- inference batch/concurrency levels;
- selected stable verifier options, only if actually supported by installed engine;
- optionally fixed prompt chunk size or request scheduling outside the model.

Never let policy arms modify target weights or the evaluator when the claim is lossless speculative acceleration. Engine restarts may be needed between configs; instrument their time and cost.

**Candidate search space must be pinned once**. Example proposed lookahead values: 2, 3, 4, 5, subject to actual engine support and memory. Do not assert values are supported until tested.

### Budget fairness

Set a common maximum scored-trial budget AND track GPU-seconds, wall time, LLM tokens, verifier calls, model server restart cost and dollars. Total research cost is not merely dollars for training. Pre-register how warmup and setup costs enter the comparison.

A pilot could do 1 target-only run, 1 fixed speculative run, and 2–3 proposals per policy arm. Full evidence requires more trials, repeated seeds, paired inputs and matched compute.

**Do NOT claim statistical significance from the hackathon pilot.**

---

## 7. Metrics and Gates — Exact Scientific Definitions

### Validity gate (must pass before promotion)

- The target model revision, tokenizer, generation options and engine version match baseline as planned.
- Deterministic greedy completion token IDs match target-only reference for every paired prompt for a **lossless comparison**. Use exact token IDs and output stop criteria, not only string-level normalized text. Explain unavoidable nondeterminism where applicable.
- No benchmark/evaluator tampering, split leakage or missing provenance.
- No NaN/inf, empty result, untrusted self-reported output, timeout without state or malformed source run payload.
- Same GPU class, driver, memory config and load regime for comparable throughput claims.
- No disallowed p95 latency / memory regression.

If any gate fails: mark INVALID or NEEDS_INVESTIGATION; **never** mark as better due to its apparently high throughput.

### Primary real-system objective

For candidates passing the correctness gate, measure throughput improvement on development workloads:

speedup = median_decode_tokens_per_second(candidate) / median_decode_tokens_per_second(target_only)

Evaluate actual service decode tokens/s (not GPU theoretical FLOPs), with explicit request concurrency and batch setup. Consider a balanced per-domain geometric mean speedup to avoid one large domain dominating. For short sequences, report full end-to-end latency separately from decode throughput.

### Required displayed metrics

- decode output tokens/sec (median and per-domain)
- p50 and p95 end-to-end latency, ms
- time to first token, ms
- acceptance rate and accepted tokens per verification, if engine exposes reliable counters
- peak VRAM GB
- completion correctness: token ID equality / compared prompt count
- total output tokens and prompt tokens
- GPU-seconds, serving wall-time, warmup overhead
- LLM reasoning token count and cost, only when sourced and known
- observed/estimated total cost with separate components and unknowns shown as null
- failures, timeouts, retry counts
- baseline vs current valid best
- speedup confidence estimate / replicate variance
- audit pass/fail with domain-specific reason

Do not convert unspecified costs to $0.00, or infer total dollar cost merely from time. Display "not available" until a source exists.

### Experiment-level research metrics

- best VALID held-out objective at matched all-in budget (post-freeze only)
- area under best-so-far **development** metric vs cumulative spent research resources
- false-promotion rate: number of promoted dev winners regressing on sealed final / total promoted winners, computed retrospectively after unsealing
- audit/replication rate; number of experiments until first valid gain
- regret relative to best fully evaluated candidate, if known
- dev-to-final effect correlation (retrospective only)
- proportion of invalid or irreproducible trials
- proportion of run steps with complete provenance

### Proposed initial decision heuristic

This is a transparent V0 heuristic, NOT a proven optimal value-of-information algorithm:

1. If no valid baseline: RUN_BASELINE.
2. If candidate is invalid: REJECT + reason.
3. If apparent dev gain is within measured variation: REPLICATE (subject to budget).
4. If gain is large but concentrated in one domain or a long-tail stress case: AUDIT relevant independent slice (subject to budget).
5. If gain survives predefined checks and available audit: PROMOTE best valid.
6. If expected value of another attempt is small relative to remaining spend: STOP.
7. Else: PROPOSE_NEXT.

Return structured rationale including evidence IDs and explicit "uncertainty" notes. Never expose a post-hoc hidden test score while the policy is still choosing actions.

For later paper work, evaluate calibrated VOI acquisition policies:

VOI(action) = expected reduction in downstream decision error
              - lambda * expected total resource cost
              - mu * invalid/overfitting risk.

Do not invent expected improvements. V0 can use confidence intervals or risk heuristics derived from observed replicates, with formula and limitations recorded.

---

## 8. Typed Contracts: Persist Evidence, Not Just Pretty Numbers

Every persisted run must contain immutable source metadata and evidence class.

### Run lifecycle state machine

~~~text
DRAFT → QUEUED → RUNNING → SUCCEEDED → VALIDATING
                                   └→ FAILED
                                   └→ TIMED_OUT
VALIDATING → VALID / INVALID / NEEDS_AUDIT
VALID → PROMOTED / NOT_PROMOTED / AUDITED
NEEDS_AUDIT → AUDITED → PROMOTED / REJECTED
Any nonterminal → CANCEL_REQUESTED → CANCELLED
~~~

Transitions must be explicit and tested. Re-import must not duplicate event history or cost reservations. Failure statuses are first-class.

### Pydantic / JSON response shape (design contract)

~~~json
{
  "id": "run_6f0d",
  "study_id": "study_20261122",
  "policy_arm": "evidence_aware",
  "candidate_id": "cand_0004",
  "parent_candidate_id": "cand_0002",
  "hypothesis": "A shorter lookahead reduces overhead on short code prompts",
  "candidate_config": {
    "lookahead_tokens": 3,
    "max_output_tokens": 128,
    "request_concurrency": 1
  },
  "status": "validating",
  "mode": "autolab",
  "evidence_class": "observed",
  "source": {
    "provider": "autolab",
    "project_slug": null,
    "run_id": null,
    "run_url": null
  },
  "repro": {
    "git_sha": null,
    "target_model_revision": null,
    "draft_model_revision": null,
    "evaluator_sha256": null,
    "hill_version": null,
    "dataset_split_sha256": null,
    "engine_version": null,
    "gpu_model": null,
    "seed": 2026
  },
  "metrics": {
    "throughput_tokens_per_sec": null,
    "speedup_vs_target": null,
    "p95_latency_ms": null,
    "output_token_match_rate": null,
    "gpu_seconds": null,
    "gpu_cost_usd": null
  },
  "validation": {
    "exact_token_match": null,
    "heldout_accessed_online": false,
    "gates_passed": [],
    "gates_failed": []
  },
  "decision": {
    "action": "pending",
    "reason": null,
    "evidence_run_ids": []
  }
}
~~~

This is **an example schema with null placeholders**, not a recorded result. Never show it as a successful run.

### Database tables (proposed)

1. projects: id, name, created_at, readonly_public.
2. studies: id, project_id, goal, template_id, state, active_policy, created_at, frozen_eval_hash, mode.
3. dataset_snapshots: id, source_id, revision, split_name, hash, license, role.
4. candidate_proposals: id, study_id, policy_arm, parent_id, description, rationale, config_json, patch_sha, proposer_id.
5. experiments: id, candidate_id, study_id, policy_arm, source_run_id, source_platform, state, timings, failure.
6. metric_observations: experiment_id, name, value, unit, higher_is_better, domain, sample_size, evidence_class, observed_at.
7. verification_checks: experiment_id, check_name, status, evidence_hash, details.
8. decision_events: id, study_id, experiment_id, action, rationale, evidence_ids, controller_version, time.
9. budgets: study_id, approved_cap, reserved_spend, observed_spend, estimated_spend, currency.
10. budget_events: id, study_id, source, reserve/release/spend, amount, source_run_id, timestamp.
11. audit_events: append-only event_id, actor, action, json_payload_hash, created_at.
12. imported_source_events: platform, external_id, payload_hash, first_seen_at — unique platform+external_id+semantic_event.

Use real foreign keys, indexes on study_id/state/timestamp, SQLite transactions and WAL. Keep migrations ordered; no unbounded full-table scans in every SSE poll.

### Budget control

- Separate planned maximum, committed/reserved spend, actual observed spend, and unknown spend.
- Starting a remote study with no explicit approved cap is forbidden.
- Reserve estimated maximum **before** dispatching a paid run; reject if reservation would exceed cap.
- Verify actual billed amount after run, reconcile reservation and record the source.
- Costs that cannot be known precisely must be shown as estimated, with source and uncertainty.
- Cost exhaustion stops proposal scheduling even if an external experiment is still finishing.
- Provide stop/cancel action and explain if external cancellation is best effort.

---

## 9. Backend API — First Public Contract

Keep API versioned under /api/v1. Examples:

| Method | Route | Purpose |
| --- | --- | --- |
| GET | /api/v1/health | backend availability / version |
| GET | /api/v1/templates | available study types, data sources, model choices |
| GET | /api/v1/connections | Autolab/Egbe/compute readiness **without leaking secrets** |
| POST | /api/v1/studies | create a DRAFT study |
| GET | /api/v1/studies | list studies |
| GET | /api/v1/studies/{id} | study, budget, current decision |
| POST | /api/v1/studies/{id}/approve-budget | explicit approval gate (local authenticated action only) |
| POST | /api/v1/studies/{id}/start | schedule baseline/next run, idempotent |
| POST | /api/v1/studies/{id}/pause | stop new work |
| POST | /api/v1/studies/{id}/resume | resume within approved limits |
| POST | /api/v1/studies/{id}/cancel | stop/reconcile pending work |
| GET | /api/v1/studies/{id}/experiments | paginated timeline and filters |
| GET | /api/v1/experiments/{id} | detail, raw source, metrics, decision |
| GET | /api/v1/studies/{id}/events | Server-Sent Events with replay/resume |
| GET | /api/v1/studies/{id}/comparison | compare policy arms and baselines |
| GET | /api/v1/studies/{id}/report | factual Markdown report |
| GET | /api/v1/studies/{id}/export.json | machine-readable artifact |
| POST | /api/v1/import/autolab | privileged source-run synchronization |

Version all input/output Pydantic models and validation errors. Use consistent JSON names and explicit enums. Use idempotency keys for POST writes where retries can duplicate paid work.

### Example study creation request

~~~json
{
  "template_id": "qwen3_speculative_ood",
  "goal": "Improve serving throughput without changing greedy outputs",
  "mode": "local",
  "model_id": "Qwen/Qwen3-8B",
  "policy_arm": "evidence_aware",
  "budget": {
    "max_trials": 3,
    "approved_spend_usd": 0,
    "max_wall_seconds": 3600
  },
  "dataset_manifest": "experiments/frontier2026/speculative/dataset_registry.yaml",
  "notes": "Pilot only: no statistical significance claim"
}
~~~

Starting local mode must not silently rent a GPU. Starting Autolab mode must check connection, Hill hash, compute and cost gate before queueing.

### SSE event contract

~~~text
id: 128
event: experiment.validated
data: {"study_id":"study_20261122","experiment_id":"run_6f0d","valid":true,"source":"autolab"}
~~~

Event types: study.created, study.started, research.hypothesis, experiment.queued, experiment.started, experiment.metrics, experiment.validated, research.decision, budget.updated, study.paused, study.completed, source.error.

On reconnect, support Last-Event-ID or query "after_event_id". UI should fall back to periodic GET if SSE unsupported. No client control via SSE. Ensure all messages contain observed server truth; no optimistic "passed" before evaluation.

---

## 10. Möbius Rust CLI Integration — Precise Rules

First prove the existing CLI:

~~~bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p mobius-cli -- --help
cargo run -p mobius-cli -- ask --help
cargo run -p mobius-cli -- tell --help
~~~

Check actual binary package/crate names from Cargo.toml before finalizing invocation; do not assume.

Example of *intended* bridge, after building and pinning a study-specific config:

~~~bash
# These are illustrative; isolate each policy arm's HOME and cwd.
mobius ask --strategy tpe --metric throughput_tokens_per_sec --store sqlite
mobius tell --config '{"lookahead_tokens":3}' \
  --metrics '{"throughput_tokens_per_sec":118.2}' --store sqlite
~~~

**118.2 above is a schema-only illustration, NOT a measured throughput; never display it as one.**

Use subprocess argv arrays instead of shell string interpolation. The Rust CLI's state is under ~/.mobius; create an isolated ephemeral HOME per **study + policy arm** before invoking, and provide the relevant mobius.toml in that study working directory. Verify that dirs::home_dir actually respects this isolation on the current OS. If not, implement an explicit configurable state root behind a failing Rust test or use a dedicated container; no global history mingling.

Never call tell on a failed/timeout/invalid experiment, because tell marks status Success by design. Failures belong in authoritative app SQLite + audit log. Once a valid result is committed, derive numeric metric observations and import exactly once. The API must be able to restore the derived optimizer store from authoritative records if necessary.

For V0 **evidence-aware** decisions, implement the controller in Python as a typed policy wrapper. After it works and metrics justify it, consider implementing a Rust Strategy/Hook as a small tested extension. Do not block the demo rewriting agent.rs.

TPE and other search policies may require minimum warm-start observations; explicitly log how initial configurations were selected.

---

## 11. Autolab Integration — Official Surfaces, Not Imagined REST

Read official docs and installed CLI skill:
- https://docs.autolab.ai/install/
- https://docs.autolab.ai/guides/hills/
- https://docs.autolab.ai/guides/experiments/
- https://docs.autolab.ai/reference/cli/
- https://docs.autolab.ai/guides/coding-agents/

### Connection and preparation

Official workflow, only after the operator agrees to install and authorize:

~~~bash
# Review the official installer first; don't blindly pipe an uninspected script.
curl -fsSL https://app.autolab.ai/install.sh | sh
autolab --version
autolab login
autolab whoami
autolab install claude
~~~

Confirm the actual installed CLI release and help outputs. Do not log tokens or dump ~/.config/autolab/credentials.json.

### Frozen evaluation Hill

~~~bash
# Use separate scoped workspace with official Hill structure.
autolab hills new mobius-qwen3-inference
# Implement hill.yaml, eval.py, README.md, private/, examples/baseline/, tests/.
autolab hills check mobius-qwen3-inference
autolab hills eval examples/baseline -H mobius-qwen3-inference
autolab hills commit mobius-qwen3-inference -m "freeze inference evaluation v1"
autolab hills push mobius-qwen3-inference
~~~

The Hill evaluator uses an explicit eval contract, not an arbitrary exit-code check. The official docs describe eval(submission, *, final=False, **params). Read generated scaffold and tool-specific schema before implementing. Immutable evaluator version must be recorded as Hill hash. Private final data must be outside agent writable/visible area.

**Important:** The hill can be local and private; no need to make it public for the hack. The benchmark's proposed pass/fail criteria and metric direction must match hill.yaml exactly. Test a baseline *and* a deliberately broken candidate. Ensure no candidate can alter private data, scorer, labels, runtime, or outputs.

### Run a Climb

~~~bash
# Replace OWNER with authenticated user from autolab whoami.
autolab init --hill OWNER/mobius-qwen3-inference --max-cost 10
autolab status
autolab serve
autolab nodes
# Attach verified compute node or explicitly authorize an eligible rental.
autolab start
autolab open
~~~

**The $10 is an example maximum, not permission to spend.** Keep the budget gate in the app at zero until approved. The Autolab agent's maximum cost cap and compute rental expenses must both be checked; do not assume the cap covers all cost categories. The project may start paused; inspect settings and node assignment before calling start.

### Candidate submission modes

- **Code-locked candidate**: Claude Code or Möbius constructs the allowed patch; submit the snapshot with autolab submit -m "...". The Autolab agent must not alter supplied candidate code.
- **Idea-only**: autolab submit --nocode -m "..."; Autolab's own agent can write code. Label proposer_origin="autolab_agent", not "mobius_agent".
- **Soft submission**: autolab submit --soft only when allowing the agent to revise pre-written code is intentional, and mark provenance accordingly.

To isolate *Möbius's* decision value, use a consistent proposer/coder and evaluation for metric_greedy and evidence_aware arms. Do not silently swap coders, tools or available context.

### Read results

~~~bash
autolab log
autolab metrics ACTUAL_EXPERIMENT_ID --format json
autolab graph --format json
autolab open
~~~

The JSON metrics payload contains series, params and samples; inspect a **real** captured payload before writing parser mappings. Do not assume it is a flat dict or that the most recent series point represents the official final scored result. Preserve raw sanitized JSON and authoritative Hill verdict/signature if available. Handle a pending run without a score.

### Autolab adapter interface (new code)

~~~python
class AutolabAdapter(Protocol):
    async def readiness(self) -> dict: ...
    async def submit_candidate(self, study_id: str, candidate_id: str, mode: str) -> str: ...
    async def poll_run(self, external_run_id: str) -> dict: ...
    async def get_metrics(self, external_run_id: str) -> dict: ...
    async def cancel(self, external_run_id: str) -> bool: ...
~~~

Implementation may use official subprocess calls with argv and strict deadlines. Names are **internal proposed interfaces**, not official Autolab SDK definitions.

Adapter must:
- verify CLI available and auth present without exposing secret;
- capture stdout/stderr securely and redact auth material;
- use actual sponsor run IDs, not local guesses;
- reject schema mismatch and incomplete/missing metrics;
- idempotently import run+cost; consider CLI retry exhaustion;
- link to source dashboard;
- distinguish "official Hill validated" from "local precheck";
- not report a successful submission as a completed evaluation.

---

## 12. Inference Benchmark Runner: First Real Scientific Slice

The headline machine should be Linux with NVIDIA CUDA GPU sufficient for target Qwen3-8B, draft model, KV cache and requested batch. Measure actual GPU memory; don't infer universal VRAM requirements. Mac/Apple Silicon can serve as UI/orchestration, and may use a separate compatible inference engine for a **locally labeled non-comparable** test. Do not pretend vLLM CUDA performance reflects the Mac.

### Verified vLLM start commands

~~~bash
# On a verified compatible GPU machine, in separate sequential benchmark sessions.
# Run 1: regular generation
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 1

# Run 2: speculative decoding (from official speculator tutorial)
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 1 \
  --speculative-config '{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3"}'
~~~

Validate actual command flags against installed version. Set same environment/precision, sampling config, request concurrency and max output length, with server restarts as needed. Use explicit ports and check readiness; do not have both servers compete for GPU resources during matched latency measurement. The alternative documented shortcut is serving RedHatAI/Qwen3-8B-speculator.eagle3 directly, which loads the target model from its config.

### Runner requirements

Inputs: engine endpoint, model revision, prompt-manifest hash, generation config, concurrency, seed, warmup iterations, timed iterations.  
Outputs: per-request JSONL timings and token IDs, aggregate metrics, hardware and runtime metadata, full command, source hashes and exit code.  
Support baseline and candidate from exactly the same prompt IDs, ordered and shuffled via seeded stable schedule; record ordering.  
Warm up both configurations identically. Repeat runs; collect p50/p95 E2E, TTFT, throughput, token equivalence, memory telemetry if source available.  
If engine exposes accepted-token metrics, use actual counter names from installed version; never derive fake counters.

**Correctness testing:** For greedy generation use temperature=0, same system template, stop tokens, maximum generation and model revisions. Compare generated token IDs. If non-determinism prevents exact equality, classify the run as not establishing lossless equivalence; investigate version and kernel causes. For stochastic sampling, output identity is not the right correctness test—use a separate distribution-preserving protocol only if time permits.

### Experiment candidate contracts

~~~json
{
  "candidate_id": "cand_0005",
  "parent_id": "cand_0003",
  "research_arm": "evidence_aware",
  "hypothesis": "Lookahead 3 will reduce wasted verification on short prompts",
  "proposal": {
    "lookahead_tokens": 3,
    "request_concurrency": 1
  },
  "rationale": "Prior chat-only gain was uncertain on long code prompts",
  "verification_request": "audit_code_long",
  "expected_direction": "increase_throughput",
  "estimated_max_cost_usd": null
}
~~~

Do not submit arbitrary shell commands from candidate JSON. Validate values against pre-registered allowlist.

### What to do when GPU is unavailable

- Complete UI/backend/local evaluator and tests with the real engine abstraction.
- Capture/import genuine Autolab runs from any connected account if available; label them OBSERVED.
- Offer **clearly labeled** REPLAY of authentic previously recorded run traces. Never synthesize favorable sponsor metrics.
- Offer LOCAL CHECK mode on an actually available small workload / model that uses fixed independent evaluator, labeled non-comparable to Qwen3 CUDA benchmark.
- Show no "speedup" card if no valid baseline exists.
- Present external blocker precisely in UI: no GPU node, auth missing, cost approval missing, model unavailable, or compatible speculator missing.
- Do not switch silently to UCI shopper classification as flagship. That dataset is only plumbing smoke test.

---

## 13. Research Controller: Decisions That Make Möbius Different

### Interfaces

~~~python
class ResearchPolicy(Protocol):
    def next_action(self, state: "StudyState") -> "ResearchAction": ...

class ResearchAction(BaseModel):
    kind: Literal[
        "run_baseline", "propose_new", "execute_candidate",
        "replicate", "audit", "promote", "reject", "stop"
    ]
    target_candidate_id: str | None = None
    rationale: str
    evidence_ids: list[str]
    estimated_resource_cost: float | None = None
~~~

Controller must never receive the sealed final results during search.

### Metric-greedy arm

Given same proposer and cost budget, choose next config maximizing the current dev score; only enforce mandatory validity. This is not a malicious strawman; pre-register objective and tie-breakers, and follow identical budget/compute constraints.

### TPE arm

Call existing Möbius ask with strategy tpe and current named metric; tell only validated numeric outcomes. Freeze param search space and warm-start rules. Limit same number of scored trials and total cost as the compared arm.

### Evidence-aware arm

Use a simple deterministic stateful heuristic first:
- Check if improvement is larger than prior run noise (replicate if borderline).
- Check whether a proposed improvement is domain-specific (audit OOD-like visible audit split if suspicious).
- Check data leakage/metric change/command change (reject).
- Reserve budget for independent audit before promote.
- If costs exceed remaining resources, stop with honest uncertainty.
- Update a transparent confidence flag, not an invented posterior probability.

Log every decision reason and input evidence IDs, expose it in UI. LLM can help formulate explanations, but cannot change controller rules or mark unscored experiments valid.

### Hypothesis generation (Claude Code) layer

Use structured-output prompts that include:
- task objective and hard constraints;
- previous candidate configs, their measured outcomes, invalidity reasons;
- current allowed search space;
- per-domain uncertainties;
- remaining budget and allowed actions.

Output candidate JSON and **hypothesis**, never execute code straight from text. If JSON invalid, ask for one repair or reject. Cache proposer decisions and record model/provider/version and prompt hash. Do not give the agent held-out prompt contents or final scores.

### When to compare the policies

Run matched policy trials after an identical warm-up baseline. Use source run IDs for each arm. Match hardware, models, prompt order, evaluation rules and cost accounting. If budgets differ due to rejected trials, report that explicitly.

One tiny demo trace can show a functioning policy, but cannot prove policy superiority. Separate "pilot evidence" from "research claims".

---

## 14. Frontend Product Design and Interactions

Build an elegant, functional engineering product, not a dashboard theme pack.

### Information architecture

1. **Research Home** — goal composer, templates, recent studies, connection health.
2. **Study Workspace** — objective, status, active research loop, budget, timeline, comparison charts.
3. **Evidence Drawer** — experiment source, diff/config, metadata, metrics, logs, accept/reject reason.
4. **Report** — markdown with exact reproducibility info, downloadable JSON and link to sponsor run.
5. **Integrations** — actual connection readiness, never credentials.

### Design guidance

- Desktop-first, responsive to phone for demo browsing.
- Visual identity: monochrome/dark neutral with one restrained violet/indigo accent around the Möbius infinity icon. Accessible WCAG contrast, readable monospaced metrics.
- Hierarchy: research question prominent, status and source labels unmistakable, beautiful but quiet charts.
- Hypothesis timeline resembling Git commit lineage: each node shows state, cost, observed score and decision. The researcher should immediately see why node B branched from node A.
- Distinguish measured/estimated/replayed/synthetic/draft with persistent plain-language text, not color alone.
- Critical metric values show units, samples, provenance and their comparison scope.
- Empty states guide next action without invented data.
- Show loading, disconnected, permission-denied, invalid benchmark, completed and failed modes.
- All advertised buttons work. No inert "launch", "approve", "copy", "export", "replay" or "disconnect" interactions.

### Specific components and acceptance

**GoalComposer:** template selector; model; primary metric; budget trial/cost; three modes; exact intent summary. Disable remote start without authenticated connection & budget approval.

**ResearchTimeline:** live statuses and node links; click to open evidence. Candidate lines connect parents.

**DecisionReason:** "Why Möbius did this" panel with action, cited run IDs, failed gates, uncertainty and next action.

**ExperimentGraph:** best VALID throughput vs cumulative trial cost; allow toggle raw candidate score. Missing values not plotted as zero.

**CandidateComparison:** baseline and two candidate columns; meaningful metric alignment; highlight latency/VRAM tradeoffs and validity gate.

**BudgetGauge:** approved/reserved/observed/unknown; no progress pretending unknown cost is free.

**SourceBadge:** OBSERVED (real), REPLAY (recorded), LOCAL (actual local test), FIXTURE (tests only), PENDING; never overload one color-coded badge as evidence.

**ReportExport:** generate source-linked factual summary, reproducibility JSON, and safe markdown including negative results.

### Client/server behavior

Use TanStack Query for fetch/cache, SSE for incremental events, retry with backoff and event ID. All mutations go through the API; never implement pretend client-only status transitions. Validate user input and escape untrusted raw logs/markdown. Avoid rendering raw HTML from agent output.

### Suggested screenshots for judges

- Main research goal before run.
- Live experiment timeline with Autolab source link.
- A specific evidence drawer showing a replication or invalidity check.
- Chart comparing actual baseline and chosen valid candidate.
- Report export with hardware, hashes, costs, and next experiment.

Do not create fabricated result screenshots. Visual mockups are acceptable only with clear "concept" or "demo fixture" label.

---

## 15. Egbe Integration: Useful and Small

Egbe is **not** part of the scientific scoring path. Build it after one end-to-end ML experiment is working.

Official docs:
- https://docs.egbe.ai/start/quickstart/
- https://docs.egbe.ai/start/connect-agent/
- https://docs.egbe.ai/

Connection is via business Dashboard → Connect your agent → Any AI agent → Copy instructions. In supported Claude Code pilot setups, a direct plugin connection may be offered. The per-business instructions are private and must NEVER be checked in, placed in issue comments, or displayed in the UI.

Persistent connected workspace keeps installed skills, editable playbooks, and egbe-docs/. Credentials go in git-ignored .env.local. Check actual enabled capabilities and spending allowances. Egbe does not require inventing an "egbe" CLI command.

First Egbe task after connection:
- Research the strongest ICP for Möbius Lab: independent ML researchers vs engineering teams running inference.
- Source five genuine problems/objections, two positioning variants, and a proposed landing page.
- Draft one acquisition experiment with metric qualified_signups / eligible_unique_visitors.
- If authorized, publish a landing page and instrument actual unique visits, CTA clicks, qualified signups and activation. Minimum traffic thresholds are not assumed.
- Do not email contacts, buy ads, spend credits, scrape private leads, or launch a public site without explicit approval.
- Do not merge campaign metrics with ML throughput into one invented scalar objective. They are separate research domains sharing a generic evidence ledger.

For Sundai pitch, showing a verified Egbe research output or landing draft is enough if no real traffic exists. Never claim users signed up unless verified analytics record them.

---

## 16. Test-Driven Implementation Plan

No implementation phase is "done" until its acceptance tests pass. Prefer failing tests first for changes in behavior. Work incrementally and commit after each gate.

### Gate 0 — Repository health and architecture read

Actions:
1. git status, branch, log; ensure branch correct.
2. Inspect CLAUDE.md, README, research docs and implementation interfaces.
3. Run cargo fmt check, cargo clippy, cargo test workspace, existing hello-experiment; capture literal outputs.
4. Verify Node/Python/Rust versions and installed packages.
5. Identify source paths, CLI invariants, current CI, operating system and GPU/Autolab readiness.
6. Write docs/MOBIUS_LAB_ARCHITECTURE.md with 5–10 deliberate decisions and risk register.

Deliverable: a concise repo audit with exact file:line evidence and explicit blockers.

Proposed commit: docs(mobius-lab): record architecture and verified baseline.

### Gate 1 — Typed scientific contract and local API

Actions:
1. Pydantic study/candidate/run/metric/decision/budget schemas.
2. SQLite tables and idempotent writes.
3. Implement create/read/list study and valid state transitions.
4. Add mode-specific connection checks and budget gates.
5. Include three deterministic local API fixtures for contract tests but label source=FIXTURE.
6. Implement a tiny local evaluator of a **real** local workload if available; no fake CUDA performance.
7. SSE event stream with replay.
8. Test missing/invalid metrics, duplicate imports, invalid transitions, overlapping spend, stop/cancel.

Acceptance:
- GET /health real version.
- POST study creates DRAFT and persists after backend restart.
- A local evaluator run gets source=LOCAL.
- A fixture is distinguishable from actual run and forbidden from a measured performance claim.
- Budget gating prevents remote paid dispatch.
- Retry of same POST does not duplicate run or charge.

Proposed commit: feat(lab-api): add typed study ledger and event stream.

### Gate 2 — Research controller and Rust bridge

Actions:
1. Define ResearchPolicy and ResearchAction.
2. Implement baseline, metric-greedy, TPE bridge and evidence-aware heuristic.
3. Create per-arm isolated Möbius state; prove ask/tell roundtrip with arbitrary metric, not f1 alias.
4. Verify identity and idempotency across import/restoration.
5. Record why each action occurs.
6. Test deterministic next decision from fixed observed inputs and fixed seed.
7. Test candidate regression, audit pass/fail, unscored outcome, budget exhaustion, input leakage.

Acceptance:
- Same known input state gives same deterministic policy action.
- Invalid metrics never get into Rust successful-history store.
- No policy reads sealed final.
- Same source run imported twice produces one experiment/charge.
- A real complete local loop: baseline → candidate → validate → decision → report.

Proposed commit: feat(lab-controller): add evidence-aware experiment selection.

### Gate 3 — Real data + Qwen inference benchmark

Actions:
1. Create dataset registry, prompt downloader and immutable split manifest.
2. Write portable benchmark runner with target-only mode.
3. Add exact-token correctness, p95 latency, throughput and telemetry collection.
4. Record model/runtime/hardware hashes.
5. Run local smoke checks; execute actual GPU baseline only if compute/auth approved.
6. Run one fixed speculative candidate on same device and compare.
7. Add source artifacts and explicit status/unknowns.

Acceptance:
- No prompt overlap; split hashes deterministic.
- Baseline and candidate have identical model revisions and prompt IDs.
- Every displayed measurement has a real run artifact and source.
- Correctness failure forbids promoting throughput.
- Full GPU run reproducible from documented command (when GPU available).

Proposed commit: feat(research-bench): add pinned Qwen speculative decoding evaluation.

### Gate 4 — Autolab sponsor integration

Actions:
1. Read installed CLI, authenticate with human, verify account and compute.
2. Author Hill evaluator; prove check, baseline and deliberately broken candidate.
3. Commit Hill version and keep scorer/held-out private.
4. Configure explicit climb cost cap, compute, auto-agent autonomy.
5. Submit candidate, observe real completion, retrieve metrics and source reference.
6. Import sanitized JSON, numeric metrics and provenance.
7. Compare imported sponsor record against direct CLI output.

Acceptance:
- Captured external ID, Hill hash, score provenance and source link.
- Run is marked OBSERVED only after verified metric source.
- Parser rejects invalid/missing/NaN, duplicate, incomplete runs.
- No outside spend without approval.
- Correct auto-agent vs Möbius proposer source labels.

Proposed commit: feat(autolab): integrate frozen Hill and authentic experiment evidence.

### Gate 5 — Production-quality web interface

Actions:
1. Build pages/components above, with polished no-run state first.
2. Connect API; all buttons backed by real mutations.
3. SSE event updates and reconnect.
4. Charts render only valid unit-correct metrics.
5. Evidence drawer and comparison view.
6. Markdown/JSON export.
7. Screenshot/e2e tests at desktop and phone breakpoints.
8. Improve visual quality after flows work, not before.

Acceptance:
- Can create/inspect/stop a local study without terminal.
- Read-only replay of real runs works without cloud auth, clearly labeled.
- API error surfaces are actionable.
- No hardcoded metric cards, placeholder success states, or dead buttons.
- All results match backend evidence.

Proposed commit: feat(mobius-lab-web): build evidence-focused study UI.

### Gate 6 — Egbe and demo readiness

Actions:
1. Connect privately to Egbe with actual installed instructions (human approval).
2. Research audience and prepare approved launch copy.
3. Optionally instrument real landing page, no fake funnel.
4. E2E smoke check full pitch flow with an actual experiment or labeled replay.
5. Write runbook, demo script, reproducibility report and known limitations.

Proposed commit: docs(demo): add live demo runbook and research report.

---

## 17. Unit, Integration, and Scientific Test Matrix

### Rust regression baseline

- cargo fmt --all --check
- cargo clippy --workspace --all-targets -- -D warnings
- cargo test --workspace
- no unrelated baseline failures introduced; report existing failures distinctly.

### Backend contract tests

- Creation, list/read, valid transitions, idempotent starts and imports.
- Reject invalid JSON, unknown dataset revisions, missing model, unapproved spend.
- Budget reservations atomic; concurrent requests cannot exceed cap.
- Evidence class never silently switches from replay/fixture to observed.
- Cancel releases future reservations while preserving actual spend.
- SSE order monotonic, event replay works.
- Export after restart is identical.

### Science tests

- Prompt hashes stable; no split overlap.
- Uniform settings for matched runs.
- Candidate changes cannot alter evaluator or sealed inputs.
- Token ID equality calculation checks complete generation; mismatches produce failure.
- Non-finite metrics rejected; no zero-fill on missing data.
- No success metrics on failed/timeout job.
- Variation across replicates handled transparently.
- Audit requests counted against policy budget.
- Sealed final not available in any online policy data structure.

### Sponsor adapter tests

- CLI missing / login missing / connection lost / zero credit.
- Schema variants and missing metrics; timeout/pending/failed.
- Multiple metric series; correct final score extraction.
- Duplicate import and out-of-order events.
- External source link and hashes stored.
- No secrets in logs, exports or error reports.

### Web tests

- Initial empty state and explicit "not connected" state.
- Budget approve / start / stop buttons functional.
- Evidence drawer shows correct source and decision.
- Failed run cannot appear as best valid candidate.
- Graph no fabricated samples.
- Narrow screen UI works.
- Keyboard controls and basic accessibility semantics.

### Research reproducibility checks

- Pin model and dataset source revisions; evaluator Hill hash, code SHA and runtime.
- Re-run same config/split/seed at least twice in pilot.
- Record warmup and GPU load.
- Freeze policy before sealed test.
- Negative result documented; no statistical "significant" claim without repeated rigorous analysis.

---

## 18. Security and Responsible Automation

**Threat model:** untrusted agent-generated candidate code, hidden benchmark contamination, malicious prompt content, external spend, secret leaks, and misleading demo evidence.

Controls:
- Candidate proposer may only output allowlisted settings or patch files in an isolated candidate directory.
- Reject absolute/parent-traversal paths, symlink escapes, executable binaries and modification of scorer/Hill or private data.
- Candidate execution must run in sandbox/container/Autolab compute with no real project secrets injected. Network disabled during hidden eval where possible.
- No arbitrary user shell commands from the public UI.
- APIs binding to 0.0.0.0 or public hosting must have authentication on write/admin routes and reasonable rate limits. Demo may remain localhost-only.
- CORS narrow; never expose local files, tokens, environment variables or private logs.
- Protect and .gitignore .env.local, .autolab auth, data cache, large model artifacts, private/ hidden test, SQLite personal run database and screenshots containing access keys.
- No email/ad/hosting publication via Egbe without approval.
- Third-party code/tool versions and licenses recorded before external release.
- Agent must not self-approve spending, silently change metric constraints or override INVALID verdict.

---

## 19. How to Run It Locally (Target Commands, Implement and Verify)

These commands are desired developer experience, not a claim the paths already exist:

~~~bash
# Repository checks
git status --short --branch
cargo test --workspace

# Install API
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn mobius_lab.main:app --reload --host 127.0.0.1 --port 8000

# In another terminal
cd apps/web
npm install
npm run dev

# Real local API health
curl -fsS http://127.0.0.1:8000/api/v1/health

# Tests
cd apps/api && python -m pytest -q
cd apps/web && npm test -- --run
cd apps/web && npm run build
~~~

Create scripts/demo_local.sh that checks dependencies, initializes the database, runs API/UI with correctly scoped workspace state and opens the proper local URL. Do NOT use hardcoded demo values for measured outputs.

Create scripts/demo_autolab.sh that performs read-only prechecks and prints next authorized commands. It must never spend or publish automatically when the operator just runs the script.

---

## 20. Hackathon Demonstration: 90-Second Judge Script

**0–15 s:** "Everyone automates ML experiments; Möbius automates the judgment of which evidence is convincing." Enter Qwen inference goal, show locked model and limits.

**15–30 s:** Show two research policies with identical budget: conventional metric-driven vs evidence-aware. Highlight the research action menu: propose, replicate, audit, stop.

**30–50 s:** Launch or show one real Autolab-scored run and the linked immutable Hill (if available). Explain where model, dataset and scorer hashes come from. For replay, say **recorded real run replay**.

**50–70 s:** Click a candidate whose speedup claim is uncertain; reveal its domain breakdown, correctness checks and why Möbius requests an audit or rejects it. Do NOT guarantee a negative example occurs. Use actual outcome.

**70–85 s:** Show evidence chart: best valid result vs total spend and side-by-side candidate comparisons. Explain why a high proxy metric is not automatically a trusted improvement.

**85–90 s:** Export reproducibility report: hypothesis, trials, source hashes, verification, cost and next question.

Optional final sentence: "Autolab executes. Hugging Face provides the open models and prompts. Möbius decides what to believe. Egbe helps share the result with researchers."

### Judge questions and grounded answers

- **"Is this just Autolab?"** Autolab runs and scores experiments; Möbius controls budget allocation and evidence decisions across recorded runs and can require replication or audit. Show the distinct decision trace.
- **"What did the agent actually learn?"** Only claim changes visibly supported by run history. It may learn which settings worked and how uncertain their benefits are; the current prototype does not modify its fundamental algorithm automatically.
- **"How do you stop reward hacking?"** Pinned scorer, independent evaluator, allowlisted candidate changes, audit split, sealed final test.
- **"Where is the novelty?"** Proposed meta-research policy for evidence calibration and selection under budget; needs real multi-task validation, not claimed SOTA.
- **"Can it improve anything?"** Architecture is domain-agnostic; this demo supports one real inference workload. Multi-domain transfer is future work, not tested.
- **"Did it improve throughput?"** Answer only with actual artifact-linked measured numbers, or say the evidence is still incomplete.

---

## 21. Exact Contingency Ladder and Non-Negotiables

**A. Real GPU + Autolab authenticated** → best demo: live queue, authentic source metrics, compared candidate, evidence decision, UI report.

**B. GPU access blocked but prior authentic scored runs available** → import those verified runs, run controller locally, replay source-linked events, disclose replay.

**C. No sponsor run, but local real inference possible** → actual local benchmark through identical study/decision path; explicitly note model/hardware differ, do not claim GPU/Autolab speedups.

**D. No runnable model** → fully functional control-plane UI + deterministic contract fixtures, labeled "simulation/fixture"; show only that the workflow works. Never claim the benchmark hypothesis was proven.

If the original benchmark is too expensive, **do not** replace the hidden outcome with a made-up score. Prefer a smaller real inference workload and disclose limitations.

### Non-negotiable acceptance before presenting as real research

- A source-linked baseline and at least one real candidate evaluation, or an explicit limitation statement.
- Pass/fail on independent correctness and provenance gates.
- At least one evidence-aware decision from actual observed metrics.
- All research costs/unknowns displayed honestly.
- No credentials or hidden data in published artifacts.
- No claim of superior research policy without matched-arm evidence.
- Honest explanation of what is implemented versus proposed.

---

## 22. Schedule: Prioritize a Thin Demo Over a Grand Rewrite

This is a rapid, AI-assisted hackathon implementation, not a time guarantee.

1. **First sprint:** read-only repo audit, tests, contract/API, minimal end-to-end local run.
2. **Second sprint:** UI connected to real API, research state transitions, result evidence and report.
3. **Third sprint:** freeze Hill and connect genuine Autolab scoring, if credentials/compute exist.
4. **Final sprint:** a recorded reliable demo flow, correctness audit and only then optional Egbe positioning.

An AI coding agent can scaffold major UI/backend pieces in roughly 1–2 focused hours, but working infrastructure, real GPU benchmarking, cost accounting and research validation may take substantially longer. Do not pretend they are one prompt away. If time is short, **defer Egbe, broader RL, and fancy visual polish** to preserve the honest evidence loop.

Parallel subagents can inspect UI, sponsor documentation and tests in read-only mode, but one lead agent must own architecture, shared schemas, budgets and commits. Avoid concurrent writes to the same files.

---

## 23. Commit Discipline and Handoff Reporting

After each small verified vertical slice:
1. git diff --check, relevant unit tests and existing Rust tests.
2. Review changed files for secrets and unrelated refactoring.
3. Commit only intentional changes with precise conventional commit message.
4. Update the runbook with exact commands and observed results.
5. Report DONE / PARTIAL / BLOCKED per gate, not vague confidence.

Suggested commits:
- docs(mobius-lab): record architecture baseline and execution assumptions
- test(lab): add state, budget and evaluator contract tests
- feat(lab-api): add study ledger and SSE
- feat(lab): implement controller and Möbius ask/tell bridge
- feat(frontier-eval): add pinned Qwen inference evaluation
- feat(autolab): import authenticated scored runs
- feat(lab-web): build research workspace and evidence panels
- test(e2e): cover run, inspect, compare and export flows
- docs(demo): add verified live demo and reproducibility runbook

**After every milestone, report:**
- files changed and why;
- exact commands executed and outcomes;
- source run IDs and URLs, if available;
- spending/approval status;
- gaps and failed tests;
- one next recommended task.

Do not open a non-draft PR or merge into main until the user reviews the outcome. Do not delete old project work for convenience.

---

## 24. Final Deliverables Checklist

- [ ] Existing Rust project revalidated and unchanged or narrowly patched.
- [ ] Real API + database with study/run lifecycle and hard cost gates.
- [ ] Functional frontend at localhost; all controls connected.
- [ ] Baseline, metric-greedy, TPE and evidence-aware policy decision traces.
- [ ] Frozen prompt/data registry with split hashes and no leakage.
- [ ] Genuine Qwen inference baseline on verified hardware (or explicit blocker).
- [ ] Genuine candidate experiment, correctness gate and measured performance.
- [ ] Autolab Hill frozen and checked (if sponsored access available).
- [ ] Authenticated Autolab results imported with source run IDs (if available).
- [ ] Source evidence and raw sanitized metrics link for every reported number.
- [ ] Replay mode clearly distinguished from live and fixture modes.
- [ ] One valid decision to promote, reject, replicate, audit, or stop, with evidence.
- [ ] Exported Markdown and JSON research report, including negative results.
- [ ] Frontend build, backend tests and Rust regression suite passing.
- [ ] Egbe audience research/publishing optional, only after approval.
- [ ] 90-second demo script and exact reproducibility commands.
- [ ] No secret leaks, unapproved spend, fabricated scores or hidden final access.

---

## 25. First Message to Paste Into Claude Code

You are implementing Möbius Lab on the existing lonexreb/mobius repository, branch revival/sundai-recursive-experiments-2026. Read docs/MOBIUS_LAB_CLAUDE_CODE_HANDOFF.md in full, then CLAUDE.md, docs/MOBIUS_FRONTIER_RESEARCH_NOV_2026.md, docs/SUNDAI_SPONSOR_INTEGRATION.md, experiments/frontier2026/research_plan.yaml, crates/mobius-claw/src/agent.rs, crates/mobius-cli/src/main.rs, crates/mobius-cli/src/commands/ask.rs and tell.rs, and crates/mobius-core/src/compute.rs.

**FIRST ACTION = READ-ONLY DISCOVERY AND BASELINE.**
Run git status, cargo fmt --all --check, cargo clippy --workspace --all-targets -- -D warnings, cargo test --workspace, and existing CLI help. Report concrete results and exact relevant file:line locations; do not assume prior 161 tests are still passing. Check hardware, Rust/Python/Node and whether Autolab is installed and logged in; do not log secrets.

**THEN IMPLEMENT THE SMALLEST WORKING VERTICAL SLICE, NOT ALL PHASES AT ONCE:**
- FastAPI schemas and SQLite research ledger for a study, baseline, candidate, metrics, evidence and decision.
- Per-study cost approval gate defaulting to $0, with tests for failure/duplicate runs.
- One local independent evaluator with real measurable output, or an honest unavailable state if no model is runnable.
- Möbius ask/tell bridge using named throughput metrics; isolate global ~/.mobius storage. Do not call "agent" and pretend it supports non-f1 metrics.
- Evidence-aware heuristic making one explainable, persisted decision.
- A React/Vite UI showing goal composer, experiment timeline, evidence drawer, budget, compare chart and export. Connect it to REAL API outputs; no inert buttons or fabricated statistics.
- Run tests and give me exact localhost start commands, changed files, commit IDs and a screenshot/evidence if available.

Once this vertical slice is running, proceed to the **Qwen3-8B / EAGLE3 Autolab Hill** only if GPU, credentials and costs have been explicitly authorized. Read the installed Autolab CLI/skills to confirm actual commands, evaluator and run JSON schemas. Import one real scored baseline and one candidate; require exact greedy-output equivalence to claim lossless acceleration. Never manufacture results, fail open on missing metrics, or submit spending/public outreach without approval.

Keep main untouched. Make small logical commits after each passing gate. Prioritize a genuine reliable demo over full generalization or simulated SOTA.

---

## 26. Official and Project References (Verify Versions Before Implementation)

### Current relevant public sources
- Autolab overview: https://docs.autolab.ai/
- Autolab Hill authoring: https://docs.autolab.ai/guides/hills/
- Autolab experiment lifecycle: https://docs.autolab.ai/guides/experiments/
- Autolab CLI reference: https://docs.autolab.ai/reference/cli/
- Autolab Claude Code integration: https://docs.autolab.ai/guides/coding-agents/
- Egbe quickstart: https://docs.egbe.ai/start/quickstart/
- Egbe Bring Your Own Agent: https://docs.egbe.ai/start/connect-agent/
- vLLM EAGLE3 serving: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/serve_vllm/
- Hugging Face November 20, 2026 harness adapters: https://huggingface.co/blog/harness-adapters-at-scale
- HF ML Intern: https://huggingface.co/docs/chat-ui/ml-intern/getting-started
- Induction Labs FrontierML: https://www.frontierml.org/
- METR November 20 research: https://metr.org/blog/2026-11-20-q4-2026-capability-update/

### Repository-specific prior plans
- https://github.com/lonexreb/mobius/blob/revival/sundai-recursive-experiments-2026/docs/MOBIUS_FRONTIER_RESEARCH_NOV_2026.md
- https://github.com/lonexreb/mobius/blob/revival/sundai-recursive-experiments-2026/docs/SUNDAI_SPONSOR_INTEGRATION.md
- https://github.com/lonexreb/mobius/blob/revival/sundai-recursive-experiments-2026/experiments/frontier2026/research_plan.yaml

### Last caution

The research thesis is deliberately ambitious. The hackathon deliverable is **a real product slice with source-linked experimental evidence and a defensible decision policy**, not proof of a general autonomous scientific discovery breakthrough. Treat every stronger claim as a hypothesis to be tested later.

**One loop. One twist. Every pass learns — only when the evidence supports it.**
