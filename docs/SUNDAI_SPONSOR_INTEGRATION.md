# Möbius × Autolab × Egbe — Sundai Hackathon Integration
Date: 2026-11-22
Status: Build plan; no live sponsor account connection or experiment outcome has been verified.

## Why this composition

Möbius is the **meta-experiment controller** (hypothesis prioritization, budget, cross-run provenance, decision policy).
Autolab is the **ML experiment environment** (fixed Hill evaluation, Climb compute, run tracking, agent-written candidate code).
Egbe is the **distribution and feedback environment** (authorized research, product/landing pages, analytics, acquisition playbooks).

Avoid building a second copy of Autolab's agent or Egbe's campaign engine. A differentiating capability is one Möbius ledger comparing **whether to invest the next unit of effort in model quality or customer acquisition**, not merely a dashboard showing sponsor logos.

## Verified underlying Möbius baseline

- Rust workspace with core, bench, claw, cli, mcp crates; documented 10 strategies and 161 tests.
- Verified GitHub Actions run passed 2026-04-04, but do not claim fresh tests before executing them.
- Agent loop in crates/mobius-claw/src/agent.rs has a RESEARCH no-op and currently optimizes fixed parameter sweep values, not agent-generated patches.
- Existing 'mobius ask' and 'mobius tell' accept JSON configs/metrics; 'tell' does NOT preserve run provenance by itself.
- The current CLI agent sets primary metric to 'f1'. Avoid treating that as an arbitrary-metric interface without changing/testing code.
- Historical docs/MEMORY.md is stale relative to README, FEATURES and latest commit.

## Sponsor integration A: Autolab (first priority)

Documentation:
- https://docs.autolab.ai/quickstart/
- https://docs.autolab.ai/guides/hills/
- https://docs.autolab.ai/reference/cli/
- https://docs.autolab.ai/guides/coding-agents/

1. Install the **official CLI** on an appropriate local development machine (review installer before running):
   ```bash
   curl -fsSL https://app.autolab.ai/install.sh | sh
   autolab --version
   autolab login
   autolab whoami
   autolab install claude
   ```
2. Work in a dedicated small benchmark workspace rather than submitting the entire repository as mutable model code.
3. Turn a fixed real-data task into a Hill:
   ```bash
   autolab hills new mobius-digits
   # Customize .autolab/hills/mobius-digits/hill.yaml, eval.py, README.md,
   # private/ (held-out test data), and examples/baseline/
   autolab hills check mobius-digits
   autolab hills commit mobius-digits -m "frozen real-data benchmark"
   autolab hills push mobius-digits  # private by default
   ```
   Dataset: scikit-learn's builtin digits dataset as a CPU-accessible real-data baseline. Immutable deterministic data split. Optimize validation accuracy, with time budget and held-out final test. As soon as GPU access is confirmed, consider a more demanding benchmark as an independent Hill (e.g. CIFAR-10), not by silently changing this Hill.
4. Create a Climb pointing at **actual** account slug; never hardcode a fake user:
   ```bash
   autolab init --hill YOUR_USERNAME/mobius-digits
   autolab status
   # Verify and configure maximum LLM spend, compute allowance and stopping policy.
   autolab serve
   # If needed: autolab nodes attach NODE --project YOUR_USERNAME/YOUR_CLIMB
   autolab start
   autolab open
   ```
   Alternatively pick 'Your coding agent' mode from the dashboard, turning off Autolab autonomous idea generation so the Möbius/Claude proposer can choose the experiments; the Autolab agent can still write code for idea-only submissions.
5. Queue a hypothesis:
   ```bash
   autolab submit --nocode -m "Hypothesis: standardize features before logistic regression to improve fixed validation accuracy without exceeding runtime budget"
   autolab log
   autolab metrics RUN_ID --format json
   ```
   Metrics are sourced from official Autolab results, never from imagined run values. `autolab metrics` accepts actual run IDs from `autolab log`; its JSON contains metric series and params, so inspect exact shape before writing parser.
6. Small adapter (new `integrations/autolab/`): `sync_metrics.py` takes the Autolab run ID, executes the official metrics command, picks the exact final scorer metric recorded by the pinned Hill, validates finite numeric values, and records:
   - run ID, climb ID, hill version hash, code SHA, dataset hash, timestamp, experiment hypothesis, validation metric, wall time, cost (if available), verdict.
   - A structured `mobius tell --config ... --metrics ...` invocation for compatible numeric metrics, plus append-only JSONL provenance for source metadata and failures. Confirm `mobius tell --help` flags against locally built CLI.
   - Idempotent syncing: repeat import does not duplicate the same sponsor run.
7. Comparing Mobius decisions: use fixed-budget sequential experiment policy (random baseline vs history-aware hypotheses); show best-so-far validation trace, not one cherry-picked lucky experiment.

Autolab itself can autonomously generate hypotheses. **For a credible Möbius contribution, either set Autolab's idea generation OFF and have Möbius choose which experiments are submitted, or clearly mark the sponsor agent's suggestions as Autolab-originated.**

## Sponsor integration B: Egbe (second priority)

Docs:
- https://docs.egbe.ai/start/quickstart/
- https://docs.egbe.ai/start/connect-agent/
- https://docs.egbe.ai/reference/capabilities/
- https://docs.egbe.ai/reference/access-and-costs/

1. In Egbe Dashboard create/open business 'Möbius'. Choose 'Connect your agent' → 'Any AI agent' → 'Copy instructions', and paste **privately** into Claude Code in a persistent workspace.
2. Agent installs skills/playbooks, verifies authorized business identity, puts project credentials in git-ignored `.env.local`, and creates `egbe-docs/`. Never paste project credentials into repository issues or public hack demo.
3. Prompt Egbe-connected Claude:
   > Use Egbe's verified market research capabilities to identify the best first user profile for Möbius (indie ML researchers vs startup ML platform engineers). Produce sources, a short audience hypothesis, and two landing-page positioning variants. Do not send emails, run paid campaigns, deploy publicly, or spend credits without explicit approval. Check actual project allowances before proposing any paid tool.
4. Use research output to propose one launch experiment, with metric `qualified_interest_rate = qualified_signups / eligible_visits` on a real publicly approved landing page. A/B comparison requires clean assignment, instrumentation and sufficient visitors; low-volume findings are exploratory.
5. After user approval, use verified Egbe hosting/analytics skill to publish. Exact implementation comes from installed current Egbe skills, not invented API paths.
6. Small adapter `integrations/egbe/`: export verified, non-personal aggregate experiment results to Möbius as numeric metrics, while retaining provenance separately. No private lead lists, emails or raw personal data in public artifacts.
7. The free Egbe credits mentioned by the event may require account provisioning; verify allowance, permissions, and cost inside the connected business.

## Suggested shared ledger

```json
{
  "experiment_id": "uuid",
  "environment": "autolab_or_egbe",
  "sponsor_run_id": "external-id",
  "objective": "val_accuracy_or_qualified_interest_rate",
  "hypothesis": "testable statement",
  "metric": { "name": "val_accuracy", "value": null },
  "cost_usd": null,
  "duration_s": null,
  "benchmark_version": "immutable-reference",
  "status": "pending",
  "evidence_url": null,
  "decision": "pending"
}
```

This is a **proposed** interface, not an API implemented in Möbius today. Normalize improvement versus baseline or require explicit unit-specific decision rules; do NOT directly add validation accuracy to signup conversion to choose budgets.

## Execution plan for today

**Before 14:00 checkpoint:** Build/test existing Möbius, scaffold/check real-data Hill, attach CPU compute, get 1 official scored Autolab baseline, sync result into Möbius provenance ledger.

**14:00 to 16:00:** Submit 2–3 real hypotheses; instrument importer with tests for duplicate runs, malformed JSON, missing metrics, NaNs, and failed jobs. Measure improvement vs same-seed baseline.

**16:00 to 17:00:** Connect Egbe, research two audience hypotheses, draft landing copy and acquire permission for public publishing. If credits/auth are blocked, show research drafts, not fabricated analytics.

**17:00 to 18:00:** Final demo: Möbius decides next experiment; Autolab executes and scores; ledger compares prior runs; Egbe shows a separate growth experiment plan and first verifiable observation if any.

## Required integration tests

- Autolab parser handles real CLI JSON, invalid input, missing metric, non-finite values, and retries.
- Imported result is idempotent and linked to source run and pinned Hill hash.
- Fixed evaluator and held-out data cannot be edited by proposing agent.
- Budget exhaustion and failed candidate stop appropriately; no unseen outgoing public actions.
- Egbe unavailable or no traffic: no success claim, no made-up customer metrics.
- Existing Rust workspace fmt/clippy/tests pass before and after the integration.

## Claude Code handoff

> Work in lonexreb/mobius branch revival/sundai-recursive-experiments-2026. Read CLAUDE.md, Cargo.toml, the real agent/ask/tell sources, and this document. First run Rust fmt/clippy/tests and existing hello-experiment without changing code; report factual outputs. Then implement only integrations/autolab/sync_metrics.py with a sample-capture fixture from the **actual** CLI and tests; ensure no secrets get committed. Scaffold a fixed real-data Autolab Hill and run its check. Use your installed Autolab skill and its CLI to create the Climb; ask for login/budget/compute approvals when needed. After one real run is logged, implement Egbe connection in a separate task using the Dashboard's current private instructions. Keep the Rust core unchanged until a failing integration test requires a narrowly scoped change. Do not edit main.

## What success means

- Existing framework actually runs again;
- Autolab official scored experiments imported without forged values;
- A self-contained experiment source, score, and selection narrative;
- Egbe business research and (if permitted and real traffic exists) a verifiable growth experiment;
- Clear human approval on all paid/public actions;
- Distinguish **meta-controller decision** from sponsor tool work.