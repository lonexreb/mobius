# Möbius Lab runbook

[Implementation handoff](MOBIUS_LAB_CLAUDE_CODE_HANDOFF.md) · [External dependencies for Codex](MOBIUS_LAB_CODEX_REQUESTS.md)

Möbius Lab is a local web app: you give it an ML optimization objective and a budget; a transparent
controller proposes experiments, a fixed evaluator scores them, and the controller promotes, replicates,
audits, rejects, investigates or stops, with every decision and its evidence stored in a SQLite ledger.

Everything below was run on 2026-11-22 on an Apple M4 Max (64 GB), macOS 26.2. Spend: **$0**.

## Start it

```bash
./scripts/demo_local.sh          # builds what is missing, starts API + UI, Ctrl-C stops both
```

or by hand, in two terminals:

```bash
# one-time setup
cargo build -p mobius-cli
uv run --python 3.12 --with pyarrow python experiments/frontier2026/speculative/fetch_prompts.py
cd apps/api && uv venv --python 3.12 .venv && uv pip install -p .venv -e ".[dev]"
cd apps/web && npm install

# terminal 1: API on 127.0.0.1:8000
cd apps/api && .venv/bin/uvicorn mobius_lab.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 3

# terminal 2: UI on http://127.0.0.1:5173
cd apps/web && npm run dev
```

Check: `curl -fsS http://127.0.0.1:8000/api/v1/health`

Local studies need `llama-server` on PATH (`brew install llama.cpp`) and the Ollama models
`qwen2.5:7b` and `qwen2.5:1.5b` (`ollama pull ...`), or GGUF paths in `MOBIUS_LAB_TARGET_GGUF` /
`MOBIUS_LAB_DRAFT_GGUF`. Without them the UI shows the local evaluator as unavailable and blocks Start.

State lives in `apps/api/var/` (ledger `lab.sqlite`, run artifacts, per-arm Rust optimizer state). It is
git-ignored. Delete the directory to reset.

## Tests

| Command | Result on 2026-11-22 |
|---------|----------------------|
| `cargo fmt --all -- --check` | clean (before and after) |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (before and after) |
| `cargo test --workspace` | 161 passed, 0 failed (bench 4, claw 74, cli 25, core 24, mcp 30 + 4 e2e) |
| `cd apps/api && .venv/bin/pytest -q` | 89 passed (75 API/policy/budget/import/replay/bridge/modal + 14 evaluator) |
| `cd apps/web && npm run build` | type-check + build OK, 79.6 kB gzipped JS |
| `cd apps/web && npx vitest run` | 4 passed |

No Rust source was changed. The Rust CLI is used as built: `mobius ask --strategy tpe --metric speedup_vs_target --store sqlite`
and `mobius tell`, with a separate `HOME` and working directory per study and policy arm
(`apps/api/var/mobius_state/<study>/<arm>/`). Verified: the global `~/.mobius` is untouched.

Not automated: browser end-to-end tests. The UI was checked by hand in Chrome at desktop width and at
768 / 390 / 320 px (no horizontal overflow), against real API data.

## Demo flow (about 90 seconds of talking, 6 minutes of machine time)

1. Home: point at Connections. Local evaluator and the Rust optimizer are ready; Autolab, GPU, LLM proposer and Egbe say "not connected" with the reason.
2. Leave the defaults (local validation, `evidence_aware`, 8 runs, $0) and press **Create and start**.
3. The timeline fills in live: target-only baseline first, then candidates branching from each other.
4. Click a candidate. The drawer shows the hypothesis, each validity gate, the measured metrics, per-domain speedup, provenance hashes and per-prompt rows.
5. "Why Möbius did this" shows the current decision with the run IDs it cites; expand the full trace.
6. Export the report (.md) and the evidence (.json).
7. For a no-wait demo: create a study in **Replay recorded runs** mode from a finished study. It re-decides from recorded artifacts in about a second and is labelled REPLAY everywhere.

If time allows, run the same budget with `metric_greedy` and compare how many runs each policy spent.

## What was measured

Workload: 24 dev prompts (8 each from LiveCodeBench code_generation_lite, MATH-500 and UltraChat at pinned
revisions), greedy decoding, 96 output tokens. Target `qwen2.5:7b`, draft `qwen2.5:1.5b`, llama.cpp
build 7480 on Metal. Speedup is candidate ÷ interleaved in-run target-only control.

### Study `study_3919d763ab`: `evidence_aware`, 8-run budget

| Run | Config | Speedup (95% CI) | Prompts with identical tokens | Draft acceptance | Verdict |
|-----|--------|------------------|-------------------------------|------------------|---------|
| baseline | target-only | reference, 63.8 tok/s | reference | n/a | valid reference |
| 1 | lookahead 4, p_min 0.75 | 0.744× [0.719, 0.766] | 8 / 24 | 85% | invalid, rejected |
| 2 | lookahead 4, p_min 0.5 | 0.719× [0.681, 0.776] | 6 / 24 | 77% | invalid, rejected |
| 3 | lookahead 2, p_min 0.5 | 0.781× [0.748, 0.818] | 12 / 24 | 83% | invalid, rejected |

Final decision: **INVESTIGATE** after 4 of 8 runs (6.1 minutes). Three distinct configs failed the exact-token gate
and none passed, so the controller stopped searching knobs and left 4 runs unspent.

### Study `study_68dddde577`: `metric_greedy`, same 8-run budget

| Run | Config | Speedup (95% CI) | Prompts with identical tokens | Draft acceptance | Verdict |
|-----|--------|------------------|-------------------------------|------------------|---------|
| baseline | target-only | reference, 56.7 tok/s | reference | n/a | valid reference |
| 1 | lookahead 4, p_min 0.75 | 0.741× [0.713, 0.767] | 8 / 24 | 85% | invalid, rejected |
| 2 | lookahead 4, p_min 0.5 | 0.704× [0.675, 0.733] | 6 / 24 | 77% | invalid, rejected |
| 3 | lookahead 2, p_min 0.5 | 0.804× [0.760, 0.833] | 12 / 24 | 83% | invalid, rejected |
| 4 | lookahead 2, p_min 0.75 | 0.791× [0.772, 0.822] | 14 / 24 | 86% | invalid, rejected |
| 5 | lookahead 2, p_min 0.9 | 0.808× [0.762, 0.864] | 12 / 24 | 86% | invalid, rejected |
| 6 | lookahead 4, p_min 0.9 | 0.773× [0.725, 0.833] | 9 / 24 | 87% | invalid, rejected |
| 7 | lookahead 8, p_min 0.9 | 0.831× [0.803, 0.880] | 9 / 24 | 87% | invalid, rejected |

Final decision: **STOP** after all 8 runs (13.6 minutes): none was valid and faster than target-only.

Both arms reached the same conclusion; `evidence_aware` got there in half the runs. The first three configs are the
same in both studies and their numbers agree closely (0.744 / 0.741, 0.719 / 0.704, 0.781 / 0.804, identical match
counts), which is one informal replication.

### Study `study_e9e2e4f286`: replay of the first study

Re-served the four recorded runs in about a second, labelled REPLAY, and reached the same INVESTIGATE decision.

### What this does and does not show

- On this engine and model pair, speculative decoding was **slower** than target-only and **not lossless**. Most mismatches are outputs that stop a few tokens short of the 96-token limit with an identical prefix; a minority diverge mid-sequence. The in-run target-only control reproduced the stored reference tokens in every run, so the gate itself is trustworthy.
- This is a result about llama.cpp build 7480 with a Qwen2.5-1.5B draft on Apple Metal. It says **nothing** about Qwen3-8B with EAGLE-3 on vLLM/CUDA, which has not been run.
- It does **not** show that one policy is better than another. Two studies, one machine, pilot-sized prompt sets: no significance is claimed.
- No candidate passed validity, so the replicate, audit and promote paths were not exercised on real data. They are covered by tests with fixture data only (`apps/api/tests/`), which is labelled FIXTURE and cannot be replayed as evidence.

## How the pieces fit

| Piece | Where | Notes |
|-------|-------|-------|
| Evaluator | `experiments/frontier2026/speculative/evaluator.py` | Separate process, stdlib only. Config allowlist, token-ID equality, paired speedup, bootstrap CI |
| Prompt registry | `experiments/frontier2026/speculative/dataset_registry.json` | Revisions, checksums, prompt and split hashes; no prompt text in git |
| Ledger | `apps/api/src/mobius_lab/persistence.py` | SQLite WAL; studies, experiments, decisions, budget events, event stream, import claims |
| Controller | `apps/api/src/mobius_lab/policies.py` | Pure function of observed state; six arms |
| Orchestrator | `apps/api/src/mobius_lab/orchestrator.py` | decide → reserve budget → run → validate → record |
| Adapters | `apps/api/src/mobius_lab/adapters.py` | Local evaluator, replay, Rust `ask`/`tell` bridge, connection readiness |
| API | `apps/api/src/mobius_lab/main.py` | `/api/v1`: health, templates, connections, studies, experiments, events (SSE), report, export, import |
| UI | `apps/web/src/` | React + TypeScript + Vite |

Evidence labels: **LOCAL** (measured here), **OBSERVED** (imported external run with source ID),
**REPLAY** (recorded run re-served), **FIXTURE** (tests only).

Controller rules, in order: no valid baseline → run it. Failed or invalid run → reject. `evidence_aware` only:
gain inside the within-run CI → replicate; replicates that disagree in direction → flag as unstable; always hold
back enough budget to audit on the longer-prompt split before promoting; three distinct configs invalid with none
valid → investigate; three valid configs all below 0.90× → stop. Other arms promote the best valid dev score.
Thresholds are constants at the top of `policies.py`.

Budget: scored runs, wall time and USD are checked atomically before each run. Unknown cost is refused rather
than treated as zero. Cost categories are tracked separately (rented GPU, LLM inference, Autolab agent,
imported Autolab runs); unknown amounts are counted as unknown. A non-zero cap needs an explicit approval call.
Starting twice, or importing the same external run twice, creates one run and one charge.

Restart behaviour: studies, results and decisions persist. A run that was in flight is cancelled (no
measurement recorded, reservation released) and its study is paused; press Resume.

## Flagship (Qwen3-8B + EAGLE-3): prepared, not run

**Modal path (added after the owner approved a $25 cap): wired, not yet run.** `integrations/modal/bench_app.py` runs
the same evaluator inside an A100 80 GB container with vLLM; the API has a `modal` mode whose adapter reserves an
upper-bound cost per run (about $1.28) against the study cap, records an estimated cost for every run including
failures, and stops scheduling when the cap would be exceeded. Covered by tests with a fixture standing in for the
remote call. **Blocked:** the Modal workspace on this machine is paused, so the weight download, the vLLM probe and
every GPU run were refused; $0 spent. `VllmEngine.complete` is deliberately unfinished until the probe shows how this
vLLM build returns generated token IDs. The paragraphs below describe the state before Modal was chosen.

Ready: pinned model revisions (`manifest.json` → `flagship`), the prompt registry and splits, the correctness
and scoring code, the ledger contract, and the ingestion endpoint `POST /api/v1/import/autolab`
(`AutolabImport` in `schemas.py`: source run ID, Hill hash, verdict, cost, evaluator artifact; idempotent).

Missing, and why: a CUDA GPU and vLLM (none here, $0 authorized), a `VllmEngine` in the evaluator (not written
because its serve flags and token-ID output could not be verified without an installation), the Autolab CLI,
account and Hill, and a real Autolab metrics payload to map. Each is itemised in
[MOBIUS_LAB_CODEX_REQUESTS.md](MOBIUS_LAB_CODEX_REQUESTS.md).

## Known limitations

- No authentication. The API must stay on 127.0.0.1.
- Stopping the API while a browser tab is open can hang on the open event stream; the shutdown timeout flag above bounds that to 3 seconds. Without it, the old process keeps the ledger lock and a new one refuses to start.
- One API process per ledger: a second instance refuses to start (`var/lab.lock`). Running `demo_local.sh` twice fails on purpose.
- An imported Autolab run with `cost_usd: null` is recorded with an unknown-cost event rather than refused, including on a $0 study. It is visible in the budget panel but is not blocked.
- Double-run protection relies on one study loop per process plus the single-process lock; there is no distributed guard.
- One policy arm per study. There is no matched multi-arm comparison view; compare two studies by hand.
- Autolab mode can only receive imported runs. It cannot dispatch, poll or cancel remote work.
- Replay re-serves runs recorded in the same ledger. Importing an exported JSON from another machine is not implemented.
- Hypotheses come from a rule-based neighbourhood proposer, not an LLM.
- The sealed split has never been evaluated; there is no post-freeze unsealing workflow beyond the evaluator's `--unseal-final` flag.
- Bootstrap CIs cover prompt-to-prompt variation within a run only. Background load moved target-only throughput by about 30% between back-to-back runs on this machine, which is why speedup uses an interleaved control.
- Peak memory is the server process RSS, which under-reports Metal memory. Time-to-first-token is client-measured over a streaming response.
- The TPE arm's first suggestions are random until the Rust optimizer has enough history, and are not seeded.
- Not built from the handoff: Egbe integration, comparison route, Playwright suite, TanStack Query, a separate architecture doc.

## Commits on `revival/sundai-recursive-experiments-2026`

| Commit | Contents |
|--------|----------|
| `0f0c7d0` | Prompt registry and paired speculative-decoding evaluator |
| `7930539` | API: ledger, controller, orchestration, tests |
| `60363ca` | Web UI, demo scripts, Codex request list |
| `a4c33b7` | Fixes from an independent code review (10 findings, with regression tests) |
| (next) | This runbook: see `git log --oneline` |

`main` was not touched. Nothing was pushed.
