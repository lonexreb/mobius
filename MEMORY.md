# Mobius Session Memory

```
 _____ _____ _____ _____ _____ _____
|     |     | __  |     |  |  |   __|
| | | |  |  | __ -|-   -|  |  |__   |
|_|_|_|_____|_____|_____|_____|_____|
            SESSION MEMORY
```

## Project Identity

- **Repo**: github.com/lonexreb/mobius
- **Path**: `/Users/shubh-trips/Documents/personal-project/cli-based-tools/mobius/`
- **Language**: Rust (edition 2024, resolver 2)
- **Workspace**: 5 crates (`mobius-core`, `mobius-bench`, `mobius-claw`, `mobius-cli`, `mobius-mcp`)
- **License**: MIT OR Apache-2.0
- **Tests**: 161 passing (24 core + 4 bench + 74 claw + 25 cli + 30 mcp + 4 e2e)
- **Phase**: 8 complete; Phase 9 (SDK + Remote) planned

## Architecture Decisions

1. **Trait-driven extensibility** — `ExperimentStore`, `ComputeBackend` / `AsyncComputeBackend`, `Matcher`, `DimensionScorer`, `Strategy`, `PreExecuteHook`, `PostEvaluateHook`
2. **Dual storage backends** — JSONL (default, append-only) and SQLite (WAL mode, indexed, prepared-statement cached) selectable via `--store`
3. **Gradient-guided default strategy** — accumulated learning signals guide exploration; 9 alternatives available
4. **Domain-agnostic schema** — `GroundTruth` / `Prediction` use `HashMap<String, Value>` attributes
5. **Budget as first-class** — `BudgetGuard` with persistent state and pre-execution checks
6. **Auto log-scale detection** — TPE and CMA-ES detect log-scale parameters (>2 orders of magnitude span) and operate in log-space
7. **Multi-fidelity scheduling** — Hyperband meta-scheduler + ASHA / Median pruners for cost reduction
8. **MCP-native** — `mobius-mcp` exposes 10 tools and 3 resources via rmcp over stdio

## What Has Been Built

### Phase 1: Core Framework [COMPLETE]
ExperimentResult / Config types, JsonlStore, BudgetGuard, ErrorClassifier, MobiusConfig (TOML), SubprocessBackend (with real timeout kill), OutputParser, data loaders, Pareto front, macro/micro aggregation.

### Phase 2: Agent + Evaluation Engine [COMPLETE]
GreedyTimestampMatcher, F1/Accuracy/Timestamp scorers, Evaluator, GradientGuidedTuning, LearningStore, AgentLoop (7-step ORIENT-DECIDE), 3 hooks (budget/regression/overfitting), 8 CLI commands.

### Phase 3: MCP + Strategy Expansion [COMPLETE]
`mobius-mcp` server (10 tools, 3 resources), RandomSearch, GridSearch, TpeSearch, AsyncComputeBackend + SyncAdapter.

### Phase 4: Production Hardening [COMPLETE]
Config wiring (`mobius.toml` drives strategy/sweep), strategy switching on consecutive reverts, parallel sweeps with `--max-concurrency`, CI (fmt + clippy + test), example config, publish prep.

### Phase 5: Multi-Objective + Dashboard [COMPLETE]
`NsgaTwo`, `AshaPruner` (`--pruning` flag), live ratatui TUI dashboard.

### Phase 6: Tree Search + SQLite [COMPLETE]
`UcbTreeSearch`, `SqliteStore` (WAL, prepared-statement cache, `json_extract` get_best), `--store` CLI flag.

### Phase 7: Best-in-Class [COMPLETE]
SubprocessBackend timeout-kill bug fix; SQLite WAL + PRAGMAs (5-10x writes); JSONL `to_writer` zero-alloc; release profile (LTO, codegen-units=1, strip, panic=abort); Rayon parallelization (TPE KDE, NSGA-II crowding); `CmaEs`; `AutoStrategy`; `Pbt`; `compute_importance()` (fANOVA); `owo-colors` styling; `indicatif` progress bars; `mobius ask` / `tell` / `compare` / `importance`.

### Phase 8: Production ML Power Features [COMPLETE]
Log-scale sampling auto-detection (TPE log-KDE, CMA-ES log-normalized covariance); `Hyperband` meta-scheduler; `MedianPruner`; fidelity-aware experiments; configurable strategy params via `[agent.strategy_params]`; `build_strategy_with_params()`; outcome constraints (upper/lower bounds); `mobius enqueue`; auto-retry; `mobius export` (CSV/JSON).

### Not Yet Built (Phase 9 PLANNED)
- Python SDK (`pymobius`) — PyO3 bindings
- Go SDK (`go-mobius`) — CGo bindings
- Remote compute backends — Modal, RunPod, SSH
- Plugin system — dynamic scorer/strategy loading
- Distributed coordination — multi-node agent swarms
- Security & isolation — sandboxed execution, credential gateway

### Known Empty Slots
- `examples/basketball-shot-detection/` — empty dir, referenced in CLAUDE.md
- `examples/gpt2-training/` — empty dir

## Strategy Catalog (10)

| Name | Aliases | Type | Best For |
|------|---------|------|----------|
| `gradient_guided` | `gradient_guided_tuning` | Gradient-based | Default; learns from each run |
| `tpe` | `tpe_search` | Bayesian (log-scale aware) | Mixed params, 10-100 trials |
| `cmaes` | `cma_es` | Evolution (log-scale aware) | Continuous, low-dim |
| `hyperband` | — | Multi-fidelity | Expensive runs, 40-70% cost saving |
| `nsga2` | `nsga_ii` | Multi-objective | Conflicting metrics |
| `ucb1` | `tree_search` | AIDE-style tree | Hypothesis exploration |
| `pbt` | `population_based` | Population evolutionary | Dynamic schedules |
| `auto` | `auto_strategy` | Meta-selector | When unsure |
| `grid` | `grid_search` | Exhaustive | Small spaces |
| `random` | `random_search` | Baseline | Initial exploration |

## CLI Commands (17)

`init`, `run`, `suggest`, `sweep` (`--parallel`), `agent` (`--strategy`, `--pruning`, `--store`), `evaluate`, `status`, `history`, `dashboard`, `importance`, `ask`, `tell`, `enqueue`, `compare`, `export`.

## Code Quality Discipline

- Edition 2024 idioms; zero rustc + clippy warnings (`-D warnings`)
- Functions ≤ 80 lines; every public type/method has `///` doc
- `anyhow::Result` in app code, `thiserror` in library errors
- No `.unwrap()` in library code (tests/CLI formatting OK)
- `impl AsRef<Path>` over `&str` for paths
- All shared deps in workspace root with `.workspace = true`

## Key File Paths

| What | Path |
|------|------|
| Workspace config | `Cargo.toml` |
| Project guide | `CLAUDE.md` |
| Roadmap | `FEATURES.md` |
| Core types | `crates/mobius-core/src/experiment.rs` |
| Schema | `crates/mobius-core/src/schema.rs` |
| Compute backends | `crates/mobius-core/src/compute.rs` |
| JSONL store | `crates/mobius-core/src/store/jsonl.rs` |
| SQLite store | `crates/mobius-core/src/store/sqlite.rs` |
| Evaluator | `crates/mobius-bench/src/evaluator.rs` |
| Strategy factory | `crates/mobius-claw/src/strategy.rs` |
| Agent loop | `crates/mobius-claw/src/agent.rs` |
| Importance (fANOVA) | `crates/mobius-claw/src/importance.rs` |
| Hooks | `crates/mobius-claw/src/hooks.rs` |
| Pruners | `crates/mobius-claw/src/pruning.rs` |
| Hyperband | `crates/mobius-claw/src/hyperband.rs` |
| CLI entry | `crates/mobius-cli/src/main.rs` |
| MCP server | `crates/mobius-mcp/src/lib.rs` |

## State Files

Mobius writes to `~/.mobius/`:
- `history.jsonl` — append-only experiment results (JSONL backend)
- `history.db` — SQLite experiment database (`--store sqlite`, WAL mode)
- `learnings.jsonl` — append-only learning entries
- `budget.json` — current spend tracking

## Build / Test Commands

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                # 161 passing
cargo build --release                 # LTO, single codegen unit, stripped
```
