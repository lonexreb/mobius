# Mobius

```
 __  __  ___  ___ ___ _   _ ___
|  \/  |/ _ \| _ )_ _| | | / __|
| |\/| | (_) | _ \| || |_| \__ \
|_|  |_|\___/|___/___|\___/|___/
  One loop. One twist. Every pass learns.  v0.1
```

High-performance Rust framework for autonomous ML experimentation. One library: experiment loops, evaluation, agent orchestration, and MCP-native tool integration.

## Architecture

```
mobius/
  Cargo.toml              # Workspace root (edition 2024, resolver 2, LTO release profile)
  crates/
    mobius-core/           # Types, config, storage (JSONL + SQLite), compute, schema, aggregation, pareto
    mobius-bench/          # Evaluation: matchers, dimension scorers, multi-dim evaluator
    mobius-claw/           # Autonomous agent: 10 strategies, 4 hooks, 2 pruners, learning, importance
    mobius-cli/            # CLI binary: 17 commands, colored output, progress bars, TUI dashboard
    mobius-mcp/            # MCP server: 10 tools, 3 resources, rmcp over stdio
```

### Crate Dependency Graph

```
mobius-core  (no internal deps)
    ├── mobius-bench  (core)
    │       └── mobius-claw  (core + bench)
    │               └── mobius-cli  (core + bench + claw)
    └── mobius-mcp   (core + bench + claw)
```

### Crate Responsibilities

| Crate | Key Types |
|-------|-----------|
| `mobius-core` | `ExperimentResult`, `ExperimentConfig`, `ExperimentStatus` (Success/Error/Timeout/Pending), `MobiusConfig`, `StrategyParams`, `BudgetGuard`, `GroundTruth`, `Prediction`, `BenchResult`, `ComputeBackend`/`AsyncComputeBackend` traits, `SyncAdapter`, `ParallelBackend`, `ExperimentStore` trait, `JsonlStore`, `SqliteStore` (WAL mode, prepared statement caching) |
| `mobius-bench` | `Evaluator`, `Matcher` trait, `DimensionScorer` trait, `GreedyTimestampMatcher`, `F1Scorer`, `ClassificationAccuracyScorer`, `TimestampMaeScorer` |
| `mobius-claw` | `AgentLoop`, `Strategy` trait (10 impls: `GradientGuidedTuning`, `RandomSearch`, `GridSearch`, `TpeSearch` (log-scale), `NsgaTwo`, `UcbTreeSearch`, `CmaEs` (log-scale), `Pbt`, `AutoStrategy`, `Hyperband`), `AshaPruner`, `MedianPruner`, `build_strategy()`/`build_strategy_with_params()`, `LearningStore`, `compute_importance()`, `PreExecuteHook`/`PostEvaluateHook` traits (4 hooks: Budget, Regression, Overfitting, Constraints), `Decision`, `StopReason` |
| `mobius-cli` | 17 clap commands: init, evaluate, status, history, run, suggest, sweep (`--parallel`), agent (`--strategy`, `--pruning`, `--store`), dashboard, importance, ask, tell, compare, enqueue, export |
| `mobius-mcp` | `MobiusServer`, `SharedState`, 10 tool handlers, 3 resource handlers |

## Code Quality

```bash
cargo fmt --all                                          # Format
cargo clippy --workspace --all-targets -- -D warnings    # Lint (zero warnings)
cargo test --workspace                                   # Test (161 passing)
cargo test -p mobius-core                                # Core only (24 tests)
cargo test -p mobius-bench                               # Bench only (4 tests)
cargo test -p mobius-claw                                # Claw only (74 tests)
cargo test -p mobius-cli                                 # CLI only (25 tests)
cargo test -p mobius-mcp                                 # MCP only (34 tests)
```

### Rules

- **Edition 2024** — use Rust 2024 idioms
- **Zero warnings** — both `rustc` and `clippy`
- **Functions <= 80 lines** — extract helpers for longer logic
- **Every public type and trait method gets a `///` doc comment**
- **Error handling** — `anyhow::Result` for app code, `thiserror` for library errors
- **No `.unwrap()` in library code** — only in tests and CLI formatting
- **`impl AsRef<Path>`** over `&str` for file paths
- **Workspace deps** — all shared deps in root `Cargo.toml` with `.workspace = true`

## Key Design Patterns

### Trait-Based Extension

All core abstractions are traits for swappable implementations:
- `ExperimentStore` (`JsonlStore`, `SqliteStore` — selectable via `--store`)
- `ComputeBackend` / `AsyncComputeBackend` (default: `SubprocessBackend`, async via `SyncAdapter`)
- `Matcher` (default: `GreedyTimestampMatcher`)
- `DimensionScorer` (F1, accuracy, timestamp MAE)
- `Strategy` (10 strategies via `build_strategy()` / `build_strategy_with_params()`)
- `PreExecuteHook` / `PostEvaluateHook` (4 hooks: budget, regression, overfitting, constraints)

### Strategy List

| Name | Aliases | Key Parameters |
|------|---------|---------------|
| `gradient_guided` | `gradient_guided_tuning` | `plateau_window`, `plateau_threshold` |
| `random` | `random_search` | — |
| `grid` | `grid_search` | — |
| `tpe` | `tpe_search` | `tpe_gamma` (log-scale auto-detect) |
| `nsga2` | `nsga_ii` | `nsga_objectives`, `nsga_population_size` |
| `ucb1` | `tree_search` | `ucb1_exploration_constant` |
| `cmaes` | `cma_es` | `cmaes_population_size` (log-scale auto-detect) |
| `pbt` | `population_based` | `pbt_population_size` |
| `auto` | `auto_strategy` | — (auto-selects from above) |
| `hyperband` | — | `hyperband_max_resource`, `hyperband_eta` |

### The 7-Step Agent Loop

```
ORIENT -> PROPOSE -> EXECUTE -> EVALUATE -> LEARN -> DECIDE
          Strategy   Compute    PostHooks  Learning  Budget/
          .suggest() Backend    (4 hooks)  Store     Targets
```

### Configuration Hierarchy

1. `mobius.toml` — project config (loaded by `MobiusConfig::load()`)
2. `[agent.strategy_params]` — strategy-specific hyperparameters
3. CLI flags — override config
4. JSON overrides — `--config '{"param": value}'`
5. Environment variables — via `config_to_env()`

### Data Flow

```
mobius.toml -> ExperimentConfig -> ComputeBackend.submit() -> RawOutput
  -> OutputParser.parse() -> ParsedMetrics -> ExperimentResult (stored)
  -> Learning (extracted) -> ParamGradient -> Strategy.suggest() -> next config
```

## State Files

Mobius stores state in `~/.mobius/`:
- `history.jsonl` — append-only experiment results (JSONL backend)
- `history.db` — SQLite experiment database (SQLite backend, `--store sqlite`)
- `learnings.jsonl` — append-only learning entries
- `budget.json` — current spend tracking

## How to Extend

### Adding a Strategy

1. Create `crates/mobius-claw/src/<name>.rs` implementing `Strategy` trait
2. Add `pub mod <name>;` to `crates/mobius-claw/src/lib.rs`
3. Add match arm in `build_strategy_with_params()` in `crates/mobius-claw/src/strategy.rs`
4. Available via CLI `--strategy name` and MCP `strategy` param automatically

### Adding a DimensionScorer

1. Implement `DimensionScorer` trait in `crates/mobius-bench/src/scorers.rs`
2. Register in `crates/mobius-cli/src/commands/evaluate.rs::build_scorer()`
3. Add to example config `[[bench.dimensions]]`

### Adding a Hook

1. Implement `PreExecuteHook` or `PostEvaluateHook` in `crates/mobius-claw/src/hooks.rs`
2. Return `HookAction::Proceed`, `Warn(msg)`, or `Block(msg)`
3. Register via `agent.add_pre_hook()` / `agent.add_post_hook()`

### Adding a CLI Command

1. Create `crates/mobius-cli/src/commands/<name>.rs`
2. Add `pub mod <name>;` to `commands/mod.rs`
3. Add variant to `Commands` enum in `main.rs`
4. Add match arm calling `commands::<name>::run()`
