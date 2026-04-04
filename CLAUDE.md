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
  Cargo.toml              # Workspace root (edition 2024, resolver 2)
  crates/
    mobius-core/           # Types, config, storage, compute, schema, aggregation, pareto
    mobius-bench/          # Evaluation: matchers, dimension scorers, multi-dim evaluator
    mobius-claw/           # Autonomous agent: strategy, learning, hooks, agent loop
    mobius-cli/            # CLI binary: init, evaluate, status, history, run, suggest, sweep, agent
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
| `mobius-core` | `ExperimentResult`, `ExperimentConfig`, `MobiusConfig`, `BudgetGuard`, `GroundTruth`, `Prediction`, `BenchResult`, `ComputeBackend`/`AsyncComputeBackend` traits, `SyncAdapter`, `ParallelBackend`, `ExperimentStore` trait |
| `mobius-bench` | `Evaluator`, `Matcher` trait, `DimensionScorer` trait, `GreedyTimestampMatcher`, `F1Scorer`, `ClassificationAccuracyScorer`, `TimestampMaeScorer` |
| `mobius-claw` | `AgentLoop`, `Strategy` trait, `GradientGuidedTuning`, `RandomSearch`, `GridSearch`, `build_strategy()`, `LearningStore`, `PreExecuteHook`/`PostEvaluateHook` traits, `Decision`, `StopReason` |
| `mobius-cli` | Clap commands: init, evaluate, status, history, run, suggest, sweep, agent (`--strategy` flag) |
| `mobius-mcp` | `MobiusServer`, `SharedState`, 10 tool handlers, 3 resource handlers |

## Code Quality

```bash
cargo fmt --all                                          # Format
cargo clippy --workspace --all-targets -- -D warnings    # Lint (zero warnings)
cargo test --workspace                                   # Test (82 passing)
cargo test -p mobius-core                                # Core only (17 tests)
cargo test -p mobius-bench                               # Bench only (4 tests)
cargo test -p mobius-claw                                # Claw only (19 tests)
cargo test -p mobius-cli                                 # CLI only (8 tests)
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
- `ExperimentStore` (default: `JsonlStore`)
- `ComputeBackend` / `AsyncComputeBackend` (default: `SubprocessBackend`, async via `SyncAdapter`)
- `Matcher` (default: `GreedyTimestampMatcher`)
- `DimensionScorer` (F1, accuracy, timestamp MAE)
- `Strategy` (`GradientGuidedTuning`, `RandomSearch`, `GridSearch` via `build_strategy()`)
- `PreExecuteHook` / `PostEvaluateHook`

### The 7-Step Agent Loop

```
ORIENT -> RESEARCH -> PROPOSE -> EXECUTE -> EVALUATE -> LEARN -> DECIDE
           (MCP)     Strategy   Compute    PostHooks  Learning  Budget/
                     .suggest() Backend               Store     Targets
```

### Configuration Hierarchy

1. `mobius.toml` — project config (loaded by `MobiusConfig::load()`)
2. CLI flags — override config
3. JSON overrides — `--config '{"param": value}'`
4. Environment variables — via `config_to_env()`

### Data Flow

```
mobius.toml -> ExperimentConfig -> ComputeBackend.submit() -> RawOutput
  -> OutputParser.parse() -> ParsedMetrics -> ExperimentResult (stored)
  -> Learning (extracted) -> ParamGradient -> Strategy.suggest() -> next config
```

## State Files

Mobius stores state in `~/.mobius/`:
- `history.jsonl` — append-only experiment results
- `learnings.jsonl` — append-only learning entries
- `budget.json` — current spend tracking

## How to Extend

### Adding a DimensionScorer

1. Implement `DimensionScorer` trait in `crates/mobius-bench/src/scorers.rs`
2. Register in `crates/mobius-cli/src/commands/evaluate.rs::build_scorer()`
3. Add to example config `[[bench.dimensions]]`

### Adding a Strategy

1. Implement `Strategy` trait in `crates/mobius-claw/src/strategy.rs`
2. Register in `build_strategy()` factory in the same file
3. Available via CLI `--strategy name` and MCP `strategy` param automatically

### Adding a Hook

1. Implement `PreExecuteHook` or `PostEvaluateHook` in `crates/mobius-claw/src/hooks.rs`
2. Return `HookAction::Proceed`, `Warn(msg)`, or `Block(msg)`
3. Register via `agent.add_pre_hook()` / `agent.add_post_hook()`

### Adding a CLI Command

1. Create `crates/mobius-cli/src/commands/<name>.rs`
2. Add `pub mod <name>;` to `commands/mod.rs`
3. Add variant to `Commands` enum in `main.rs`
4. Add match arm calling `commands::<name>::run()`
