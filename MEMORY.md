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
- **Path**: `/Users/shubh-trips/Documents/personal-projects/cli-based-tools/mobius/`
- **Language**: Rust (edition 2024)
- **Workspace**: 5 crates (core, bench, claw, cli, mcp)
- **License**: MIT OR Apache-2.0
- **Tests**: 32 passing (15 core, 4 bench, 13 claw)

## Architecture Decisions

1. Trait-driven extensibility — all core abstractions are traits (`ExperimentStore`, `ComputeBackend`, `Matcher`, `DimensionScorer`, `Strategy`, hooks)
2. JSONL append-only storage — history and learnings use append-only JSONL
3. Gradient-guided tuning — strategy uses accumulated learning signals to guide exploration
4. Domain-agnostic schema — `GroundTruth`/`Prediction` use `HashMap<String, Value>` attributes
5. Budget as first-class — `BudgetGuard` with persistent state and pre-execution checks

## What Has Been Built

### Phase 1: Core Framework
ExperimentResult types, JsonlStore, BudgetGuard, ErrorClassifier, MobiusConfig (TOML), SubprocessBackend, OutputParser, data loaders, Pareto front, aggregation

### Phase 2: Agent + Evaluation
GreedyTimestampMatcher, F1/Accuracy/Timestamp scorers, Evaluator, GradientGuidedTuning strategy, LearningStore, AgentLoop (7-step), 3 hooks (budget/regression/overfitting), 8 CLI commands

### Not Yet Built
- mobius-mcp (Phase 3 stub)
- Python/Go SDK bindings
- Tree-search strategies
- Swarm/multi-agent coordination
- Remote compute backends
- Async agent loop

## Known Issues

1. `new_f1` field in `learning_store.rs:68` — dead_code warning (Observation struct field never read)
2. CLI `run`/`sweep` use placeholder `echo '{"f1": 0.0}'` instead of user-configured command
3. `env_map` not populated from config in CLI commands
4. MCP crate is a stub

## Key File Paths

| What | Path |
|------|------|
| Workspace config | `Cargo.toml` |
| Core types | `crates/mobius-core/src/experiment.rs` |
| Schema | `crates/mobius-core/src/schema.rs` |
| Compute backends | `crates/mobius-core/src/compute.rs` |
| JSONL store | `crates/mobius-core/src/store/jsonl.rs` |
| Evaluator | `crates/mobius-bench/src/evaluator.rs` |
| Agent loop | `crates/mobius-claw/src/agent.rs` |
| Strategy | `crates/mobius-claw/src/strategy.rs` |
| Learning system | `crates/mobius-claw/src/learning.rs` + `learning_store.rs` |
| Hooks | `crates/mobius-claw/src/hooks.rs` |
| CLI entry | `crates/mobius-cli/src/main.rs` |
