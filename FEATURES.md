# Mobius Feature Roadmap

```
 _____ _____ _____ _____ _____ _____
|     |     | __  |     |  |  |   __|
| | | |  |  | __ -|-   -|  |  |__   |
|_|_|_|_____|_____|_____|_____|_____|
             FEATURES ROADMAP
```

## Vision

The unified framework for autonomous ML experimentation. Combines autoresearch loops (Karpathy), tree-search optimization (AIDE/WecoAI), production agent orchestration (Meta REA), MCP-native tool integration, and high-performance Rust evaluation.

## Phase 1: Core Framework [COMPLETE]

- [x] `ExperimentResult` / `ExperimentConfig` — domain-agnostic experiment types
- [x] `ExperimentStore` trait + `JsonlStore` — append-only JSONL storage
- [x] `BudgetGuard` — persistent budget tracking with per-run cost
- [x] `ErrorClassifier` — pattern-based error categorization (GPU OOM, quota, timeout)
- [x] `MobiusConfig` — TOML-based project configuration
- [x] `ComputeBackend` trait + `SubprocessBackend` — pluggable execution
- [x] `OutputParser` — JSON + regex metric extraction
- [x] `GroundTruth` / `Prediction` / `BenchResult` — domain-agnostic evaluation schema
- [x] `loader` — auto-detecting JSON format loader
- [x] `pareto_front()` — Pareto-optimal experiment identification
- [x] `macro_average()` / `micro_average()` — cross-segment aggregation

## Phase 2: Agent + Evaluation Engine [COMPLETE]

- [x] `Matcher` trait + `GreedyTimestampMatcher` — prediction-to-GT matching
- [x] `DimensionScorer` trait + 3 scorers (F1, Classification Accuracy, Timestamp MAE)
- [x] `Evaluator` — multi-dimensional weighted evaluation engine
- [x] `Strategy` trait + `GradientGuidedTuning` — gradient-guided parameter optimization
- [x] `LearningStore` — gradient signal computation from experiment history
- [x] `AgentLoop` — 7-step ORIENT-DECIDE autonomous loop
- [x] `PreExecuteHook` / `PostEvaluateHook` traits + 3 built-in hooks
- [x] CLI: init, evaluate, status, history, run, suggest, sweep, agent
- [x] 32 unit tests across workspace

## Phase 3: MCP + Swarm Intelligence [PLANNED]

- [ ] MCP Server (`mobius-mcp`) — expose experiment tools via Model Context Protocol
- [ ] Tree-Search Strategy — AIDE-style UCB1 hypothesis exploration
- [ ] Multi-Strategy Orchestration — automatic phase transitions
- [ ] Swarm Agent Coordination — lead-teammate parallel exploration
- [ ] Remote Compute Backends — Modal, RunPod, SSH

## Phase 4: Production Hardening [PLANNED]

- [ ] Async Agent Loop — tokio-based with progress streaming
- [ ] Database Storage — SQLite + PostgreSQL backends
- [ ] Terminal UI — ratatui-based real-time monitoring
- [ ] Python SDK (`pymobius`) — PyO3 bindings
- [ ] Go SDK (`go-mobius`) — CGo bindings
- [ ] Plugin System — dynamic scorer/strategy loading
- [ ] Distributed Coordination — multi-node agents
- [ ] Security & Isolation — sandboxed execution, credential gateway
