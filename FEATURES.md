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

## Phase 3: MCP + Strategy Expansion [COMPLETE]

- [x] MCP Server (`mobius-mcp`) — 10 tools, 3 resources via Model Context Protocol
- [x] `RandomSearch` strategy — pure random parameter sampling
- [x] `GridSearch` strategy — systematic exhaustive search
- [x] `TpeSearch` strategy — Bayesian optimization via Tree-structured Parzen Estimators
- [x] `AsyncComputeBackend` + `SyncAdapter` — async execution via tokio
- [x] 82 tests across workspace

## Phase 4: Production Hardening [COMPLETE]

- [x] Config wiring — `mobius.toml` drives agent/sweep/strategy selection
- [x] Strategy switching — automatic rotation on consecutive reverts
- [x] Parallel sweeps — `--parallel` with configurable `--max-concurrency`
- [x] CI pipeline — fmt + clippy + test in CI
- [x] Example config — `examples/mobius.toml.example` with `mobius init`
- [x] Publish preparation — workspace metadata, crate descriptions, README

## Phase 5: Multi-Objective + Dashboard [COMPLETE]

- [x] `NsgaTwo` — NSGA-II multi-objective optimization with Pareto front evolution
- [x] `AshaPruner` — ASHA trial pruning to early-stop underperforming experiments
- [x] Live TUI dashboard — ratatui-based real-time monitoring of experiment progress
- [x] `--pruning` flag — enable ASHA pruning in agent loop

## Phase 6: Tree Search + SQLite [COMPLETE]

- [x] `UcbTreeSearch` — UCB1 tree-search strategy for AIDE-style hypothesis exploration
- [x] `SqliteStore` — SQLite experiment storage backend with indexed queries
- [x] `--store` CLI flag — choose between `jsonl` (default) and `sqlite` backends
- [x] `--strategy ucb1` — register UCB1/tree_search in strategy factory
- [x] 107 tests across workspace

## Phase 7: Best-in-Class Performance, Algorithms & UX [COMPLETE]

### 7A: Performance & Bug Fixes
- [x] Fix SubprocessBackend timeout — processes now killed on timeout (was running forever)
- [x] SQLite WAL mode + PRAGMAs — `journal_mode=WAL`, `synchronous=NORMAL`, 5-10x write speedup
- [x] SQLite `prepare_cached()` — cached prepared statements for all queries
- [x] SQLite `get_best()` SQL optimization — `json_extract()` + `ORDER BY LIMIT 1` instead of loading all rows
- [x] JSONL `to_writer` optimization — direct serialization to file, no intermediate String allocation
- [x] Release profile — LTO, codegen-units=1, strip, panic=abort for optimized binaries
- [x] Rayon parallel computation — TPE KDE and NSGA-II crowding distance parallelized

### 7B: Competitive Algorithms
- [x] `CmaEs` — Covariance Matrix Adaptation Evolution Strategy (Optuna-competitive)
- [x] `AutoStrategy` — auto-selects best algorithm based on problem characteristics (Optuna AutoSampler)
- [x] `Pbt` — Population-Based Training (Ray Tune-style evolutionary strategy)
- [x] `compute_importance()` — fANOVA-style parameter importance analysis (W&B-competitive)

### 7C: UX Excellence
- [x] Colored CLI output — `owo-colors` style module for consistent formatting
- [x] Progress bars — `indicatif` integration for sweep/agent/run
- [x] `mobius ask` / `mobius tell` — ask-and-tell interface for external pipeline integration
- [x] `mobius compare` — side-by-side experiment comparison with color-coded deltas
- [x] `mobius importance` — parameter importance ranking CLI command
- [x] 152 tests across workspace

## Phase 8: Production ML Power Features [COMPLETE]

### 8A: Optimization Performance
- [x] Log-scale sampling — automatic detection for parameters spanning >2 orders of magnitude (learning rates, weight decay)
- [x] TPE log-space KDE — kernel density estimation in log-space for log-scale parameters
- [x] CMA-ES log-space normalization — covariance adaptation in log-space for better continuous optimization
- [x] Hyperband meta-scheduler — multi-fidelity optimization across multiple successive halving brackets
- [x] MedianPruner — prune trials below median performance (simpler, more robust than ASHA alone)
- [x] Fidelity-aware experiments — metadata-driven resource allocation (epochs, data fraction)

### 8B: Production Essentials
- [x] Configurable strategy parameters — TPE gamma, CMA-ES population, NSGA objectives, Hyperband settings via mobius.toml
- [x] `build_strategy_with_params()` — config-aware strategy factory
- [x] Outcome constraints — upper/lower bounds on metrics (latency < 100ms, memory < 8GB)
- [x] Trial enqueue — `mobius enqueue` to force specific configs (warm-starting, expert knowledge)
- [x] Auto-retry — configurable max_retries for transient experiment failures
- [x] `mobius export` — CSV/JSON export of experiment history
- [x] 159+ tests across workspace

## Phase 9: SDK + Remote Execution [PLANNED]

- [ ] Python SDK (`pymobius`) — PyO3 bindings for core + bench
- [ ] Go SDK (`go-mobius`) — CGo bindings
- [ ] Remote Compute Backends — Modal, RunPod, SSH
- [ ] Plugin System — dynamic scorer/strategy loading via shared libraries
- [ ] Distributed Coordination — multi-node agent swarms
- [ ] Security & Isolation — sandboxed execution, credential gateway
