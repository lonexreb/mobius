<div align="center">

<img src="docs/mobius-strip.svg" alt="Mobius Strip" width="200"/>

```
 __  __  ___  ___ ___ _   _ ___
|  \/  |/ _ \| _ )_ _| | | / __|
| |\/| | (_) | _ \| || |_| \__ \
|_|  |_|\___/|___/___|\___/|___/
```

**The only Rust-native framework for autonomous ML experimentation.**

*One loop. One twist. Every pass learns.*

[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/Tests-161_passing-brightgreen)]()
[![Strategies](https://img.shields.io/badge/Strategies-10-blue)]()
[![License](https://img.shields.io/badge/License-MIT%20%7C%20Apache--2.0-blue)]()
[![Phase](https://img.shields.io/badge/Phase-8%20Complete-purple)]()

</div>

---

## Why Mobius?

Every ML experiment framework is Python. Mobius is **Rust** — single binary, no dependency hell, fastest wall-clock optimization, memory-safe parallelism, and MCP-native AI integration.

| vs. | Mobius Advantage |
|-----|----------------|
| **Optuna** | 10 strategies (incl. CMA-ES, Hyperband, PBT), Rust speed, no GIL, single binary |
| **Ray Tune** | Zero setup complexity, no cluster needed, same ASHA/PBT/Hyperband algorithms |
| **W&B Sweeps** | Free, local-first, parameter importance built-in, ask-and-tell API |
| **Ax/BoTorch** | Lighter weight, no PyTorch dependency, outcome constraints, configurable params |

### Inspirations

| Source | What We Took | How Mobius Extends It |
|--------|-------------|---------------------|
| [Karpathy AutoResearch](https://github.com/karpathy/autoresearch) | Edit-run-eval-keep/revert loop | Gradient signals that *learn* which params help |
| [AIDE / WecoAI](https://github.com/WecoAI/aideml) | Tree-search over hypothesis space | UCB1 tree search + 9 more pluggable strategies |
| [Meta REA](https://engineering.fb.com/) | Multi-day agent orchestration | Persistent state, budget guards, regression hooks |
| [MCP Protocol](https://github.com/modelcontextprotocol/rust-sdk) | Tool integration standard | Native MCP server — any AI agent can drive experiments |

---

## Quick Start

```bash
# Build
git clone https://github.com/lonexreb/mobius.git
cd mobius && cargo build --release

# Initialize
cargo run --release -- init              # Creates mobius.toml

# Run experiments
cargo run --release -- run --config '{"learning_rate": 0.01}'
cargo run --release -- suggest           # AI-powered suggestion
cargo run --release -- sweep --spec '{"lr": [0.001, 0.01, 0.1], "bs": [16, 32, 64]}'

# Autonomous agent (10 strategies available)
cargo run --release -- agent --budget 20.0 --strategy auto
cargo run --release -- agent --strategy hyperband --store sqlite

# Ask-and-tell interface (for external pipelines)
cargo run --release -- ask --strategy tpe --metric f1
cargo run --release -- tell --config '{"lr": 0.01}' --metrics '{"f1": 0.85}'

# Analysis
cargo run --release -- importance        # Parameter importance rankings
cargo run --release -- compare exp-001 exp-002
cargo run --release -- export --format csv --output results.csv

# Monitor
cargo run --release -- status            # Best config + KPI gap
cargo run --release -- history --last 20
cargo run --release -- dashboard         # Live TUI dashboard
```

---

## 10 Optimization Strategies

| Strategy | CLI Flag | Type | Best For |
|----------|----------|------|----------|
| **GradientGuidedTuning** | `gradient_guided` | Gradient-based | Default, learns from each run |
| **TPE** | `tpe` | Bayesian (log-scale aware) | Mixed params, 10-100 trials |
| **CMA-ES** | `cmaes` | Evolution strategy (log-scale aware) | Continuous params, low-dimensional |
| **Hyperband** | `hyperband` | Multi-fidelity | Expensive experiments, 40-70% cost savings |
| **NSGA-II** | `nsga2` | Multi-objective | Multiple conflicting metrics |
| **UCB1 Tree Search** | `ucb1` | AIDE-style exploration | Hypothesis exploration |
| **PBT** | `pbt` | Population-based | Dynamic hyperparameter schedules |
| **AutoStrategy** | `auto` | Meta-selector | Don't know which to pick |
| **Grid Search** | `grid` | Exhaustive | Small search spaces |
| **Random Search** | `random` | Baseline | Initial exploration |

**Auto-features:** TPE and CMA-ES automatically detect log-scale parameters (learning rates, weight decay) and operate in log-space for better sampling.

---

## The Agent Loop

```
   ┌──────────┐
   │  ORIENT  │ Load history, find best, check budget
   └────┬─────┘
        ▼
   ┌──────────┐
   │ PROPOSE  │ Strategy.suggest() → next config (or dequeue pending)
   └────┬─────┘
        ▼
   ┌──────────┐
   │ EXECUTE  │ Pre-hooks → ComputeBackend → parse output (with retry)
   └────┬─────┘
        ▼
   ┌──────────┐
   │ EVALUATE │ Post-hooks: regression, overfitting, constraint checks
   └────┬─────┘
        ▼
   ┌──────────┐
   │  LEARN   │ Extract gradient signals → LearningStore
   └────┬─────┘
        ▼
   ┌──────────┐     ┌─────────────────────────────┐
   │  DECIDE  │────►│ Continue | Switch | Stop     │
   └──────────┘     └─────────────────────────────┘
```

**Stop conditions:** targets met, budget exhausted, max iterations, plateau, or Ctrl-C.

**Safety hooks:** Budget check (pre), regression detection (post), overfitting detection (post), outcome constraints (post).

---

## Configuration

```toml
[project]
name = "my-ml-project"

[experiment]
budget_usd = 50.0
cost_per_run = 1.0
command = "python train.py"
timeout_secs = 600

[experiment.targets]
f1 = 0.90
precision = 0.85

[experiment.sweep_space]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
batch_size = [16, 32, 64, 128]
dropout = [0.1, 0.2, 0.3, 0.5]

[experiment.env_map]
learning_rate = "TRAIN_LR"
batch_size = "TRAIN_BS"

[bench]
match_window = 6.0

[[bench.dimensions]]
name = "Detection"
weight = 0.50
scorer = "f1"

[[bench.dimensions]]
name = "Classification"
weight = 0.30
scorer = "accuracy"

[[bench.dimensions]]
name = "Timestamp"
weight = 0.20
scorer = "timestamp_mae"

[agent]
strategies = ["auto", "tpe", "gradient_guided"]
plateau_window = 5
plateau_threshold = 0.02
max_retries = 2

[agent.strategy_params]
tpe_gamma = 0.25
cmaes_population_size = 10
pbt_population_size = 8
hyperband_max_resource = 81
hyperband_eta = 3
nsga_objectives = ["f1", "precision"]

[agent.strategy_params.constraints_upper]
latency_ms = 100.0
memory_gb = 8.0
```

---

## Architecture

```
mobius/
  Cargo.toml                 # Workspace (edition 2024, LTO release profile)
  crates/
    mobius-core/              # Types, config, storage (JSONL + SQLite), compute, parsing
    mobius-bench/             # Evaluation: matchers, dimension scorers, multi-dim evaluator
    mobius-claw/              # Agent: 10 strategies, 4 hooks, 2 pruners, learning, importance
    mobius-cli/               # CLI: 17 commands, colored output, progress bars, TUI dashboard
    mobius-mcp/               # MCP server: 10 tools, 3 resources, rmcp over stdio
```

### Crate Dependency Graph

```
mobius-core  (no internal deps)
    ├── mobius-bench  (core)
    │       └── mobius-claw  (core + bench)
    │               └── mobius-cli  (core + bench + claw)
    └── mobius-mcp   (core + bench + claw)
```

---

## CLI Commands (17)

| Command | Description |
|---------|------------|
| `init` | Generate `mobius.toml` template |
| `run` | Run a single experiment with config overrides |
| `suggest` | Get AI-powered config suggestion |
| `sweep` | Parameter sweep (sequential or `--parallel`) |
| `agent` | Autonomous agent loop (`--strategy`, `--pruning`, `--store`) |
| `evaluate` | Score predictions against ground truth |
| `status` | Show best config and target gaps |
| `history` | Show recent experiments |
| `dashboard` | Live TUI dashboard with sparklines and progress |
| `importance` | Parameter importance rankings (fANOVA) |
| `ask` | Get suggestion as JSON (ask-and-tell) |
| `tell` | Record external result (ask-and-tell) |
| `enqueue` | Queue a specific config for the agent |
| `compare` | Side-by-side experiment comparison |
| `export` | Export history to CSV or JSON |

---

## Extensibility

Everything is a trait. Swap any component:

| Trait | Implementations | You Can Add |
|-------|----------------|-------------|
| `ExperimentStore` | `JsonlStore`, `SqliteStore` | Postgres, S3, DynamoDB |
| `ComputeBackend` | `SubprocessBackend`, `SyncAdapter`, `ParallelBackend` | Modal, RunPod, SSH |
| `Strategy` | 10 built-in (GGT, TPE, CMA-ES, NSGA-II, UCB1, PBT, Auto, Hyperband, Grid, Random) | Bayesian GP, BOHB |
| `DimensionScorer` | F1, Accuracy, Timestamp MAE | Custom metrics |
| `Hook` | Budget, Regression, Overfitting, Constraints | Slack alerts, logging |
| `Pruner` | ASHA, Median | Percentile, Threshold |

---

## Performance

- **Release profile:** LTO + single codegen unit + stripped binary
- **SQLite:** WAL mode, NORMAL sync, 64MB cache, prepared statement caching
- **Parallel computation:** Rayon-parallelized TPE KDE and NSGA-II crowding distance
- **Log-scale sampling:** Auto-detects parameters spanning >2 orders of magnitude
- **Real timeout enforcement:** Processes killed on timeout (not just checked after)

---

## Testing

```bash
cargo test --workspace          # 161 tests (24 core + 4 bench + 74 claw + 25 cli + 34 mcp)
cargo clippy --workspace --all-targets -- -D warnings   # Zero warnings
cargo fmt --all --check         # Consistent formatting
```

---

## Roadmap

| Phase | Status | Highlights |
|-------|--------|-----------|
| **1-2: Core + Agent** | COMPLETE | Types, storage, eval, strategies, agent loop, 8 commands |
| **3-4: MCP + Production** | COMPLETE | MCP server, parallel sweeps, strategy switching, CI |
| **5: Multi-Objective** | COMPLETE | NSGA-II, ASHA pruning, live TUI dashboard |
| **6: Tree Search + SQLite** | COMPLETE | UCB1, SqliteStore, `--store` flag |
| **7: Best-in-Class** | COMPLETE | CMA-ES, AutoStrategy, PBT, importance, colors, ask/tell, compare |
| **8: ML Power Features** | COMPLETE | Hyperband, log-scale, MedianPruner, constraints, enqueue, export, config params |
| **9: SDK + Remote** | PLANNED | Python/Go SDKs, cloud backends, WASM plugins, distributed agents |

---

## License

MIT OR Apache-2.0

---

<div align="center">
<sub>Built with Rust. The only Rust-native HPO framework. 10 strategies, 17 commands, 161 tests, zero Python required.</sub>
</div>
