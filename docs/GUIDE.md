# Mobius Usage Guide

## Installation

**From source:**

```bash
git clone https://github.com/lonexreb/mobius.git
cd mobius
cargo install --path crates/mobius-cli
cargo install --path crates/mobius-mcp
```

## Quick Start

```bash
# 1. Initialize project
mobius init

# 2. Edit mobius.toml with your experiment config
#    Set sweep_space, targets, and command

# 3. Run a single experiment
mobius run --config '{"learning_rate": 0.01}'

# 4. Get a suggestion
mobius suggest

# 5. Run a parameter sweep
mobius sweep --spec '{"learning_rate": [0.001, 0.01, 0.1]}'

# 6. Start autonomous agent
mobius agent --budget 50.0 --strategy gradient_guided
```

## Configuration

Mobius is configured through `mobius.toml`. Run `mobius init` to generate a template, then edit it for your project.

See `examples/mobius.toml.example` for a fully annotated configuration file.

### Key Sections

**`[project]`** - Project metadata (name, version).

**`[experiment]`** - Budget, cost per run, target metrics, and the parameter sweep space.

**`[bench]`** - Evaluation settings: match window, scoring dimensions (F1, accuracy, timestamp MAE).

**`[compute]`** - Backend type (currently `"local"` subprocess).

**`[agent]`** - Strategy selection, plateau detection window and threshold.

## Running Experiments

### Single Run

```bash
mobius run --config '{"learning_rate": 0.01, "batch_size": 32}'
```

Executes the configured command with parameters passed as environment variables. Results are stored in `~/.mobius/history.jsonl`.

### Parameter Sweep

```bash
mobius sweep --spec '{"learning_rate": [0.001, 0.01, 0.1], "batch_size": [16, 32]}'
```

Runs all combinations (cartesian product) and ranks by f1 score.

### Autonomous Agent

```bash
mobius agent --budget 50.0 --strategy gradient_guided
```

Runs the 7-step agent loop (Orient -> Research -> Propose -> Execute -> Evaluate -> Learn -> Decide) until a stop condition is met:
- All targets achieved
- Budget exhausted
- Max iterations reached
- Plateau detected

## Strategy Selection

Three strategies are available:

| Strategy | Flag | Description |
|----------|------|-------------|
| Gradient-Guided | `--strategy gradient_guided` | Uses learning signals to guide parameter changes (default) |
| Random Search | `--strategy random` | Pure random sampling from sweep space |
| Grid Search | `--strategy grid` | Systematic enumeration of all combinations |

The strategy can also be set in `mobius.toml`:

```toml
[agent]
strategies = ["gradient_guided"]
```

## Viewing Results

```bash
# Current status with target gaps
mobius status

# Recent experiment history
mobius history --last 20

# Get next suggestion based on learnings
mobius suggest
```

## Evaluation

```bash
mobius evaluate --predictions preds.json --ground-truth gt.json
```

Evaluates predictions against ground truth using configurable dimensions (Detection F1, Classification Accuracy, Timestamp MAE).

## MCP Integration

Mobius can be used as an MCP server, allowing AI assistants to run experiments autonomously. See [MCP_SETUP.md](MCP_SETUP.md) for configuration instructions.

## State Files

Mobius stores state in `~/.mobius/`:

| File | Contents |
|------|----------|
| `history.jsonl` | Append-only experiment results |
| `learnings.jsonl` | Append-only learning entries (parameter gradients) |
| `budget.json` | Current spend tracking |
