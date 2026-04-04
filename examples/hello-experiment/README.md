# Hello Experiment

A minimal working example for Mobius. Uses a simulated training script with a quadratic response surface (optimum at `learning_rate=0.01`, `batch_size=32`).

## Prerequisites

- Python 3 (for the training simulation)
- Mobius CLI (`cargo install --path ../../crates/mobius-cli`)

## Usage

```bash
cd examples/hello-experiment

# Run a single experiment
mobius run --config '{"learning_rate": 0.01, "batch_size": 32}'

# Get a gradient-guided suggestion
mobius suggest

# Run a parameter sweep
mobius sweep --spec '{"learning_rate": [0.001, 0.01, 0.1]}'

# Run a parallel sweep
mobius sweep --spec '{"learning_rate": [0.001, 0.01, 0.1], "batch_size": [16, 32, 64]}' --parallel

# Start autonomous agent
mobius agent --budget 5.0 --strategy gradient_guided

# Check status
mobius status
mobius history --last 10
```

## How it works

`train.sh` reads `LEARNING_RATE` and `BATCH_SIZE` from environment variables (mapped via `env_map` in `mobius.toml`), computes a deterministic f1/precision/recall from a quadratic response surface, and outputs JSON to stdout.

The response surface peaks at `lr=0.01, bs=32` with `f1~0.92`. The agent should converge to this region within a few iterations.
