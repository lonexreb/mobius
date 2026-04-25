# Basketball Shot Detection

A multi-dimensional evaluation example showing the full Mobius surface:
**hyperparameter optimization** + **structured prediction matching** + **weighted bench scoring**.

The simulated detector has three knobs and a clear sweet spot — letting the agent demonstrably converge:

| Param | Search space | Optimum |
|-------|--------------|---------|
| `detection_threshold` | 0.30 – 0.70 | 0.50 |
| `lookback_window`     | 3, 5, 7, 10 | 5    |
| `nms_seconds`         | 0.5 – 2.0   | 1.0  |

At the optimum: `f1 ≈ 0.90`, `bench_score ≈ 88` (Excellent grade).

## Prerequisites

- Python 3 (for the simulated detector)
- Mobius CLI (`cargo build --release` from repo root, or `cargo install --path ../../crates/mobius-cli`)

## Files

| File | Purpose |
|------|---------|
| `mobius.toml` | Project config — sweep space, multi-dim bench, agent strategies |
| `detect.sh` | Simulated detector — reads env-var hyperparams, emits JSON metrics, writes `predictions.json` |
| `ground_truth.json` | 12 labeled shots with `shot_type`, `shooter_jersey`, edge-case tags |
| `predictions.example.json` | Pre-canned predictions for a quick `mobius evaluate` demo |

## Walkthrough

### 1. Single run with the optimum

```bash
cd examples/basketball-shot-detection
mobius run --config '{"detection_threshold": 0.50, "lookback_window": 5, "nms_seconds": 1.0}'
```

You should see something like:

```text
f1=0.898  precision=0.938  recall=0.868  bench_score=88.09
```

### 2. Score the produced predictions

The detector writes `predictions.json` on every run. Score it with the multi-dimensional evaluator:

```bash
mobius evaluate --predictions predictions.json --ground-truth ground_truth.json
```

```text
BENCH SCORE: 100.0 / 100  (Excellent)
  Detection        100.0 x 0.50 =  50.0
  Classification   100.0 x 0.30 =  30.0
  Timestamp        100.0 x 0.20 =  20.0
  Matches: 12 | FPs: 0 | Missed: 0
```

(At the optimum the detector matches all 12 ground-truth shots within the 6-second match window.)

### 3. Parameter sweep

```bash
mobius sweep \
  --spec '{"detection_threshold": [0.30, 0.50, 0.70], "lookback_window": [3, 5, 10]}' \
  --parallel
```

### 4. Autonomous agent

Let Mobius find the optimum by itself. `auto` picks the best strategy based on problem characteristics:

```bash
mobius agent --budget 5.0 --strategy auto
```

Try other strategies (all fully wired):

```bash
mobius agent --strategy tpe                       # Bayesian (TPE with log-scale aware KDE)
mobius agent --strategy hyperband                 # Multi-fidelity, fastest convergence
mobius agent --strategy nsga2                     # Multi-objective (f1 + precision)
mobius agent --strategy gradient_guided --pruning # ASHA-pruned gradient descent
```

### 5. Compare runs and inspect importance

```bash
mobius status
mobius history --last 10
mobius importance         # Which knob matters most? (fANOVA-based ranking)
mobius compare exp-001 exp-002
mobius export --format csv --output sweep.csv
mobius dashboard          # Live TUI sparklines
```

## How `detect.sh` works

The script computes a smooth quadratic response surface around the optimum, adds a small
deterministic hash-based noise term (so each config is reproducible), and writes both:

- **stdout JSON** — `f1`, `precision`, `recall`, `classification_acc`, `timestamp_mae`, `bench_score`. Mobius parses this to score the run.
- **`predictions.json`** — perturbed timestamps + threshold-driven recall/false-positives, so the structured `mobius evaluate` command can be run on the artefacts of any single experiment.

This dual output mirrors a real ML pipeline: the trainer reports its own metrics, and a
separate evaluator scores the predictions against held-out ground truth.

## What this example shows off

- **Multi-dimensional weighted scoring** — F1 (0.50) + Accuracy (0.30) + Timestamp MAE (0.20)
- **Greedy timestamp matching** — predictions paired to GT within `match_window=6.0s`
- **Edge-case tags** — `free_throw`, `fast_break`, `contested` for breakdown analysis
- **Multi-strategy agent loop** — `auto`, `tpe`, `gradient_guided` with revert-driven rotation
- **Reproducibility** — deterministic hash-based noise so re-runs of the same config are stable
