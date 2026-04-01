# /bench — Run Evaluation

Evaluate predictions against ground truth using the Mobius bench engine.

```bash
mobius evaluate --predictions <path> --ground-truth <path>
```

Loads predictions and GT (auto-detecting JSON format), matches via configured matcher, scores each dimension, computes weighted bench score (0-100).

## Configuration (`mobius.toml`)

```toml
[bench]
match_window = 6.0

[[bench.dimensions]]
name = "Detection"
weight = 0.40
scorer = "f1"
```

## Available Scorers

- `f1` — Detection quality (precision/recall/F1)
- `accuracy` — Label correctness on matched pairs
- `timestamp_mae` — Temporal precision (<=1s=100 ... >12s=0)

## Grading: Fail (<50), Pass (50-69), Good (70-84), Excellent (85+)
