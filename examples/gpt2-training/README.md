# GPT-2 Training (advanced)

A more realistic Mobius example showcasing Phase 7-8 power features:

| Feature | How this example exercises it |
|---------|-------------------------------|
| **Log-scale auto-detection** | `learning_rate` spans `1e-5 .. 3e-3` (5 orders of magnitude); TPE/CMA-ES auto log-scale it |
| **Hyperband multi-fidelity** | `epochs` is the resource axis with rungs `1 → 3 → 9 → 27 → 81`, `eta=3` |
| **Outcome constraints** | `latency_ms_per_token < 25.0`, `peak_memory_gb < 16.0`; constraint hook flags violators |
| **MedianPruner** | `--pruning median` early-stops trials below median val_loss |
| **Strategy chaining** | `["hyperband", "tpe", "cmaes"]` — agent rotates on consecutive reverts |
| **Negative-loss convention** | `val_loss` reported as `-true_loss` so Mobius (max-oriented) can target it directly |

## Prerequisites

- Python 3 (the simulated trainer uses only the stdlib)
- Mobius CLI (`cargo build --release` from repo root)

## Files

| File | Purpose |
|------|---------|
| `mobius.toml` | Sweep space, Hyperband config, outcome constraints |
| `train.sh` | Simulated GPT-2 trainer — parametric loss surface with realistic interactions |

## The simulated response surface

`train.sh` models a recognisable training landscape:

- **Loss is a log-bowl over (`learning_rate`, `weight_decay`)** — peak at `lr ≈ 3e-4`, `wd ≈ 1e-4`
- **LR–batch interaction** — bigger batches need higher learning rate (`peak_lr ∝ √(bs/16)`)
- **Batch-size penalty** — too small is noisy, too large generalises worse (optimum near `bs=24`)
- **Warmup helps high LRs** — bonus only triggers when `lr > peak_lr`
- **Multi-fidelity penalty** — low `epochs` ADDs to loss; full fidelity removes the penalty
- **Resource scaling** — both `latency_ms_per_token` and `peak_memory_gb` scale super-linearly in batch size

Reference points (deterministic, reproducible):

| Config | val_loss | latency | memory | Notes |
|--------|----------|---------|--------|-------|
| Peak, epochs=81 | 2.16 | 19.8 | 8.7 | Full fidelity optimum |
| Peak, epochs=27 | 1.71 | 19.8 | 8.7 | Hyperband promotion candidate |
| Peak, epochs=1  | 0.68 | 19.8 | 8.7 | Cheap screening rung |
| Off-peak (`bs=64`) | -0.42 | 71.3 | 34.7 | Constraint violation + bad loss |

Even at the cheapest fidelity (1 epoch), the peak config still beats the off-peak config (0.68 > -0.42),
which is exactly what makes Hyperband a good fit for this surface.

## Walkthrough

### 1. Single run

```bash
cd examples/gpt2-training
mobius run --config '{"learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 16, "warmup_steps": 500, "epochs": 27}'
```

### 2. Hyperband (best for this problem)

```bash
mobius agent --strategy hyperband --budget 20.0
```

Hyperband first runs many short trials (1 epoch), promotes the top `1/eta` to 3 epochs, etc.
Expect ~40-70% wall-clock savings vs naive TPE.

### 3. TPE with log-scale awareness

```bash
mobius agent --strategy tpe --budget 20.0
```

TPE auto-detects that `learning_rate` and `weight_decay` span >2 orders of magnitude and
fits its kernel density estimator in log-space.

### 4. CMA-ES on continuous params

```bash
mobius agent --strategy cmaes --budget 20.0
```

Best when the search is dominated by smoothly-varying continuous knobs.

### 5. Combine with MedianPruner

```bash
mobius agent --strategy tpe --pruning median --budget 20.0
```

Trials performing below the running median val_loss are killed early.

### 6. Inspect what mattered

```bash
mobius importance               # fANOVA-style ranking
mobius compare exp-001 exp-007  # Side-by-side delta
mobius export --format csv --output gpt2-sweep.csv
mobius dashboard                # Live TUI
```

### 7. Warm-start from expert knowledge

```bash
mobius enqueue --config '{"learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 16, "warmup_steps": 500, "epochs": 27}'
mobius agent --strategy tpe
```

The enqueued config is dequeued by the agent before any strategy proposal.

### 8. Ask-and-tell (external pipeline integration)

```bash
# Get a suggestion as JSON
mobius ask --strategy tpe --metric val_loss

# Run the suggestion in your own pipeline, then report back:
mobius tell \
  --config '{"learning_rate": 1e-4, "weight_decay": 1e-5, "batch_size": 32, "warmup_steps": 100, "epochs": 27}' \
  --metrics '{"val_loss": 1.85, "val_accuracy": 0.91, "latency_ms_per_token": 24.0, "peak_memory_gb": 12.5}'
```

## Constraint behaviour

The constraint hook treats `latency_ms_per_token > 25` and `peak_memory_gb > 16` as warnings
in the default config. Edit `mobius.toml` to make them blocking:

```toml
[agent.strategy_params]
constraints_strict = true
```

(or write a custom `PostEvaluateHook` returning `Decision::Revert` — see
`crates/mobius-claw/src/hooks.rs`.)
