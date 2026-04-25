#!/usr/bin/env bash
# Simulated GPT-2 training run for the Mobius example.
#
# Models a realistic ML training response surface:
#   * val_loss is a smooth log-bowl over learning_rate around lr=3e-4
#   * weight_decay has a mild quadratic optimum near 1e-4
#   * larger batches need higher LR (interaction term)
#   * more epochs help but with diminishing returns (Hyperband-friendly)
#   * latency / memory scale with batch_size
#
# Reads env vars: LR, WEIGHT_DECAY, BATCH_SIZE, WARMUP_STEPS, EPOCHS
# Outputs JSON metrics on stdout — Mobius parses these for the agent.

set -euo pipefail

LR="${LR:-3e-4}"
WD="${WEIGHT_DECAY:-1e-4}"
BS="${BATCH_SIZE:-16}"
WARMUP="${WARMUP_STEPS:-100}"
EPOCHS="${EPOCHS:-3}"

python3 - "$LR" "$WD" "$BS" "$WARMUP" "$EPOCHS" <<'PY'
import sys, json, math, hashlib

lr, wd  = float(sys.argv[1]), float(sys.argv[2])
bs, ws  = int(float(sys.argv[3])), int(float(sys.argv[4]))
epochs  = int(float(sys.argv[5]))

# ---------------------------------------------------------------------------
# Validation loss: log-bowl over (lr, wd) with a batch-size interaction.
# ---------------------------------------------------------------------------
peak_lr = 3e-4 * (bs / 16) ** 0.5            # bigger batches want larger LR
log_lr_off = math.log10(lr) - math.log10(peak_lr)
lr_penalty = 0.85 * (log_lr_off ** 2)        # parabolic in log-space

peak_wd = 1e-4
log_wd_off = math.log10(wd) - math.log10(peak_wd)
wd_penalty = 0.20 * (log_wd_off ** 2)

# Batch size: too-small batches are noisy, too-large batches generalize worse.
bs_penalty = 0.0008 * (bs - 24) ** 2

# Warmup helps for high LRs only.
warmup_bonus = 0.05 * min(1.0, ws / 500.0) * max(0.0, log_lr_off + 0.3)

# Multi-fidelity: low epochs ADD to loss (worse); large epochs leave it alone.
# epoch_factor ranges 0..1 across the Hyperband ladder (1, 3, 9, 27, 81).
epoch_factor = math.log(epochs + 1) / math.log(82.0)
val_loss = -2.20 + lr_penalty + wd_penalty + bs_penalty - warmup_bonus
val_loss = val_loss + 1.8 * (1.0 - epoch_factor)  # additive fidelity penalty

# Deterministic noise per-config for reproducibility
key = f"{lr:.6g}-{wd:.6g}-{bs}-{ws}-{epochs}"
h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
val_loss += ((h % 200) - 100) * 0.0006

val_perplexity = math.exp(val_loss * -1.0) * -1.0   # report negative for "higher is better"
val_acc = max(0.05, min(0.97, 0.55 - 0.20 * val_loss))

# ---------------------------------------------------------------------------
# Resource constraints — these are checked by the outcome constraint hook.
# ---------------------------------------------------------------------------
# Latency in ms per generated token (decoder-only inference style).
latency_ms_per_token = 6.0 + 0.7 * bs + 0.04 * (bs ** 1.5)
# Peak GPU memory in GiB.
peak_memory_gb = 2.0 + 0.30 * bs + 0.04 * (bs ** 1.4)

# Negate val_loss too so Mobius (max-oriented) can target it directly.
print(json.dumps({
    "val_loss":              round(-val_loss, 4),       # positive = good
    "val_perplexity":        round(val_perplexity, 3),  # negative; closer to 0 is good
    "val_accuracy":          round(val_acc, 4),
    "latency_ms_per_token":  round(latency_ms_per_token, 2),
    "peak_memory_gb":        round(peak_memory_gb, 2),
    "epochs_run":            epochs,
}))
PY
