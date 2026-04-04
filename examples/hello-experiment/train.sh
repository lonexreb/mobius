#!/usr/bin/env bash
# Simulated ML training script for Mobius demo.
# Reads LEARNING_RATE and BATCH_SIZE from env vars.
# Outputs JSON metrics to stdout.
#
# The response surface has an optimum near lr=0.01, bs=32.

LR="${LEARNING_RATE:-0.01}"
BS="${BATCH_SIZE:-32}"

# Deterministic quadratic response surface
python3 -c "
import math, hashlib
lr, bs = $LR, $BS
# Quadratic with peak near lr=0.01, bs=32
f1 = 0.92 - 80.0*(lr - 0.01)**2 - 0.0003*(bs - 32)**2
# Add small deterministic noise from param hash
h = int(hashlib.md5(f'{lr},{bs}'.encode()).hexdigest()[:8], 16)
noise = ((h % 100) - 50) * 0.002
f1 = max(0.0, min(1.0, f1 + noise))
precision = min(1.0, f1 + 0.03)
recall = max(0.0, f1 - 0.02)
print(f'{{\"f1\": {f1:.4f}, \"precision\": {precision:.4f}, \"recall\": {recall:.4f}}}')
"
