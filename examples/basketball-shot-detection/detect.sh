#!/usr/bin/env bash
# Simulated basketball shot detector for the Mobius example.
#
# Reads three hyperparameters from env vars:
#   DETECT_THRESHOLD  — confidence cutoff for emitting a detection (0.0-1.0)
#   LOOKBACK_WINDOW   — frames of temporal context used by the detector (1-15)
#   NMS_SECONDS       — non-max-suppression window for back-to-back detections
#
# Outputs JSON metrics to stdout (this is what Mobius parses).
# Also writes predictions.json so you can run `mobius evaluate` afterwards.
#
# The response surface peaks near:
#   DETECT_THRESHOLD=0.50, LOOKBACK_WINDOW=5, NMS_SECONDS=1.0  ->  f1 ~ 0.91
# producing realistic precision/recall/timestamp_mae/bench_score values.

set -euo pipefail

THRESH="${DETECT_THRESHOLD:-0.50}"
LOOKBACK="${LOOKBACK_WINDOW:-5}"
NMS="${NMS_SECONDS:-1.0}"

python3 - "$THRESH" "$LOOKBACK" "$NMS" <<'PY'
import sys, json, hashlib, math, pathlib

thresh   = float(sys.argv[1])
lookback = int(float(sys.argv[2]))
nms      = float(sys.argv[3])

# Smooth quadratic surface centered on the optimum.
peak_thresh, peak_lookback, peak_nms = 0.50, 5, 1.0
f1 = (
    0.91
    - 1.20 * (thresh   - peak_thresh)   ** 2
    - 0.006 * (lookback - peak_lookback) ** 2
    - 0.040 * (nms      - peak_nms)     ** 2
)

# Deterministic noise from param hash so each config is reproducible.
key = f"{thresh:.4f}-{lookback}-{nms:.3f}"
h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
noise = ((h % 200) - 100) * 0.0008

f1 = max(0.05, min(0.99, f1 + noise))
precision = max(0.05, min(0.99, f1 + 0.04 - 0.10 * (peak_thresh - thresh)))
recall    = max(0.05, min(0.99, f1 - 0.03 + 0.10 * (peak_thresh - thresh)))

# Timestamp accuracy degrades as lookback shrinks (less context = jitter).
timestamp_mae = max(0.05, 0.18 + 0.04 * abs(lookback - peak_lookback) - 0.01 * lookback)

# Composite bench score (matches mobius weighted dimensions: 0.5*F1 + 0.3*Acc + 0.2*Timestamp).
classification_acc = max(0.30, min(0.98, 0.78 + 0.20 * (f1 - 0.5)))
timestamp_score = max(0.0, 1.0 - timestamp_mae)  # higher is better
bench_score = 100.0 * (0.50 * f1 + 0.30 * classification_acc + 0.20 * timestamp_score)

# Write predictions.json so `mobius evaluate predictions.json ground_truth.json` can run.
# Slightly perturb timestamps and drop low-confidence detections to mimic the threshold.
gt_path = pathlib.Path(__file__).parent / "ground_truth.json"
out_path = pathlib.Path("predictions.json")
try:
    gt = json.loads(gt_path.read_text())["shots"]
except FileNotFoundError:
    gt = []

# Threshold-driven recall / precision balance.
keep_prob = max(0.10, min(1.0, 1.0 - abs(thresh - peak_thresh)))
preds = []
for i, s in enumerate(gt):
    if (h + i * 7919) % 1000 / 1000.0 > keep_prob:
        continue
    jitter = ((h >> i) % 100 - 50) * 0.004 * max(1, abs(lookback - peak_lookback))
    preds.append({
        "timestamp": round(s["time_sec"] + jitter, 3),
        "label":     s["result"],
        "confidence": round(min(0.99, max(0.05, 0.85 - 0.6 * (thresh - 0.3))), 3),
        "shot_type":  s.get("shot_type"),
    })

# Add a couple of false positives when threshold is too low.
if thresh < peak_thresh:
    for k in range(int((peak_thresh - thresh) * 10)):
        preds.append({
            "timestamp": round(60.0 + 50.0 * k + ((h >> k) % 30), 3),
            "label":     "make",
            "confidence": round(thresh + 0.02, 3),
            "shot_type":  "2pt",
        })
preds.sort(key=lambda p: p["timestamp"])
out_path.write_text(json.dumps({"predictions": preds}, indent=2))

print(json.dumps({
    "f1":                round(f1, 4),
    "precision":         round(precision, 4),
    "recall":            round(recall, 4),
    "classification_acc": round(classification_acc, 4),
    "timestamp_mae":     round(timestamp_mae, 4),
    "bench_score":       round(bench_score, 2),
}))
PY
