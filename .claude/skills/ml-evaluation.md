---
name: ml-evaluation
description: ML evaluation concepts for Mobius. Use when working on matchers, scorers, evaluators, or bench scoring.
---

# ML Evaluation in Mobius

## Matching: Predictions to Ground Truth

Before scoring, predictions must be aligned to ground truth:
- **Match window**: max temporal distance for valid match (default: 6.0s)
- **Greedy algorithm**: for each GT item, find closest unmatched prediction within window
- **Result**: matched pairs, false positives (unmatched preds), missed (unmatched GT)

## Multi-Dimensional Scoring

| Dimension | Scorer | Measures |
|-----------|--------|----------|
| Detection | F1Scorer | precision + recall |
| Classification | ClassificationAccuracyScorer | label correctness |
| Timestamp | TimestampMaeScorer | temporal accuracy |

Final bench_score = weighted sum of dimension scores.

## Grading

| Grade | Range |
|-------|-------|
| Fail | 0-49 |
| Pass | 50-69 |
| Good | 70-84 |
| Excellent | 85-100 |

## Key Metrics

- **F1**: `2 * P * R / (P + R)`
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **MAE**: average temporal distance between matched pairs

## Cross-Segment Validation

Overfitting detected when per-segment spread > threshold (default: 15pp).
Macro-average = mean of per-segment. Micro-average = aggregate TP/FP/FN.

## Gradient-Guided Optimization

```
ball_conf 0.25->0.28, f1 delta: +0.03
ball_conf 0.28->0.30, f1 delta: +0.02
=> ParamGradient: IncreaseHelps, avg=+0.025, confidence=Medium
```

## Plateau Detection

Range of metric over last N experiments < threshold -> plateau.
Response: explore untried dims, expand ranges, switch strategy phase.
