use mobius_core::schema::{GroundTruth, MatchResult, MatchedPair, Prediction};

/// Trait for matching predictions to ground truth.
///
/// Generalizes `paloa_bench/evaluator.py:_match_shots`.
pub trait Matcher: Send + Sync {
    fn match_predictions(
        &self,
        predictions: &[Prediction],
        ground_truth: &[GroundTruth],
        window: f64,
    ) -> MatchResult;
}

/// Greedy nearest-neighbor timestamp matcher.
///
/// Default matcher, same algorithm as paloa_bench: for each GT item,
/// find the closest unmatched prediction within the match window.
pub struct GreedyTimestampMatcher;

impl Matcher for GreedyTimestampMatcher {
    fn match_predictions(
        &self,
        predictions: &[Prediction],
        ground_truth: &[GroundTruth],
        window: f64,
    ) -> MatchResult {
        let mut used_pred = vec![false; predictions.len()];
        let mut matched = Vec::new();
        let mut missed = Vec::new();

        // Sort GT by timestamp for deterministic matching
        let mut gt_sorted: Vec<(usize, &GroundTruth)> =
            ground_truth.iter().enumerate().collect();
        gt_sorted.sort_by(|a, b| {
            let ta = a.1.timestamp.unwrap_or(0.0);
            let tb = b.1.timestamp.unwrap_or(0.0);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (_gt_idx, gt) in &gt_sorted {
            let gt_time = gt.timestamp.unwrap_or(0.0);
            let mut best_idx: Option<usize> = None;
            let mut best_delta = f64::MAX;

            for (pred_idx, pred) in predictions.iter().enumerate() {
                if used_pred[pred_idx] {
                    continue;
                }
                let pred_time = pred.timestamp.unwrap_or(0.0);
                let delta = (pred_time - gt_time).abs();
                if delta <= window && delta < best_delta {
                    best_delta = delta;
                    best_idx = Some(pred_idx);
                }
            }

            if let Some(idx) = best_idx {
                used_pred[idx] = true;
                matched.push(MatchedPair {
                    ground_truth: (*gt).clone(),
                    prediction: predictions[idx].clone(),
                    time_delta: Some(best_delta),
                });
            } else {
                missed.push((*gt).clone());
            }
        }

        let false_positives: Vec<Prediction> = predictions
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_pred[*i])
            .map(|(_, p)| p.clone())
            .collect();

        MatchResult {
            matched,
            false_positives,
            missed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn gt(id: &str, ts: f64, label: &str) -> GroundTruth {
        GroundTruth {
            id: id.to_string(),
            timestamp: Some(ts),
            label: label.to_string(),
            attributes: HashMap::new(),
            tags: vec![],
        }
    }

    fn pred(ts: f64, label: &str) -> Prediction {
        Prediction {
            timestamp: Some(ts),
            label: Some(label.to_string()),
            confidence: Some(0.9),
            source: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn test_perfect_matching() {
        let matcher = GreedyTimestampMatcher;
        let gts = vec![gt("1", 10.0, "make"), gt("2", 20.0, "miss")];
        let preds = vec![pred(10.5, "make"), pred(19.8, "miss")];

        let result = matcher.match_predictions(&preds, &gts, 6.0);
        assert_eq!(result.matched.len(), 2);
        assert_eq!(result.false_positives.len(), 0);
        assert_eq!(result.missed.len(), 0);
    }

    #[test]
    fn test_with_false_positives() {
        let matcher = GreedyTimestampMatcher;
        let gts = vec![gt("1", 10.0, "make")];
        let preds = vec![pred(10.5, "make"), pred(30.0, "miss")];

        let result = matcher.match_predictions(&preds, &gts, 6.0);
        assert_eq!(result.matched.len(), 1);
        assert_eq!(result.false_positives.len(), 1);
        assert_eq!(result.missed.len(), 0);
    }

    #[test]
    fn test_with_missed() {
        let matcher = GreedyTimestampMatcher;
        let gts = vec![gt("1", 10.0, "make"), gt("2", 50.0, "miss")];
        let preds = vec![pred(10.5, "make")];

        let result = matcher.match_predictions(&preds, &gts, 6.0);
        assert_eq!(result.matched.len(), 1);
        assert_eq!(result.false_positives.len(), 0);
        assert_eq!(result.missed.len(), 1);
    }

    #[test]
    fn test_outside_window() {
        let matcher = GreedyTimestampMatcher;
        let gts = vec![gt("1", 10.0, "make")];
        let preds = vec![pred(20.0, "make")];

        let result = matcher.match_predictions(&preds, &gts, 6.0);
        assert_eq!(result.matched.len(), 0);
        assert_eq!(result.false_positives.len(), 1);
        assert_eq!(result.missed.len(), 1);
    }
}
