use mobius_core::schema::{DimensionScore, GroundTruth, MatchedPair, Prediction};
use std::collections::HashMap;

/// Trait for scoring a single evaluation dimension.
///
/// Generalizes individual scoring methods from `paloa_bench/evaluator.py`.
pub trait DimensionScorer: Send + Sync {
    fn name(&self) -> &str;
    fn score(
        &self,
        matches: &[MatchedPair],
        false_positives: &[Prediction],
        missed: &[GroundTruth],
    ) -> DimensionScore;
}

/// F1 scorer — detection quality via precision/recall.
///
/// From `paloa_bench/evaluator.py:_score_shot_detection`.
pub struct F1Scorer;

impl DimensionScorer for F1Scorer {
    fn name(&self) -> &str {
        "f1"
    }

    fn score(
        &self,
        matches: &[MatchedPair],
        false_positives: &[Prediction],
        missed: &[GroundTruth],
    ) -> DimensionScore {
        let tp = matches.len() as f64;
        let fp = false_positives.len() as f64;
        let r#fn = missed.len() as f64;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + r#fn > 0.0 {
            tp / (tp + r#fn)
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let score = f1 * 100.0;
        let mut details = HashMap::new();
        details.insert("precision".into(), serde_json::json!(precision));
        details.insert("recall".into(), serde_json::json!(recall));
        details.insert("f1".into(), serde_json::json!(f1));
        details.insert("tp".into(), serde_json::json!(tp as usize));
        details.insert("fp".into(), serde_json::json!(fp as usize));
        details.insert("fn".into(), serde_json::json!(r#fn as usize));

        DimensionScore {
            name: "Detection".to_string(),
            score,
            weight: 0.0, // Set by evaluator
            weighted_score: 0.0,
            details,
        }
    }
}

/// Classification accuracy scorer — label correctness on matched pairs.
///
/// From `paloa_bench/evaluator.py:_score_make_miss`.
pub struct ClassificationAccuracyScorer {
    pub label_field: String,
}

impl ClassificationAccuracyScorer {
    pub fn new(label_field: impl Into<String>) -> Self {
        Self {
            label_field: label_field.into(),
        }
    }

    pub fn on_primary_label() -> Self {
        Self::new("label")
    }
}

impl DimensionScorer for ClassificationAccuracyScorer {
    fn name(&self) -> &str {
        "classification_accuracy"
    }

    fn score(
        &self,
        matches: &[MatchedPair],
        _false_positives: &[Prediction],
        _missed: &[GroundTruth],
    ) -> DimensionScore {
        if matches.is_empty() {
            return DimensionScore {
                name: "Classification".to_string(),
                score: 0.0,
                weight: 0.0,
                weighted_score: 0.0,
                details: HashMap::new(),
            };
        }

        let correct = matches
            .iter()
            .filter(|m| {
                let gt_label = &m.ground_truth.label;
                let pred_label = m.prediction.label.as_deref().unwrap_or("");
                gt_label.eq_ignore_ascii_case(pred_label)
            })
            .count();

        let accuracy = correct as f64 / matches.len() as f64;
        let score = accuracy * 100.0;

        let mut details = HashMap::new();
        details.insert("correct".into(), serde_json::json!(correct));
        details.insert("total".into(), serde_json::json!(matches.len()));
        details.insert("accuracy".into(), serde_json::json!(accuracy));

        DimensionScore {
            name: "Classification".to_string(),
            score,
            weight: 0.0,
            weighted_score: 0.0,
            details,
        }
    }
}

/// Timestamp accuracy scorer — temporal precision on matched pairs.
///
/// From `paloa_bench/evaluator.py:_score_timestamp_accuracy`.
pub struct TimestampMaeScorer;

impl DimensionScorer for TimestampMaeScorer {
    fn name(&self) -> &str {
        "timestamp_mae"
    }

    fn score(
        &self,
        matches: &[MatchedPair],
        _false_positives: &[Prediction],
        _missed: &[GroundTruth],
    ) -> DimensionScore {
        if matches.is_empty() {
            return DimensionScore {
                name: "Timestamp".to_string(),
                score: 0.0,
                weight: 0.0,
                weighted_score: 0.0,
                details: HashMap::new(),
            };
        }

        // Score each match: <=1s=100, <=2s=90, <=3s=80, <=5s=60, <=8s=40, <=12s=20, >12s=0
        let total_score: f64 = matches
            .iter()
            .map(|m| {
                let delta = m.time_delta.unwrap_or(f64::MAX).abs();
                if delta <= 1.0 {
                    100.0
                } else if delta <= 2.0 {
                    90.0
                } else if delta <= 3.0 {
                    80.0
                } else if delta <= 5.0 {
                    60.0
                } else if delta <= 8.0 {
                    40.0
                } else if delta <= 12.0 {
                    20.0
                } else {
                    0.0
                }
            })
            .sum();

        let score = total_score / matches.len() as f64;
        let mae: f64 = matches
            .iter()
            .map(|m| m.time_delta.unwrap_or(0.0).abs())
            .sum::<f64>()
            / matches.len() as f64;

        let mut details = HashMap::new();
        details.insert("mae_seconds".into(), serde_json::json!(mae));
        details.insert("matched_count".into(), serde_json::json!(matches.len()));

        DimensionScore {
            name: "Timestamp".to_string(),
            score,
            weight: 0.0,
            weighted_score: 0.0,
            details,
        }
    }
}
