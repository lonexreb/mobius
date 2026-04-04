//! Parameter importance analysis using simplified fANOVA.
//!
//! Ranks parameters by their impact on the primary metric using
//! between-group variance analysis. For each parameter, experiments
//! are grouped by value, and the variance of group means is computed
//! relative to total variance.

use mobius_core::experiment::ExperimentResult;
use std::collections::HashMap;

/// Importance score for a single parameter.
#[derive(Debug, Clone)]
pub struct ParamImportance {
    /// Parameter name.
    pub param: String,
    /// Normalized importance score (0.0 to 1.0, all scores sum to 1.0).
    pub importance: f64,
    /// Number of unique values observed for this parameter.
    pub num_unique_values: usize,
    /// Range of group means: (min_group_mean, max_group_mean).
    pub metric_range: (f64, f64),
}

/// Compute parameter importance scores for all parameters in sweep space.
///
/// Uses between-group variance analysis (simplified fANOVA): for each
/// parameter, groups experiments by value and computes how much of the
/// total metric variance is explained by that parameter.
pub fn compute_importance(
    history: &[ExperimentResult],
    sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    primary_metric: &str,
) -> Vec<ParamImportance> {
    if history.is_empty() {
        return Vec::new();
    }

    // Collect all metric values
    let all_values: Vec<f64> = history
        .iter()
        .filter_map(|r| r.metrics.get(primary_metric).copied())
        .collect();

    if all_values.len() < 2 {
        return Vec::new();
    }

    let grand_mean: f64 = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let total_variance: f64 = all_values
        .iter()
        .map(|v| (v - grand_mean).powi(2))
        .sum::<f64>()
        / all_values.len() as f64;

    if total_variance < 1e-12 {
        // All metrics identical — no importance to assign
        return sweep_space
            .keys()
            .map(|param| ParamImportance {
                param: param.clone(),
                importance: 0.0,
                num_unique_values: 0,
                metric_range: (grand_mean, grand_mean),
            })
            .collect();
    }

    let mut raw_scores: Vec<(String, f64, usize, f64, f64)> = Vec::new();

    for param in sweep_space.keys() {
        // Group experiments by parameter value
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        for result in history {
            if let Some(metric_val) = result.metrics.get(primary_metric) {
                let key = result
                    .config
                    .parameters
                    .get(param)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "null".to_string());
                groups.entry(key).or_default().push(*metric_val);
            }
        }

        if groups.len() < 2 {
            raw_scores.push((param.clone(), 0.0, groups.len(), grand_mean, grand_mean));
            continue;
        }

        // Compute group means
        let group_means: Vec<f64> = groups
            .values()
            .map(|vals| vals.iter().sum::<f64>() / vals.len() as f64)
            .collect();

        let min_mean = group_means.iter().copied().fold(f64::INFINITY, f64::min);
        let max_mean = group_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Between-group variance
        let between_var: f64 = group_means
            .iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>()
            / group_means.len() as f64;

        let importance_raw = between_var / total_variance;
        raw_scores.push((
            param.clone(),
            importance_raw,
            groups.len(),
            min_mean,
            max_mean,
        ));
    }

    // Normalize scores to sum to 1.0
    let total_raw: f64 = raw_scores.iter().map(|(_, s, _, _, _)| s).sum();
    let norm = if total_raw > 1e-12 { total_raw } else { 1.0 };

    let mut result: Vec<ParamImportance> = raw_scores
        .into_iter()
        .map(|(param, raw, n_vals, min_m, max_m)| ParamImportance {
            param,
            importance: raw / norm,
            num_unique_values: n_vals,
            metric_range: (min_m, max_m),
        })
        .collect();

    // Sort by importance descending
    result.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};

    fn make_result(id: &str, f1: f64, params: Vec<(&str, serde_json::Value)>) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut parameters = HashMap::new();
        for (k, v) in params {
            parameters.insert(k.to_string(), v);
        }
        ExperimentResult {
            id: id.into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters,
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 1.0,
            cost_usd: None,
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    #[test]
    fn test_importance_single_dominant_param() {
        // lr perfectly correlates with f1, bs has no effect
        let history = vec![
            make_result(
                "1",
                0.3,
                vec![
                    ("lr", serde_json::json!(0.01)),
                    ("bs", serde_json::json!(16)),
                ],
            ),
            make_result(
                "2",
                0.3,
                vec![
                    ("lr", serde_json::json!(0.01)),
                    ("bs", serde_json::json!(32)),
                ],
            ),
            make_result(
                "3",
                0.9,
                vec![
                    ("lr", serde_json::json!(0.1)),
                    ("bs", serde_json::json!(16)),
                ],
            ),
            make_result(
                "4",
                0.9,
                vec![
                    ("lr", serde_json::json!(0.1)),
                    ("bs", serde_json::json!(32)),
                ],
            ),
        ];

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );
        sweep.insert(
            "bs".into(),
            vec![serde_json::json!(16), serde_json::json!(32)],
        );

        let result = compute_importance(&history, &sweep, "f1");
        assert_eq!(result.len(), 2);
        // lr should be most important
        assert_eq!(result[0].param, "lr");
        assert!(
            result[0].importance > 0.9,
            "lr importance should be high: {}",
            result[0].importance
        );
        // bs should have ~0 importance
        assert!(
            result[1].importance < 0.1,
            "bs importance should be low: {}",
            result[1].importance
        );
    }

    #[test]
    fn test_importance_empty_history() {
        let sweep = HashMap::new();
        let result = compute_importance(&[], &sweep, "f1");
        assert!(result.is_empty());
    }

    #[test]
    fn test_importance_single_value_param() {
        let history = vec![
            make_result("1", 0.5, vec![("lr", serde_json::json!(0.01))]),
            make_result("2", 0.7, vec![("lr", serde_json::json!(0.01))]),
        ];

        let mut sweep = HashMap::new();
        sweep.insert("lr".into(), vec![serde_json::json!(0.01)]);

        let result = compute_importance(&history, &sweep, "f1");
        assert_eq!(result.len(), 1);
        assert!(result[0].importance.abs() < 1e-10);
    }

    #[test]
    fn test_importance_equal_params() {
        // Both params equally affect the metric
        let history = vec![
            make_result(
                "1",
                0.3,
                vec![("a", serde_json::json!(1)), ("b", serde_json::json!(10))],
            ),
            make_result(
                "2",
                0.7,
                vec![("a", serde_json::json!(1)), ("b", serde_json::json!(20))],
            ),
            make_result(
                "3",
                0.7,
                vec![("a", serde_json::json!(2)), ("b", serde_json::json!(10))],
            ),
            make_result(
                "4",
                0.3,
                vec![("a", serde_json::json!(2)), ("b", serde_json::json!(20))],
            ),
        ];

        let mut sweep = HashMap::new();
        sweep.insert("a".into(), vec![serde_json::json!(1), serde_json::json!(2)]);
        sweep.insert(
            "b".into(),
            vec![serde_json::json!(10), serde_json::json!(20)],
        );

        let result = compute_importance(&history, &sweep, "f1");
        assert_eq!(result.len(), 2);
        // Both should have similar importance (0.5 each)
        assert!(
            (result[0].importance - result[1].importance).abs() < 0.1,
            "Expected similar importance: {} vs {}",
            result[0].importance,
            result[1].importance
        );
    }
}
