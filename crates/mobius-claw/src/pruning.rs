//! ASHA-style trial pruning for the agent loop.
//!
//! Compares each completed trial against a quantile threshold of prior trials.
//! Trials performing below the threshold at their "rung" are marked for pruning,
//! saving budget for more promising directions.

use mobius_core::experiment::ExperimentResult;

/// Asynchronous Successive Halving (ASHA) trial pruner.
///
/// After each agent iteration, compares the current metric against
/// a percentile threshold derived from history. Trials in the bottom
/// `(1 - 1/eta)` quantile are pruned.
pub struct AshaPruner {
    /// Reduction factor (eta). Keep top 1/eta trials. Default: 3.
    pub reduction_factor: usize,
    /// Minimum iterations before pruning kicks in.
    pub min_iterations: usize,
}

impl AshaPruner {
    pub fn new(reduction_factor: usize, min_iterations: usize) -> Self {
        Self {
            reduction_factor,
            min_iterations,
        }
    }

    /// Should this trial direction be pruned?
    ///
    /// Returns `true` if `current_metric` falls below the survival threshold
    /// computed from history at the current rung.
    pub fn should_prune(
        &self,
        iteration: usize,
        current_metric: f64,
        history: &[ExperimentResult],
        metric_name: &str,
    ) -> bool {
        // Don't prune before minimum iterations
        if iteration < self.min_iterations {
            return false;
        }

        // Only prune at rung boundaries
        if !iteration.is_multiple_of(self.reduction_factor) {
            return false;
        }

        // Need history to compute threshold
        if history.is_empty() {
            return false;
        }

        // Collect all metric values from history
        let mut values: Vec<f64> = history
            .iter()
            .filter_map(|r| r.metrics.get(metric_name).copied())
            .collect();

        if values.is_empty() {
            return false;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Survival threshold: top 1/eta survive
        // So prune if below the (1 - 1/eta) percentile
        let prune_ratio = 1.0 - 1.0 / self.reduction_factor as f64;
        let threshold_idx = ((values.len() as f64 * prune_ratio).floor() as usize)
            .min(values.len().saturating_sub(1));
        let threshold = values[threshold_idx];

        current_metric < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::{ExperimentConfig, ExperimentStatus};
    use std::collections::HashMap;

    fn make_result(f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        ExperimentResult {
            id: "test".into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: HashMap::new(),
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
    fn test_pruner_keeps_good_trial() {
        let pruner = AshaPruner::new(3, 3);
        let history: Vec<_> = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            .into_iter()
            .map(make_result)
            .collect();
        // 0.8 is well above threshold — should not prune
        assert!(!pruner.should_prune(6, 0.8, &history, "f1"));
    }

    #[test]
    fn test_pruner_kills_bad_trial() {
        let pruner = AshaPruner::new(3, 3);
        let history: Vec<_> = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            .into_iter()
            .map(make_result)
            .collect();
        // 0.3 is in the bottom — should prune at rung 6
        assert!(pruner.should_prune(6, 0.3, &history, "f1"));
    }

    #[test]
    fn test_pruner_respects_min_iterations() {
        let pruner = AshaPruner::new(3, 5);
        let history: Vec<_> = vec![0.5, 0.6, 0.7, 0.8]
            .into_iter()
            .map(make_result)
            .collect();
        // Iteration 3 < min_iterations 5 — should not prune even with bad metric
        assert!(!pruner.should_prune(3, 0.1, &history, "f1"));
    }

    #[test]
    fn test_pruner_only_at_rung_boundaries() {
        let pruner = AshaPruner::new(3, 3);
        let history: Vec<_> = vec![0.5, 0.6, 0.7, 0.8]
            .into_iter()
            .map(make_result)
            .collect();
        // Iteration 4 is not a rung boundary (4 % 3 != 0) — should not prune
        assert!(!pruner.should_prune(4, 0.1, &history, "f1"));
        // Iteration 6 IS a rung boundary — should prune
        assert!(pruner.should_prune(6, 0.1, &history, "f1"));
    }
}
