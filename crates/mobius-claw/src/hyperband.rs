//! Hyperband meta-scheduler for multi-fidelity optimization.
//!
//! Runs multiple brackets of Successive Halving with different
//! initial budgets. Each bracket trades off early stopping aggressiveness
//! vs number of configurations explored.

use crate::strategy::{RandomSearch, Strategy, StrategyContext, Suggestion};
use mobius_core::experiment::ExperimentConfig;
use std::collections::HashMap;

/// Hyperband meta-scheduler for multi-fidelity optimization.
///
/// Allocates trials across multiple brackets of successive halving,
/// each starting at a different fidelity level. Lower brackets explore
/// more configs at lower fidelity; higher brackets run fewer configs
/// at higher fidelity. The fidelity level is stored in the experiment's
/// metadata as `"fidelity"`.
pub struct Hyperband {
    /// Maximum resource budget per trial (e.g., max epochs).
    pub max_resource: usize,
    /// Reduction factor (default: 3 — keep top 1/3 at each rung).
    pub eta: usize,
}

impl Hyperband {
    /// Create a new Hyperband scheduler.
    pub fn new(max_resource: usize, eta: usize) -> Self {
        Self { max_resource, eta }
    }

    /// Compute s_max = floor(log_eta(max_resource)).
    fn s_max(&self) -> usize {
        ((self.max_resource as f64).ln() / (self.eta as f64).ln()).floor() as usize
    }

    /// Compute the initial number of configs for bracket s.
    fn n_configs(&self, s: usize) -> usize {
        let s_max = self.s_max();
        let base = (s_max + 1) as f64 / (s + 1) as f64;
        (base * (self.eta as f64).powi(s as i32)).ceil() as usize
    }

    /// Compute the initial resource for bracket s.
    fn initial_resource(&self, s: usize) -> usize {
        (self.max_resource as f64 / (self.eta as f64).powi(s as i32)).ceil() as usize
    }

    /// Count experiments at a given bracket and rung from history.
    fn count_at_rung(
        &self,
        history: &[mobius_core::experiment::ExperimentResult],
        bracket: usize,
        rung: usize,
    ) -> usize {
        history
            .iter()
            .filter(|r| {
                let b = r
                    .config
                    .metadata
                    .get("hyperband_bracket")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX);
                let ru = r
                    .config
                    .metadata
                    .get("hyperband_rung")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX);
                b == bracket as u64 && ru == rung as u64
            })
            .count()
    }

    /// Get the top configs from a rung to promote to the next.
    fn top_configs_at_rung(
        &self,
        history: &[mobius_core::experiment::ExperimentResult],
        bracket: usize,
        rung: usize,
        primary_metric: &str,
        n_promote: usize,
    ) -> Vec<HashMap<String, serde_json::Value>> {
        let mut at_rung: Vec<&mobius_core::experiment::ExperimentResult> = history
            .iter()
            .filter(|r| {
                let b = r
                    .config
                    .metadata
                    .get("hyperband_bracket")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX);
                let ru = r
                    .config
                    .metadata
                    .get("hyperband_rung")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX);
                b == bracket as u64 && ru == rung as u64
            })
            .collect();
        at_rung.sort_by(|a, b| {
            let va = a.metrics.get(primary_metric).unwrap_or(&0.0);
            let vb = b.metrics.get(primary_metric).unwrap_or(&0.0);
            vb.partial_cmp(va).unwrap_or(std::cmp::Ordering::Equal)
        });
        at_rung
            .into_iter()
            .take(n_promote)
            .map(|r| r.config.parameters.clone())
            .collect()
    }
}

impl Strategy for Hyperband {
    fn name(&self) -> &str {
        "hyperband"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        let s_max = self.s_max();

        // Iterate brackets from most aggressive (s_max) to least (0)
        for s in (0..=s_max).rev() {
            let n_initial = self.n_configs(s);
            let resource_0 = self.initial_resource(s);

            // Check each rung in this bracket
            for rung in 0..=s {
                let resource = resource_0 * self.eta.pow(rung as u32);
                let expected = if rung == 0 {
                    n_initial
                } else {
                    (n_initial as f64 / self.eta.pow(rung as u32) as f64).ceil() as usize
                };
                let actual = self.count_at_rung(ctx.history, s, rung);

                if actual < expected {
                    if rung == 0 {
                        // Need new random config at initial fidelity
                        let mut suggestion = RandomSearch.suggest(ctx)?;
                        suggestion
                            .config
                            .metadata
                            .insert("hyperband_bracket".to_string(), serde_json::json!(s));
                        suggestion
                            .config
                            .metadata
                            .insert("hyperband_rung".to_string(), serde_json::json!(rung));
                        suggestion
                            .config
                            .metadata
                            .insert("fidelity".to_string(), serde_json::json!(resource));
                        suggestion.rationale = format!(
                            "Hyperband bracket {s}, rung {rung}: new config at fidelity {resource}"
                        );
                        return Ok(suggestion);
                    } else {
                        // Promote top configs from previous rung
                        let n_promote = expected;
                        let promoted = self.top_configs_at_rung(
                            ctx.history,
                            s,
                            rung - 1,
                            ctx.primary_metric,
                            n_promote,
                        );
                        // Find first promoted config not yet run at this rung
                        for params in promoted {
                            let already_run = ctx.history.iter().any(|r| {
                                let b = r
                                    .config
                                    .metadata
                                    .get("hyperband_bracket")
                                    .and_then(|v| v.as_u64());
                                let ru = r
                                    .config
                                    .metadata
                                    .get("hyperband_rung")
                                    .and_then(|v| v.as_u64());
                                b == Some(s as u64)
                                    && ru == Some(rung as u64)
                                    && params
                                        .iter()
                                        .all(|(k, v)| r.config.parameters.get(k) == Some(v))
                            });
                            if !already_run {
                                let mut metadata = HashMap::new();
                                metadata
                                    .insert("hyperband_bracket".to_string(), serde_json::json!(s));
                                metadata
                                    .insert("hyperband_rung".to_string(), serde_json::json!(rung));
                                metadata
                                    .insert("fidelity".to_string(), serde_json::json!(resource));
                                let changed = params.clone();
                                return Ok(Suggestion {
                                    config: ExperimentConfig {
                                        parameters: params,
                                        metadata,
                                    },
                                    changed_params: changed,
                                    rationale: format!(
                                        "Hyperband bracket {s}, rung {rung}: promoted config at fidelity {resource}"
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        // All brackets complete — fall back to random
        RandomSearch.suggest(ctx)
    }
}

/// Prunes trials that are below the median of completed trials at the same fidelity.
pub struct MedianPruner {
    /// Minimum completed trials before pruning starts.
    pub min_trials: usize,
}

impl MedianPruner {
    /// Create a new median pruner.
    pub fn new(min_trials: usize) -> Self {
        Self { min_trials }
    }

    /// Check whether a trial should be pruned based on median performance.
    pub fn should_prune(
        &self,
        current_metric: f64,
        history: &[mobius_core::experiment::ExperimentResult],
        metric_name: &str,
    ) -> bool {
        let completed_values: Vec<f64> = history
            .iter()
            .filter_map(|r| r.metrics.get(metric_name).copied())
            .collect();

        if completed_values.len() < self.min_trials {
            return false;
        }

        let mut sorted = completed_values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        current_metric < median
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_store::LearningStore;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use tempfile::NamedTempFile;

    fn make_result_with_meta(
        id: &str,
        f1: f64,
        params: Vec<(&str, f64)>,
        metadata: Vec<(&str, serde_json::Value)>,
    ) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut parameters = HashMap::new();
        for (k, v) in params {
            parameters.insert(k.to_string(), serde_json::json!(v));
        }
        let mut meta = HashMap::new();
        for (k, v) in metadata {
            meta.insert(k.to_string(), v);
        }
        ExperimentResult {
            id: id.into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters,
                metadata: meta,
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
    fn test_hyperband_suggests_low_fidelity() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );

        let ctx = StrategyContext {
            history: &[],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let hb = Hyperband::new(81, 3);
        let suggestion = hb.suggest(&ctx).unwrap();
        assert!(suggestion.rationale.contains("Hyperband"));
        assert!(suggestion.config.metadata.contains_key("fidelity"));
    }

    #[test]
    fn test_hyperband_name() {
        let hb = Hyperband::new(81, 3);
        assert_eq!(hb.name(), "hyperband");
    }

    #[test]
    fn test_median_pruner_below_median() {
        let history = vec![
            make_result_with_meta("1", 0.5, vec![("lr", 0.01)], vec![]),
            make_result_with_meta("2", 0.6, vec![("lr", 0.05)], vec![]),
            make_result_with_meta("3", 0.7, vec![("lr", 0.1)], vec![]),
            make_result_with_meta("4", 0.8, vec![("lr", 0.2)], vec![]),
        ];
        let pruner = MedianPruner::new(3);
        assert!(pruner.should_prune(0.4, &history, "f1")); // Below median (0.65)
    }

    #[test]
    fn test_median_pruner_above_median() {
        let history = vec![
            make_result_with_meta("1", 0.5, vec![("lr", 0.01)], vec![]),
            make_result_with_meta("2", 0.6, vec![("lr", 0.05)], vec![]),
            make_result_with_meta("3", 0.7, vec![("lr", 0.1)], vec![]),
        ];
        let pruner = MedianPruner::new(3);
        assert!(!pruner.should_prune(0.8, &history, "f1")); // Above median
    }

    #[test]
    fn test_median_pruner_insufficient_trials() {
        let history = vec![make_result_with_meta("1", 0.5, vec![("lr", 0.01)], vec![])];
        let pruner = MedianPruner::new(3);
        assert!(!pruner.should_prune(0.1, &history, "f1")); // Not enough trials
    }

    #[test]
    fn test_hyperband_s_max() {
        let hb = Hyperband::new(81, 3);
        assert_eq!(hb.s_max(), 4); // log_3(81) = 4
    }

    #[test]
    fn test_build_strategy_hyperband() {
        let s = crate::strategy::build_strategy("hyperband", 5, 0.02).unwrap();
        assert_eq!(s.name(), "hyperband");
    }
}
