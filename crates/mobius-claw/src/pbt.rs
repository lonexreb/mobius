//! Population-Based Training (PBT) strategy.
//!
//! Maintains a population of agents, periodically replacing low performers
//! with perturbed copies of high performers. Combines exploitation (copying
//! good configs) with exploration (perturbing parameters).

use crate::strategy::{RandomSearch, Strategy, StrategyContext, Suggestion, config_already_tried};
use mobius_core::experiment::{ExperimentConfig, ExperimentResult};
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

/// Population-Based Training strategy.
///
/// On each `suggest()` call, takes the last `population_size` experiments
/// from history as the current population, ranks them by primary metric
/// (descending), replaces the bottom 25% with perturbed copies of the
/// top 25%, and returns the next candidate.
///
/// Falls back to [`RandomSearch`] when history contains fewer experiments
/// than `population_size`.
pub struct Pbt {
    /// Number of agents in the population.
    pub population_size: usize,
}

/// Perturbation factors applied to numeric parameters.
const PERTURB_FACTORS: [f64; 3] = [0.8, 1.0, 1.2];

impl Pbt {
    /// Create a new PBT strategy with the given population size.
    pub fn new(population_size: usize) -> Self {
        Self { population_size }
    }

    /// Rank experiments by primary metric (descending) and return sorted refs.
    fn rank_population<'a>(
        population: &'a [ExperimentResult],
        primary_metric: &str,
    ) -> Vec<&'a ExperimentResult> {
        let mut sorted: Vec<&ExperimentResult> = population
            .iter()
            .filter(|r| r.metrics.contains_key(primary_metric))
            .collect();
        sorted.sort_by(|a, b| {
            let va = a.metrics.get(primary_metric).unwrap_or(&0.0);
            let vb = b.metrics.get(primary_metric).unwrap_or(&0.0);
            vb.partial_cmp(va).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Snap a numeric value to the nearest value in the sweep space.
    ///
    /// Returns the sweep-space value whose numeric representation is closest
    /// to `target`. If no values are numeric, returns the first value.
    fn snap_to_sweep(target: f64, values: &[serde_json::Value]) -> serde_json::Value {
        values
            .iter()
            .filter_map(|v| v.as_f64().map(|f| (f, v)))
            .min_by(|(a, _), (b, _)| {
                let da = (a - target).abs();
                let db = (b - target).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| values.first().cloned().unwrap_or(serde_json::Value::Null))
    }

    /// Perturb a configuration by applying a random factor to each numeric
    /// parameter and snapping the result to the nearest sweep-space value.
    fn perturb(
        base: &HashMap<String, serde_json::Value>,
        sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut rng = rand::rng();
        let mut result = HashMap::new();

        for (param, values) in sweep_space {
            let base_val = base.get(param);

            // Try numeric perturbation
            if let Some(numeric) = base_val.and_then(|v| v.as_f64()) {
                let factor = PERTURB_FACTORS.choose(&mut rng).copied().unwrap_or(1.0);
                let perturbed = numeric * factor;
                result.insert(param.clone(), Self::snap_to_sweep(perturbed, values));
            } else {
                // Non-numeric: pick a random value from sweep space
                let val = values
                    .choose(&mut rng)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                result.insert(param.clone(), val);
            }
        }

        result
    }
}

impl Strategy for Pbt {
    fn name(&self) -> &str {
        "pbt"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        // Fall back to random when insufficient history
        if ctx.history.len() < self.population_size {
            return RandomSearch.suggest(ctx);
        }

        // Take the last population_size experiments as the current population
        let population = &ctx.history[ctx.history.len() - self.population_size..];
        let ranked = Self::rank_population(population, ctx.primary_metric);

        if ranked.is_empty() {
            return RandomSearch.suggest(ctx);
        }

        let n = ranked.len();
        let top_cutoff = (n as f64 * 0.25).ceil() as usize;
        let bottom_cutoff = n.saturating_sub(top_cutoff);

        let top_performers = &ranked[..top_cutoff.min(n)];
        let bottom_performers = &ranked[bottom_cutoff..];

        let mut rng = rand::rng();

        // For each bottom performer, copy a top performer's config and perturb it
        // Return the first untried candidate
        for _bottom in bottom_performers {
            if let Some(donor) = top_performers.choose(&mut rng) {
                let perturbed = Self::perturb(&donor.config.parameters, ctx.sweep_space);

                if !config_already_tried(&perturbed, ctx.history) {
                    let mut changed = HashMap::new();
                    for (k, v) in &perturbed {
                        if ctx.production_config.get(k) != Some(v) {
                            changed.insert(k.clone(), v.clone());
                        }
                    }
                    return Ok(Suggestion {
                        config: ExperimentConfig {
                            parameters: perturbed,
                            metadata: HashMap::new(),
                        },
                        changed_params: changed,
                        rationale: format!(
                            "PBT: exploit top performer + perturb (pop={}, top {}%, {} history)",
                            self.population_size,
                            25,
                            ctx.history.len()
                        ),
                    });
                }
            }
        }

        // All perturbed candidates already tried — try more perturbations
        for _ in 0..self.population_size {
            if let Some(donor) = top_performers.choose(&mut rng) {
                let perturbed = Self::perturb(&donor.config.parameters, ctx.sweep_space);
                if !config_already_tried(&perturbed, ctx.history) {
                    let mut changed = HashMap::new();
                    for (k, v) in &perturbed {
                        if ctx.production_config.get(k) != Some(v) {
                            changed.insert(k.clone(), v.clone());
                        }
                    }
                    return Ok(Suggestion {
                        config: ExperimentConfig {
                            parameters: perturbed,
                            metadata: HashMap::new(),
                        },
                        changed_params: changed,
                        rationale: format!(
                            "PBT: extra perturbation round (pop={}, {} history)",
                            self.population_size,
                            ctx.history.len()
                        ),
                    });
                }
            }
        }

        // Exhausted — fall back to random
        RandomSearch.suggest(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_store::LearningStore;
    use mobius_core::experiment::{ExperimentConfig, ExperimentStatus};
    use tempfile::NamedTempFile;

    fn make_result(id: &str, f1: f64, params: Vec<(&str, f64)>) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut parameters = HashMap::new();
        for (k, v) in params {
            parameters.insert(k.to_string(), serde_json::json!(v));
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

    fn sweep_space() -> HashMap<String, Vec<serde_json::Value>> {
        let mut space = HashMap::new();
        space.insert(
            "lr".into(),
            vec![
                serde_json::json!(0.01),
                serde_json::json!(0.05),
                serde_json::json!(0.10),
            ],
        );
        space.insert(
            "bs".into(),
            vec![serde_json::json!(16.0), serde_json::json!(32.0)],
        );
        space
    }

    #[test]
    fn test_pbt_insufficient_history() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let ctx = StrategyContext {
            history: &[make_result("1", 0.5, vec![("lr", 0.01), ("bs", 16.0)])],
            production_config: &HashMap::new(),
            sweep_space: &sweep_space(),
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let pbt = Pbt::new(4);
        let suggestion = pbt.suggest(&ctx).unwrap();
        // Falls back to RandomSearch since history (1) < population_size (4)
        assert!(
            suggestion.rationale.contains("Random"),
            "Expected random fallback, got: {}",
            suggestion.rationale
        );
    }

    #[test]
    fn test_pbt_returns_valid_config() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let space = sweep_space();

        let history = vec![
            make_result("1", 0.90, vec![("lr", 0.10), ("bs", 32.0)]),
            make_result("2", 0.70, vec![("lr", 0.05), ("bs", 16.0)]),
            make_result("3", 0.50, vec![("lr", 0.01), ("bs", 16.0)]),
            make_result("4", 0.30, vec![("lr", 0.01), ("bs", 32.0)]),
        ];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &space,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let pbt = Pbt::new(4);
        let suggestion = pbt.suggest(&ctx).unwrap();

        // Config should only contain values from sweep space
        let lr_values: Vec<f64> = space["lr"].iter().filter_map(|v| v.as_f64()).collect();
        let bs_values: Vec<f64> = space["bs"].iter().filter_map(|v| v.as_f64()).collect();

        let lr = suggestion.config.parameters["lr"].as_f64().unwrap();
        let bs = suggestion.config.parameters["bs"].as_f64().unwrap();

        assert!(
            lr_values.contains(&lr),
            "lr={lr} not in sweep space {lr_values:?}"
        );
        assert!(
            bs_values.contains(&bs),
            "bs={bs} not in sweep space {bs_values:?}"
        );
    }

    #[test]
    fn test_pbt_name() {
        let pbt = Pbt::new(8);
        assert_eq!(pbt.name(), "pbt");
    }

    #[test]
    fn test_build_strategy_pbt() {
        let s1 = crate::strategy::build_strategy("pbt", 5, 0.02).unwrap();
        assert_eq!(s1.name(), "pbt");

        let s2 = crate::strategy::build_strategy("population_based", 5, 0.02).unwrap();
        assert_eq!(s2.name(), "pbt");
    }

    #[test]
    fn test_snap_to_sweep_exact() {
        let values = vec![
            serde_json::json!(0.01),
            serde_json::json!(0.05),
            serde_json::json!(0.10),
        ];
        let snapped = Pbt::snap_to_sweep(0.05, &values);
        assert_eq!(snapped, serde_json::json!(0.05));
    }

    #[test]
    fn test_snap_to_sweep_nearest() {
        let values = vec![
            serde_json::json!(0.01),
            serde_json::json!(0.05),
            serde_json::json!(0.10),
        ];
        // 0.08 * 1.2 = 0.096, closest to 0.10
        let snapped = Pbt::snap_to_sweep(0.096, &values);
        assert_eq!(snapped, serde_json::json!(0.10));
    }

    #[test]
    fn test_perturb_stays_in_sweep_space() {
        let space = sweep_space();
        let mut base = HashMap::new();
        base.insert("lr".to_string(), serde_json::json!(0.05));
        base.insert("bs".to_string(), serde_json::json!(16.0));

        // Run many perturbations to verify all results stay in sweep space
        let lr_values: Vec<f64> = space["lr"].iter().filter_map(|v| v.as_f64()).collect();
        let bs_values: Vec<f64> = space["bs"].iter().filter_map(|v| v.as_f64()).collect();

        for _ in 0..50 {
            let perturbed = Pbt::perturb(&base, &space);
            let lr = perturbed["lr"].as_f64().unwrap();
            let bs = perturbed["bs"].as_f64().unwrap();
            assert!(
                lr_values.contains(&lr),
                "Perturbed lr={lr} not in sweep space"
            );
            assert!(
                bs_values.contains(&bs),
                "Perturbed bs={bs} not in sweep space"
            );
        }
    }
}
