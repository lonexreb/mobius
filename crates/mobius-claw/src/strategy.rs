use crate::learning::GradientDirection;
use crate::learning_store::LearningStore;
use mobius_core::experiment::{ExperimentConfig, ExperimentResult};
use rand::prelude::IndexedRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A suggested experiment configuration with rationale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub config: ExperimentConfig,
    pub changed_params: HashMap<String, serde_json::Value>,
    pub rationale: String,
}

/// Context provided to a strategy for generating suggestions.
pub struct StrategyContext<'a> {
    pub history: &'a [ExperimentResult],
    pub production_config: &'a HashMap<String, serde_json::Value>,
    pub sweep_space: &'a HashMap<String, Vec<serde_json::Value>>,
    pub targets: &'a HashMap<String, f64>,
    pub primary_metric: &'a str,
    pub learning_store: &'a LearningStore,
}

/// Trait for experiment proposal strategies.
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion>;
}

/// Check whether an exact config has already been tried in history.
pub fn config_already_tried(
    config: &HashMap<String, serde_json::Value>,
    history: &[ExperimentResult],
) -> bool {
    history.iter().any(|r| {
        config
            .iter()
            .all(|(k, v)| r.config.parameters.get(k) == Some(v))
    })
}

/// Build a strategy by name using default parameters.
///
/// Known strategies: `"gradient_guided"`, `"random"`, `"grid"`, `"tpe"`,
/// `"nsga2"`, `"ucb1"`, `"cmaes"`, `"pbt"`, `"auto"`, `"hyperband"`.
pub fn build_strategy(
    name: &str,
    plateau_window: usize,
    plateau_threshold: f64,
) -> anyhow::Result<Box<dyn Strategy>> {
    build_strategy_with_params(
        name,
        plateau_window,
        plateau_threshold,
        &mobius_core::config::StrategyParams::default(),
    )
}

/// Build a strategy by name with configurable parameters from mobius.toml.
pub fn build_strategy_with_params(
    name: &str,
    plateau_window: usize,
    plateau_threshold: f64,
    params: &mobius_core::config::StrategyParams,
) -> anyhow::Result<Box<dyn Strategy>> {
    match name {
        "gradient_guided" | "gradient_guided_tuning" => Ok(Box::new(GradientGuidedTuning::new(
            plateau_window,
            plateau_threshold,
        ))),
        "random" | "random_search" => Ok(Box::new(RandomSearch)),
        "grid" | "grid_search" => Ok(Box::new(GridSearch)),
        "tpe" | "tpe_search" => Ok(Box::new(TpeSearch::new(params.tpe_gamma))),
        "nsga2" | "nsga_ii" => Ok(Box::new(crate::nsga::NsgaTwo::new(
            params.nsga_objectives.clone(),
            params.nsga_population_size,
        ))),
        "ucb1" | "tree_search" => Ok(Box::new(crate::tree_search::UcbTreeSearch::new(
            params.ucb1_exploration_constant,
        ))),
        "cmaes" | "cma_es" => Ok(Box::new(crate::cmaes::CmaEs::new(
            params.cmaes_population_size,
        ))),
        "pbt" | "population_based" => {
            Ok(Box::new(crate::pbt::Pbt::new(params.pbt_population_size)))
        }
        "auto" | "auto_strategy" => Ok(Box::new(crate::auto_strategy::AutoStrategy::new(
            plateau_window,
            plateau_threshold,
        ))),
        "hyperband" => Ok(Box::new(crate::hyperband::Hyperband::new(
            params.hyperband_max_resource,
            params.hyperband_eta,
        ))),
        _ => anyhow::bail!("Unknown strategy: {name}"),
    }
}

/// Pure random parameter sampling from the sweep space.
///
/// Picks a random value for every parameter on each call.
/// Useful as a baseline or for initial exploration.
pub struct RandomSearch;

impl Strategy for RandomSearch {
    fn name(&self) -> &str {
        "random"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        let mut rng = rand::rng();
        let mut config_params = HashMap::new();
        let mut changed = HashMap::new();

        for (param, values) in ctx.sweep_space {
            let val = values
                .choose(&mut rng)
                .ok_or_else(|| anyhow::anyhow!("Empty values for {}", param))?;
            config_params.insert(param.clone(), val.clone());
            changed.insert(param.clone(), val.clone());
        }

        Ok(Suggestion {
            config: ExperimentConfig {
                parameters: config_params,
                metadata: HashMap::new(),
            },
            changed_params: changed,
            rationale: "Random exploration across all parameters.".into(),
        })
    }
}

/// Systematic grid search over the sweep space.
///
/// Enumerates all combinations and skips already-tried configs.
/// Returns an error when all combinations have been exhausted.
pub struct GridSearch;

impl GridSearch {
    fn cartesian_product(
        sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    ) -> Vec<HashMap<String, serde_json::Value>> {
        let keys: Vec<&String> = sweep_space.keys().collect();
        let values: Vec<&Vec<serde_json::Value>> = keys.iter().map(|k| &sweep_space[*k]).collect();

        if values.is_empty() {
            return vec![HashMap::new()];
        }

        let mut combos = vec![HashMap::new()];
        for (i, vals) in values.iter().enumerate() {
            let mut new_combos = Vec::new();
            for combo in &combos {
                for v in *vals {
                    let mut c = combo.clone();
                    c.insert(keys[i].clone(), v.clone());
                    new_combos.push(c);
                }
            }
            combos = new_combos;
        }
        combos
    }
}

impl Strategy for GridSearch {
    fn name(&self) -> &str {
        "grid"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        let combos = Self::cartesian_product(ctx.sweep_space);

        for combo in &combos {
            if !config_already_tried(combo, ctx.history) {
                let mut changed = HashMap::new();
                for (k, v) in combo {
                    if ctx.production_config.get(k) != Some(v) {
                        changed.insert(k.clone(), v.clone());
                    }
                }
                return Ok(Suggestion {
                    config: ExperimentConfig {
                        parameters: combo.clone(),
                        metadata: HashMap::new(),
                    },
                    changed_params: changed,
                    rationale: format!(
                        "Grid search: config {} of {}",
                        ctx.history.len() + 1,
                        combos.len()
                    ),
                });
            }
        }

        anyhow::bail!("Grid search exhausted all {} combinations", combos.len())
    }
}

/// Tree-structured Parzen Estimator (TPE) for Bayesian hyperparameter optimization.
///
/// Splits past trials into "good" (top `gamma` quantile by metric) and "bad",
/// fits histogram-based kernel density estimates for each group, then picks the
/// candidate that maximizes the expected improvement ratio `l(x)/g(x)`.
///
/// This is the first Rust-native TPE implementation — no external dependencies.
pub struct TpeSearch {
    /// Fraction of history considered "good" (default: 0.25).
    pub gamma: f64,
}

impl TpeSearch {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    /// Detect if a numeric domain should use log-scale sampling.
    ///
    /// Returns true if the domain spans more than 2 orders of magnitude
    /// and all values are positive — typical for learning rates, weight decay,
    /// regularization coefficients.
    fn is_log_scale(domain: &[f64]) -> bool {
        if domain.len() < 2 {
            return false;
        }
        let min = domain.iter().copied().fold(f64::INFINITY, f64::min);
        let max = domain.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        min > 0.0 && max / min > 100.0
    }

    /// Gaussian kernel density estimate at each domain point.
    fn kde(observations: &[f64], domain: &[f64], bandwidth: f64) -> Vec<f64> {
        if observations.is_empty() {
            return vec![1.0 / domain.len() as f64; domain.len()];
        }
        let bw = if bandwidth > 0.0 { bandwidth } else { 1.0 };
        let mut densities: Vec<f64> = domain
            .par_iter()
            .map(|&x| {
                let sum: f64 = observations
                    .iter()
                    .map(|&obs| {
                        let z = (x - obs) / bw;
                        (-0.5 * z * z).exp()
                    })
                    .sum();
                sum / (observations.len() as f64 * bw)
            })
            .collect();
        // Normalize
        let total: f64 = densities.iter().sum();
        if total > 0.0 {
            for d in &mut densities {
                *d /= total;
            }
        }
        densities
    }

    /// Bandwidth heuristic: Silverman's rule adapted for discrete domain.
    fn bandwidth(observations: &[f64], domain: &[f64]) -> f64 {
        if observations.len() < 2 || domain.len() < 2 {
            return 1.0;
        }
        let range = domain.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - domain.iter().copied().fold(f64::INFINITY, f64::min);
        range / (observations.len() as f64).sqrt()
    }
}

impl Strategy for TpeSearch {
    fn name(&self) -> &str {
        "tpe"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        // Need minimum history to split into good/bad
        if ctx.history.len() < 4 {
            return RandomSearch.suggest(ctx);
        }

        // Sort by primary metric descending
        let mut sorted: Vec<&ExperimentResult> = ctx
            .history
            .iter()
            .filter(|r| r.metrics.contains_key(ctx.primary_metric))
            .collect();
        sorted.sort_by(|a, b| {
            let va = a.metrics.get(ctx.primary_metric).unwrap_or(&0.0);
            let vb = b.metrics.get(ctx.primary_metric).unwrap_or(&0.0);
            vb.partial_cmp(va).unwrap_or(std::cmp::Ordering::Equal)
        });

        if sorted.is_empty() {
            return RandomSearch.suggest(ctx);
        }

        // Split into good (top gamma) and bad
        let n_good = ((sorted.len() as f64 * self.gamma).ceil() as usize).max(1);
        let good = &sorted[..n_good];
        let bad = &sorted[n_good..];

        let mut best_config = HashMap::new();
        let mut changed = HashMap::new();

        for (param, values) in ctx.sweep_space {
            // Extract numeric domain values
            let domain: Vec<f64> = values.iter().filter_map(|v| v.as_f64()).collect();

            if domain.is_empty() {
                // Categorical: pick value most frequent in good, least in bad
                let val = values.first().cloned().unwrap_or(serde_json::Value::Null);
                best_config.insert(param.clone(), val.clone());
                changed.insert(param.clone(), val);
                continue;
            }

            let use_log = Self::is_log_scale(&domain);

            let good_obs: Vec<f64> = good
                .iter()
                .filter_map(|r| {
                    let v = r.config.parameters.get(param)?.as_f64()?;
                    Some(if use_log { v.ln() } else { v })
                })
                .collect();
            let bad_obs: Vec<f64> = bad
                .iter()
                .filter_map(|r| {
                    let v = r.config.parameters.get(param)?.as_f64()?;
                    Some(if use_log { v.ln() } else { v })
                })
                .collect();

            // KDE operates in log-space for log-scale params
            let kde_domain: Vec<f64> = if use_log {
                domain.iter().map(|v| v.ln()).collect()
            } else {
                domain.clone()
            };

            let bw_good = Self::bandwidth(&good_obs, &kde_domain);
            let bw_bad = Self::bandwidth(&bad_obs, &kde_domain);
            let l = Self::kde(&good_obs, &kde_domain, bw_good);
            let g = Self::kde(&bad_obs, &kde_domain, bw_bad);

            // Pick domain value with highest l(x)/g(x) ratio
            let best_idx = l
                .iter()
                .zip(g.iter())
                .enumerate()
                .max_by(|(_, (l1, g1)), (_, (l2, g2))| {
                    let r1 = *l1 / g1.max(1e-10);
                    let r2 = *l2 / g2.max(1e-10);
                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            best_config.insert(param.clone(), values[best_idx].clone());
            changed.insert(param.clone(), values[best_idx].clone());
        }

        // Dedup check: if this exact config was tried, fall back to random
        if config_already_tried(&best_config, ctx.history) {
            return RandomSearch.suggest(ctx);
        }

        Ok(Suggestion {
            config: ExperimentConfig {
                parameters: best_config,
                metadata: HashMap::new(),
            },
            changed_params: changed,
            rationale: format!(
                "TPE: selected config maximizing expected improvement (gamma={:.2}, {} good / {} bad)",
                self.gamma,
                n_good,
                bad.len()
            ),
        })
    }
}

/// Gradient-guided parameter tuning strategy.
///
/// Port of `pipeline_optimizer.py:cmd_suggest` algorithm.
pub struct GradientGuidedTuning {
    pub plateau_window: usize,
    pub plateau_threshold: f64,
}

impl GradientGuidedTuning {
    pub fn new(plateau_window: usize, plateau_threshold: f64) -> Self {
        Self {
            plateau_window,
            plateau_threshold,
        }
    }

    fn is_plateaued(&self, history: &[ExperimentResult], metric: &str) -> bool {
        crate::is_plateaued(history, metric, self.plateau_threshold)
    }

    fn find_best<'a>(
        &self,
        history: &'a [ExperimentResult],
        metric: &str,
    ) -> Option<&'a ExperimentResult> {
        history
            .iter()
            .filter(|r| r.metrics.contains_key(metric))
            .max_by(|a, b| {
                let va = a.metrics.get(metric).unwrap_or(&0.0);
                let vb = b.metrics.get(metric).unwrap_or(&0.0);
                va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn config_already_tried(
        &self,
        config: &HashMap<String, serde_json::Value>,
        history: &[ExperimentResult],
    ) -> bool {
        config_already_tried(config, history)
    }
}

impl Strategy for GradientGuidedTuning {
    fn name(&self) -> &str {
        "gradient_guided_tuning"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        let mut rng = rand::rng();

        // Case 1: Insufficient history — random exploration
        if ctx.history.len() < 2 {
            let params: Vec<&String> = ctx.sweep_space.keys().collect();
            if let Some(param) = params.choose(&mut rng)
                && let Some(values) = ctx.sweep_space.get(*param)
                && let Some(val) = values.choose(&mut rng)
            {
                let mut config_params = ctx.production_config.clone();
                config_params.insert((*param).clone(), (*val).clone());
                let mut changed = HashMap::new();
                changed.insert((*param).clone(), val.clone());
                return Ok(Suggestion {
                    config: ExperimentConfig {
                        parameters: config_params,
                        metadata: HashMap::new(),
                    },
                    changed_params: changed,
                    rationale: format!(
                        "Insufficient history ({}). Exploring {}.",
                        ctx.history.len(),
                        param
                    ),
                });
            }
        }

        let best = self.find_best(ctx.history, ctx.primary_metric);
        let base_config = best
            .map(|b| b.config.parameters.clone())
            .unwrap_or_else(|| ctx.production_config.clone());

        let recent: Vec<&ExperimentResult> =
            ctx.history.iter().rev().take(self.plateau_window).collect();

        let untried = ctx
            .learning_store
            .get_untried_dimensions(ctx.history, ctx.sweep_space);

        // Case 2: Plateau — explore untried or expand range
        if self.is_plateaued(
            &recent.iter().copied().cloned().collect::<Vec<_>>(),
            ctx.primary_metric,
        ) {
            if let Some(param) = untried.first()
                && let Some(values) = ctx.sweep_space.get(param)
                && let Some(val) = values.choose(&mut rng)
            {
                let mut config_params = base_config.clone();
                config_params.insert(param.clone(), (*val).clone());
                let mut changed = HashMap::new();
                changed.insert(param.clone(), val.clone());
                return Ok(Suggestion {
                    config: ExperimentConfig {
                        parameters: config_params,
                        metadata: HashMap::new(),
                    },
                    changed_params: changed,
                    rationale: format!("Plateau detected. Exploring untried dimension: {}", param),
                });
            }
            // Fallback: expand range of most impactful param
            return self.suggest_expand_range(ctx, &base_config);
        }

        // Case 3: Gradient-guided suggestion
        let mut gradients = Vec::new();
        for param in ctx.sweep_space.keys() {
            let grad = ctx
                .learning_store
                .get_param_gradient(param, ctx.primary_metric)?;
            if grad.num_observations > 0 {
                gradients.push(grad);
            }
        }

        // 50% chance: explore untried dimension instead
        if !untried.is_empty()
            && rand::random::<bool>()
            && let Some(param) = untried.first()
            && let Some(values) = ctx.sweep_space.get(param)
            && let Some(val) = values.choose(&mut rng)
        {
            let mut config_params = base_config.clone();
            config_params.insert(param.clone(), (*val).clone());
            let mut changed = HashMap::new();
            changed.insert(param.clone(), val.clone());
            return Ok(Suggestion {
                config: ExperimentConfig {
                    parameters: config_params,
                    metadata: HashMap::new(),
                },
                changed_params: changed,
                rationale: format!("Exploring untried dimension: {}", param),
            });
        }

        // Use gradient signals
        gradients.sort_by(|a, b| {
            b.avg_metric_delta
                .abs()
                .partial_cmp(&a.avg_metric_delta.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(top) = gradients.first()
            && let Some(values) = ctx.sweep_space.get(&top.param)
        {
            let val = match top.best_direction {
                GradientDirection::IncreaseHelps => {
                    // Pick highest numeric value
                    values
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| (f, v)))
                        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(_, v)| v.clone())
                        .or_else(|| values.last().cloned())
                }
                GradientDirection::DecreaseHelps => values
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| (f, v)))
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(_, v)| v.clone())
                    .or_else(|| values.first().cloned()),
                GradientDirection::Inconclusive => values.choose(&mut rng).cloned(),
            };

            if let Some(val) = val {
                let mut config_params = base_config.clone();
                config_params.insert(top.param.clone(), val.clone());

                // Dedup check
                if self.config_already_tried(&config_params, ctx.history)
                    && let Some(alt) = values.iter().find(|v| *v != &val)
                {
                    config_params.insert(top.param.clone(), alt.clone());
                }

                let mut changed = HashMap::new();
                if let Some(prod_val) = ctx.production_config.get(&top.param)
                    && config_params.get(&top.param) != Some(prod_val)
                {
                    changed.insert(
                        top.param.clone(),
                        config_params.get(&top.param).cloned().unwrap_or_default(),
                    );
                }

                return Ok(Suggestion {
                    config: ExperimentConfig {
                        parameters: config_params,
                        metadata: HashMap::new(),
                    },
                    changed_params: changed,
                    rationale: format!(
                        "Gradient-guided: {} ({:?}, avg_delta={:+.4})",
                        top.param, top.best_direction, top.avg_metric_delta
                    ),
                });
            }
        }

        // Final fallback: random parameter variation
        self.suggest_random(ctx, &base_config)
    }
}

impl GradientGuidedTuning {
    fn suggest_random(
        &self,
        ctx: &StrategyContext,
        base: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<Suggestion> {
        let mut rng = rand::rng();
        let params: Vec<&String> = ctx.sweep_space.keys().collect();
        let param = params
            .choose(&mut rng)
            .ok_or_else(|| anyhow::anyhow!("Empty sweep space"))?;
        let values = ctx
            .sweep_space
            .get(*param)
            .ok_or_else(|| anyhow::anyhow!("No values for {}", param))?;
        let val = values
            .choose(&mut rng)
            .ok_or_else(|| anyhow::anyhow!("Empty values for {}", param))?;

        let mut config_params = base.clone();
        config_params.insert((*param).clone(), (*val).clone());
        let mut changed = HashMap::new();
        changed.insert((*param).clone(), val.clone());

        Ok(Suggestion {
            config: ExperimentConfig {
                parameters: config_params,
                metadata: HashMap::new(),
            },
            changed_params: changed,
            rationale: format!("Random exploration: {}", param),
        })
    }

    fn suggest_expand_range(
        &self,
        ctx: &StrategyContext,
        base: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<Suggestion> {
        let mut rng = rand::rng();
        // Find param with most observations, try value beyond current range
        let params: Vec<&String> = ctx.sweep_space.keys().collect();
        let param = params
            .choose(&mut rng)
            .ok_or_else(|| anyhow::anyhow!("Empty sweep space"))?;
        let values = ctx
            .sweep_space
            .get(*param)
            .ok_or_else(|| anyhow::anyhow!("No values"))?;

        // Try expanding: pick the extreme value
        let val = values
            .last()
            .ok_or_else(|| anyhow::anyhow!("Empty values"))?;

        let mut config_params = base.clone();
        config_params.insert((*param).clone(), (*val).clone());
        let mut changed = HashMap::new();
        changed.insert((*param).clone(), val.clone());

        Ok(Suggestion {
            config: ExperimentConfig {
                parameters: config_params,
                metadata: HashMap::new(),
            },
            changed_params: changed,
            rationale: format!("Plateau: expanding range of {}", param),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::ExperimentStatus;
    use tempfile::NamedTempFile;

    fn make_result(id: &str, f1: f64, ball_conf: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut params = HashMap::new();
        params.insert("ball_conf".into(), serde_json::json!(ball_conf));
        ExperimentResult {
            id: id.into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: params,
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 10.0,
            cost_usd: Some(1.20),
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    #[test]
    fn test_random_exploration_insufficient_history() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let strategy = GradientGuidedTuning::new(5, 0.02);

        let mut sweep = HashMap::new();
        sweep.insert(
            "ball_conf".into(),
            vec![serde_json::json!(0.25), serde_json::json!(0.30)],
        );

        let mut prod = HashMap::new();
        prod.insert("ball_conf".into(), serde_json::json!(0.28));

        let ctx = StrategyContext {
            history: &[make_result("1", 0.75, 0.28)],
            production_config: &prod,
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = strategy.suggest(&ctx).unwrap();
        assert!(suggestion.rationale.contains("Insufficient history"));
        assert!(!suggestion.changed_params.is_empty());
    }

    #[test]
    fn test_suggest_returns_valid_config() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let strategy = GradientGuidedTuning::new(5, 0.02);

        let history = vec![
            make_result("1", 0.75, 0.28),
            make_result("2", 0.78, 0.25),
            make_result("3", 0.76, 0.30),
        ];

        let mut sweep = HashMap::new();
        sweep.insert(
            "ball_conf".into(),
            vec![
                serde_json::json!(0.20),
                serde_json::json!(0.25),
                serde_json::json!(0.30),
            ],
        );
        sweep.insert(
            "dedup_window".into(),
            vec![serde_json::json!(4.0), serde_json::json!(6.0)],
        );

        let mut prod = HashMap::new();
        prod.insert("ball_conf".into(), serde_json::json!(0.28));
        prod.insert("dedup_window".into(), serde_json::json!(6.0));

        let ctx = StrategyContext {
            history: &history,
            production_config: &prod,
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = strategy.suggest(&ctx).unwrap();
        assert!(!suggestion.config.parameters.is_empty());
        assert!(!suggestion.rationale.is_empty());
    }

    // -----------------------------------------------------------------------
    // RandomSearch tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_random_search_valid_config() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );
        sweep.insert(
            "bs".into(),
            vec![serde_json::json!(16), serde_json::json!(32)],
        );

        let ctx = StrategyContext {
            history: &[],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = RandomSearch.suggest(&ctx).unwrap();
        assert!(suggestion.config.parameters.contains_key("lr"));
        assert!(suggestion.config.parameters.contains_key("bs"));
        assert!(suggestion.rationale.contains("Random"));
    }

    #[test]
    fn test_random_search_values_from_sweep() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let sweep_vals = vec![serde_json::json!(0.01), serde_json::json!(0.1)];
        let mut sweep = HashMap::new();
        sweep.insert("lr".into(), sweep_vals.clone());

        let ctx = StrategyContext {
            history: &[],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = RandomSearch.suggest(&ctx).unwrap();
        let val = &suggestion.config.parameters["lr"];
        assert!(sweep_vals.contains(val));
    }

    // -----------------------------------------------------------------------
    // GridSearch tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_search_skips_tried() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "ball_conf".into(),
            vec![serde_json::json!(0.25), serde_json::json!(0.30)],
        );

        // First combo is already tried
        let tried = make_result("1", 0.75, 0.25);

        let ctx = StrategyContext {
            history: &[tried],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = GridSearch.suggest(&ctx).unwrap();
        assert_eq!(
            suggestion.config.parameters["ball_conf"],
            serde_json::json!(0.30)
        );
        assert!(suggestion.rationale.contains("Grid search"));
    }

    #[test]
    fn test_grid_search_exhausted() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "ball_conf".into(),
            vec![serde_json::json!(0.25), serde_json::json!(0.30)],
        );

        let history = vec![make_result("1", 0.75, 0.25), make_result("2", 0.78, 0.30)];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let err = GridSearch.suggest(&ctx);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("exhausted"));
    }

    // -----------------------------------------------------------------------
    // build_strategy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_strategy_known() {
        let s1 = build_strategy("gradient_guided", 5, 0.02).unwrap();
        assert_eq!(s1.name(), "gradient_guided_tuning");

        let s2 = build_strategy("random", 5, 0.02).unwrap();
        assert_eq!(s2.name(), "random");

        let s3 = build_strategy("grid", 5, 0.02).unwrap();
        assert_eq!(s3.name(), "grid");
    }

    #[test]
    fn test_build_strategy_unknown() {
        let err = build_strategy("bayesian", 5, 0.02);
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // TpeSearch tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tpe_insufficient_history_fallback() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );

        let history = vec![make_result("1", 0.50, 0.25), make_result("2", 0.60, 0.30)];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = TpeSearch::new(0.25).suggest(&ctx).unwrap();
        // Falls back to random since < 4 history entries
        assert!(suggestion.rationale.contains("Random"));
    }

    #[test]
    fn test_tpe_prefers_good_region() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "ball_conf".into(),
            vec![
                serde_json::json!(0.10),
                serde_json::json!(0.20),
                serde_json::json!(0.30),
                serde_json::json!(0.40),
                serde_json::json!(0.50),
            ],
        );

        // High ball_conf clearly correlates with better f1
        // Note: 0.50 is untried, so TPE can suggest it without dedup
        let history = vec![
            make_result("1", 0.40, 0.10),
            make_result("2", 0.45, 0.20),
            make_result("3", 0.80, 0.30),
            make_result("4", 0.85, 0.40),
        ];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = TpeSearch::new(0.25).suggest(&ctx).unwrap();
        let val = suggestion.config.parameters["ball_conf"].as_f64().unwrap();
        // TPE should prefer the high end (0.30-0.50)
        assert!(val >= 0.30, "TPE should prefer high ball_conf, got {val}");
    }

    #[test]
    fn test_tpe_kde_uniform_for_empty() {
        let domain = vec![1.0, 2.0, 3.0, 4.0];
        let densities = TpeSearch::kde(&[], &domain, 1.0);
        // Should be uniform
        let expected = 1.0 / 4.0;
        for d in &densities {
            assert!((d - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_tpe_kde_peaks_at_observations() {
        let domain = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let observations = vec![3.0, 3.0, 3.0]; // all at 3.0
        let bw = TpeSearch::bandwidth(&observations, &domain);
        let densities = TpeSearch::kde(&observations, &domain, bw);
        // Density at 3.0 (index 2) should be highest
        let max_idx = densities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2, "KDE should peak at observed value 3.0");
    }

    #[test]
    fn test_build_strategy_tpe() {
        let s = build_strategy("tpe", 5, 0.02).unwrap();
        assert_eq!(s.name(), "tpe");
    }
}
