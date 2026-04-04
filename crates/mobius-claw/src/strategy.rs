use crate::learning::GradientDirection;
use crate::learning_store::LearningStore;
use mobius_core::experiment::{ExperimentConfig, ExperimentResult};
use rand::prelude::IndexedRandom;
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
        history.iter().any(|r| {
            config
                .iter()
                .all(|(k, v)| r.config.parameters.get(k) == Some(v))
        })
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

        let recent: Vec<&ExperimentResult> = ctx
            .history
            .iter()
            .rev()
            .take(self.plateau_window)
            .collect();

        let untried = ctx.learning_store.get_untried_dimensions(
            ctx.history,
            ctx.sweep_space,
        );

        // Case 2: Plateau — explore untried or expand range
        if self.is_plateaued(&recent.iter().copied().cloned().collect::<Vec<_>>(), ctx.primary_metric) {
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
                    rationale: format!(
                        "Plateau detected. Exploring untried dimension: {}",
                        param
                    ),
                });
            }
            // Fallback: expand range of most impactful param
            return self.suggest_expand_range(ctx, &base_config);
        }

        // Case 3: Gradient-guided suggestion
        let mut gradients = Vec::new();
        for param in ctx.sweep_space.keys() {
            let grad = ctx.learning_store.get_param_gradient(param, ctx.primary_metric)?;
            if grad.num_observations > 0 {
                gradients.push(grad);
            }
        }

        // 50% chance: explore untried dimension instead
        if !untried.is_empty() && rand::random::<bool>()
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
                GradientDirection::DecreaseHelps => {
                    values
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| (f, v)))
                        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(_, v)| v.clone())
                        .or_else(|| values.first().cloned())
                }
                GradientDirection::Inconclusive => values.choose(&mut rng).cloned(),
            };

            if let Some(val) = val {
                let mut config_params = base_config.clone();
                config_params.insert(top.param.clone(), val.clone());

                // Dedup check
                if self.config_already_tried(&config_params, ctx.history)
                    && let Some(alt) = values
                        .iter()
                        .find(|v| *v != &val)
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
        sweep.insert("ball_conf".into(), vec![serde_json::json!(0.25), serde_json::json!(0.30)]);

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
        sweep.insert("ball_conf".into(), vec![serde_json::json!(0.20), serde_json::json!(0.25), serde_json::json!(0.30)]);
        sweep.insert("dedup_window".into(), vec![serde_json::json!(4.0), serde_json::json!(6.0)]);

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
}
