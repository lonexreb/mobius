//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
//!
//! Gold standard for continuous hyperparameter optimization. Maintains a
//! multivariate normal distribution over the parameter space and adapts
//! mean, covariance, and step size based on experiment rankings.
//!
//! State is reconstructed from history on each `suggest()` call — stateless
//! like all Mobius strategies.

use crate::strategy::{RandomSearch, Strategy, StrategyContext, Suggestion, config_already_tried};
use mobius_core::experiment::ExperimentConfig;
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

/// CMA-ES strategy for continuous parameter optimization.
///
/// Rebuilds mean vector, covariance matrix, and step size from experiment
/// history. Samples new candidates from N(mean, sigma^2 * C) using
/// Cholesky decomposition. Falls back to random search for categorical
/// parameters or insufficient history.
pub struct CmaEs {
    /// Population size per generation (None = auto from dimensionality).
    pub population_size: Option<usize>,
}

impl CmaEs {
    /// Create a new CMA-ES strategy. Pass `None` for auto population size.
    pub fn new(population_size: Option<usize>) -> Self {
        Self { population_size }
    }
}

/// Extract numeric parameters and their bounds from sweep space.
fn extract_numeric_params(
    sweep_space: &HashMap<String, Vec<serde_json::Value>>,
) -> Vec<(String, Vec<f64>)> {
    let mut params: Vec<(String, Vec<f64>)> = sweep_space
        .iter()
        .filter_map(|(name, values)| {
            let nums: Vec<f64> = values.iter().filter_map(|v| v.as_f64()).collect();
            if nums.len() == values.len() && !nums.is_empty() {
                Some((name.clone(), nums))
            } else {
                None
            }
        })
        .collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    params
}

/// Snap a continuous value to the nearest value in the sweep space.
fn snap_to_nearest(val: f64, allowed: &[f64]) -> f64 {
    allowed
        .iter()
        .copied()
        .min_by(|a, b| {
            (a - val)
                .abs()
                .partial_cmp(&(b - val).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(val)
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns lower triangular L such that A = L * L^T.
#[allow(clippy::needless_range_loop)]
fn cholesky(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = mat[i][i] - sum;
                if diag <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = (mat[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Sample from N(0, I) using Box-Muller transform.
fn sample_normal(n: usize) -> Vec<f64> {
    use rand::RngExt;
    let mut rng = rand::rng();
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let z: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        samples.push(z);
    }
    samples
}

impl Strategy for CmaEs {
    fn name(&self) -> &str {
        "cmaes"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        let numeric_params = extract_numeric_params(ctx.sweep_space);
        let d = numeric_params.len();

        if d == 0 {
            return RandomSearch.suggest(ctx);
        }

        let lambda = self
            .population_size
            .unwrap_or_else(|| 4 + (3.0 * (d as f64).ln()).floor() as usize);

        if ctx.history.len() < lambda {
            return RandomSearch.suggest(ctx);
        }

        // Detect log-scale parameters and compute bounds
        let log_scale: Vec<bool> = numeric_params
            .iter()
            .map(|(_, vals)| {
                let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
                let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                min > 0.0 && max / min > 100.0
            })
            .collect();

        let bounds: Vec<(f64, f64)> = numeric_params
            .iter()
            .zip(log_scale.iter())
            .map(|((_, vals), &is_log)| {
                let transformed: Vec<f64> = vals
                    .iter()
                    .map(|&v| if is_log { v.ln() } else { v })
                    .collect();
                let min = transformed.iter().copied().fold(f64::INFINITY, f64::min);
                let max = transformed
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                (min, max)
            })
            .collect();

        // Extract parameter vectors from history, normalized to [0, 1] (log-space for log-scale)
        let history_vecs: Vec<(Vec<f64>, f64)> = ctx
            .history
            .iter()
            .filter_map(|r| {
                let metric = r.metrics.get(ctx.primary_metric)?;
                let vec: Vec<f64> = numeric_params
                    .iter()
                    .zip(bounds.iter())
                    .zip(log_scale.iter())
                    .map(|(((name, _), (lo, hi)), &is_log)| {
                        let raw = r.config.parameters.get(name)?.as_f64()?;
                        let val = if is_log { raw.ln() } else { raw };
                        if (hi - lo).abs() < 1e-10 {
                            Some(0.5)
                        } else {
                            Some((val - lo) / (hi - lo))
                        }
                    })
                    .collect::<Option<Vec<f64>>>()?;
                Some((vec, *metric))
            })
            .collect();

        if history_vecs.len() < lambda {
            return RandomSearch.suggest(ctx);
        }

        // Take the latest generation and compute weighted mean of top mu
        let mu = lambda / 2;
        let gen_start = history_vecs.len().saturating_sub(lambda);
        let mut generation: Vec<(Vec<f64>, f64)> = history_vecs[gen_start..].to_vec();
        generation.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute CMA-ES weights for top mu
        let weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let w_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();

        // Weighted mean of top mu solutions
        let mut mean = vec![0.0; d];
        for (i, (vec, _)) in generation.iter().take(mu).enumerate() {
            for j in 0..d {
                mean[j] += weights[i] * vec[j];
            }
        }

        // Compute covariance from top solutions
        let mut cov = vec![vec![0.0; d]; d];
        for (i, (vec, _)) in generation.iter().take(mu).enumerate() {
            for j in 0..d {
                for k in 0..d {
                    cov[j][k] += weights[i] * (vec[j] - mean[j]) * (vec[k] - mean[k]);
                }
            }
        }

        // Add regularization to diagonal for numerical stability
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] += 1e-6;
        }

        // Step size: adaptive based on generation spread
        let sigma = 0.3_f64.max(
            generation
                .iter()
                .take(mu)
                .map(|(vec, _)| {
                    vec.iter()
                        .zip(mean.iter())
                        .map(|(v, m)| (v - m).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>()
                / mu as f64,
        );

        // Sample from N(mean, sigma^2 * C)
        let l = cholesky(&cov).unwrap_or_else(|| {
            // Fallback: identity covariance (diagonal)
            let mut identity = vec![vec![0.0; d]; d];
            for (i, row) in identity.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            identity
        });

        // Try multiple samples to find an untried config
        for _ in 0..lambda * 2 {
            let z = sample_normal(d);
            let mut candidate_norm: Vec<f64> = mean.clone();
            for i in 0..d {
                for j in 0..=i {
                    candidate_norm[i] += sigma * l[i][j] * z[j];
                }
                // Clamp to [0, 1]
                candidate_norm[i] = candidate_norm[i].clamp(0.0, 1.0);
            }

            // Denormalize (reverse log-transform if needed) and snap to sweep space
            let mut config_params = HashMap::new();
            let mut changed = HashMap::new();
            for (idx, (name, allowed)) in numeric_params.iter().enumerate() {
                let (lo, hi) = bounds[idx];
                let transformed = lo + candidate_norm[idx] * (hi - lo);
                let raw = if log_scale[idx] {
                    transformed.exp()
                } else {
                    transformed
                };
                let snapped = snap_to_nearest(raw, allowed);
                config_params.insert(name.clone(), serde_json::json!(snapped));
                if ctx.production_config.get(name) != Some(&serde_json::json!(snapped)) {
                    changed.insert(name.clone(), serde_json::json!(snapped));
                }
            }

            // Add categorical params randomly
            let mut rng = rand::rng();
            for (name, values) in ctx.sweep_space {
                if !config_params.contains_key(name)
                    && let Some(val) = values.choose(&mut rng)
                {
                    config_params.insert(name.clone(), val.clone());
                    changed.insert(name.clone(), val.clone());
                }
            }

            if !config_already_tried(&config_params, ctx.history) {
                return Ok(Suggestion {
                    config: ExperimentConfig {
                        parameters: config_params,
                        metadata: HashMap::new(),
                    },
                    changed_params: changed,
                    rationale: format!(
                        "CMA-ES: d={}, lambda={}, sigma={:.3}, {} history",
                        d,
                        lambda,
                        sigma,
                        ctx.history.len()
                    ),
                });
            }
        }

        // All samples tried — fall back to random
        RandomSearch.suggest(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_store::LearningStore;
    use crate::strategy::StrategyContext;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
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

    #[test]
    fn test_cmaes_insufficient_history() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );

        let ctx = StrategyContext {
            history: &[make_result("1", 0.5, vec![("lr", 0.01)])],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let cmaes = CmaEs::new(None);
        let suggestion = cmaes.suggest(&ctx).unwrap();
        assert!(suggestion.rationale.contains("Random"));
    }

    #[test]
    fn test_cmaes_returns_valid_config() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![
                serde_json::json!(0.001),
                serde_json::json!(0.01),
                serde_json::json!(0.05),
                serde_json::json!(0.1),
                serde_json::json!(0.5),
            ],
        );
        sweep.insert(
            "bs".into(),
            vec![
                serde_json::json!(8.0),
                serde_json::json!(16.0),
                serde_json::json!(32.0),
                serde_json::json!(64.0),
            ],
        );

        // Create enough history for one generation
        let history: Vec<ExperimentResult> = (0..6)
            .map(|i| {
                let lr = 0.001 + i as f64 * 0.1;
                let bs = 8.0 + i as f64 * 8.0;
                make_result(
                    &format!("exp-{i}"),
                    0.5 + i as f64 * 0.05,
                    vec![("lr", lr), ("bs", bs)],
                )
            })
            .collect();

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let cmaes = CmaEs::new(Some(6));
        let suggestion = cmaes.suggest(&ctx).unwrap();
        assert!(suggestion.config.parameters.contains_key("lr"));
        assert!(suggestion.config.parameters.contains_key("bs"));
        // Values should be from sweep space
        let lr = suggestion.config.parameters["lr"].as_f64().unwrap();
        assert!(
            [0.001, 0.01, 0.05, 0.1, 0.5].contains(&lr),
            "lr={lr} not in sweep space"
        );
    }

    #[test]
    fn test_cmaes_name() {
        let cmaes = CmaEs::new(None);
        assert_eq!(cmaes.name(), "cmaes");
    }

    #[test]
    fn test_cholesky_identity() {
        let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky(&mat).unwrap();
        assert!((l[0][0] - 1.0).abs() < 1e-10);
        assert!((l[1][1] - 1.0).abs() < 1e-10);
        assert!(l[0][1].abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_cholesky_positive_definite() {
        let mat = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&mat).unwrap();
        // Verify L * L^T = mat
        for i in 0..2 {
            for j in 0..2 {
                let sum: f64 = (0..2).map(|k| l[i][k] * l[j][k]).sum();
                assert!((sum - mat[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_snap_to_nearest() {
        assert!((snap_to_nearest(0.03, &[0.01, 0.05, 0.1]) - 0.01).abs() < 1e-10);
        assert!((snap_to_nearest(0.04, &[0.01, 0.05, 0.1]) - 0.05).abs() < 1e-10);
        assert!((snap_to_nearest(0.08, &[0.01, 0.05, 0.1]) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_build_strategy_cmaes() {
        let s = crate::strategy::build_strategy("cmaes", 5, 0.02).unwrap();
        assert_eq!(s.name(), "cmaes");
        let s2 = crate::strategy::build_strategy("cma_es", 5, 0.02).unwrap();
        assert_eq!(s2.name(), "cmaes");
    }
}
