use crate::learning::{Confidence, GradientDirection, Learning, ParamGradient};
use mobius_core::experiment::ExperimentResult;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Append-only JSONL store for Learning entries.
///
/// Mirrors `~/.paloa/optimizer/learnings.jsonl` from `experiment_learnings.py`.
pub struct LearningStore {
    path: PathBuf,
}

impl LearningStore {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self { path })
    }

    pub fn append(&mut self, learning: &Learning) -> anyhow::Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(learning)?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    pub fn load_all(&self) -> anyhow::Result<Vec<Learning>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut results = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Learning>(trimmed) {
                Ok(l) => results.push(l),
                Err(e) => tracing::warn!("Skipping malformed learning: {}", e),
            }
        }
        Ok(results)
    }

    /// Compute gradient signal for a single parameter.
    ///
    /// Port of `experiment_learnings.py:get_param_gradient`.
    pub fn get_param_gradient(
        &self,
        param: &str,
        primary_metric: &str,
    ) -> anyhow::Result<ParamGradient> {
        let learnings = self.load_all()?;

        struct Observation {
            f1_delta: f64,
            direction: String, // "increase" or "decrease"
            _new_f1: f64,
            new_value: serde_json::Value,
        }

        let mut observations: Vec<Observation> = Vec::new();

        for learning in &learnings {
            for changed in &learning.changed_params {
                if changed.param != param {
                    continue;
                }
                let f1_delta = learning
                    .metric_deltas
                    .get(primary_metric)
                    .copied()
                    .unwrap_or(0.0);

                let direction = match (&changed.old_value, &changed.new_value) {
                    (serde_json::Value::Number(old), serde_json::Value::Number(new)) => {
                        let o = old.as_f64().unwrap_or(0.0);
                        let n = new.as_f64().unwrap_or(0.0);
                        if n > o {
                            "increase"
                        } else {
                            "decrease"
                        }
                    }
                    _ => "unknown",
                };

                // Approximate new_f1 from delta
                let prev_f1 = learning
                    .metric_deltas
                    .get(primary_metric)
                    .map(|d| {
                        // We don't store prev_f1 directly in Learning, estimate from delta
                        // This is a simplification
                        0.0 - d // placeholder, will be overridden below
                    })
                    .unwrap_or(0.0);
                let _ = prev_f1;

                observations.push(Observation {
                    f1_delta,
                    direction: direction.to_string(),
                    _new_f1: f1_delta, // Use delta as proxy for ranking
                    new_value: changed.new_value.clone(),
                });
            }
        }

        if observations.is_empty() {
            return Ok(ParamGradient {
                param: param.to_string(),
                num_observations: 0,
                avg_metric_delta: 0.0,
                best_direction: GradientDirection::Inconclusive,
                best_value: None,
                confidence: Confidence::Low,
            });
        }

        let avg_delta =
            observations.iter().map(|o| o.f1_delta).sum::<f64>() / observations.len() as f64;

        let increase_deltas: Vec<f64> = observations
            .iter()
            .filter(|o| o.direction == "increase")
            .map(|o| o.f1_delta)
            .collect();
        let decrease_deltas: Vec<f64> = observations
            .iter()
            .filter(|o| o.direction == "decrease")
            .map(|o| o.f1_delta)
            .collect();

        let best_direction = if !increase_deltas.is_empty() && !decrease_deltas.is_empty() {
            let inc_avg =
                increase_deltas.iter().sum::<f64>() / increase_deltas.len() as f64;
            let dec_avg =
                decrease_deltas.iter().sum::<f64>() / decrease_deltas.len() as f64;
            if inc_avg > dec_avg {
                GradientDirection::IncreaseHelps
            } else {
                GradientDirection::DecreaseHelps
            }
        } else if !increase_deltas.is_empty() {
            if increase_deltas.iter().sum::<f64>() > 0.0 {
                GradientDirection::IncreaseHelps
            } else {
                GradientDirection::DecreaseHelps
            }
        } else if !decrease_deltas.is_empty() {
            if decrease_deltas.iter().sum::<f64>() > 0.0 {
                GradientDirection::DecreaseHelps
            } else {
                GradientDirection::IncreaseHelps
            }
        } else {
            GradientDirection::Inconclusive
        };

        // Best value: from observation with highest f1_delta
        let best_obs = observations
            .iter()
            .max_by(|a, b| a.f1_delta.partial_cmp(&b.f1_delta).unwrap_or(std::cmp::Ordering::Equal));
        let best_value = best_obs.map(|o| o.new_value.clone());

        let confidence = if observations.len() >= 5 {
            Confidence::High
        } else if observations.len() >= 2 {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        Ok(ParamGradient {
            param: param.to_string(),
            num_observations: observations.len(),
            avg_metric_delta: (avg_delta * 10000.0).round() / 10000.0,
            best_direction,
            best_value,
            confidence,
        })
    }

    /// Find parameters that have never been varied in experiment history.
    pub fn get_untried_dimensions(
        &self,
        history: &[ExperimentResult],
        sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    ) -> Vec<String> {
        if history.len() < 2 {
            return sweep_space.keys().cloned().collect();
        }

        let mut varied: HashSet<String> = HashSet::new();
        let first = &history[0].config.parameters;

        for entry in &history[1..] {
            for (key, val) in &entry.config.parameters {
                if let Some(first_val) = first.get(key)
                    && first_val != val
                {
                    varied.insert(key.clone());
                }
            }
        }

        sweep_space
            .keys()
            .filter(|k| !varied.contains(*k))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::ParamDelta;
    use tempfile::NamedTempFile;

    fn make_learning(param: &str, old: f64, new: f64, f1_delta: f64) -> Learning {
        let mut metric_deltas = HashMap::new();
        metric_deltas.insert("f1".into(), f1_delta);
        Learning {
            changed_params: vec![ParamDelta {
                param: param.to_string(),
                old_value: serde_json::json!(old),
                new_value: serde_json::json!(new),
            }],
            metric_deltas,
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_append_and_load() {
        let file = NamedTempFile::new().unwrap();
        let mut store = LearningStore::new(file.path()).unwrap();
        store.append(&make_learning("ball_conf", 0.28, 0.25, 0.02)).unwrap();
        store.append(&make_learning("ball_conf", 0.25, 0.30, -0.01)).unwrap();
        let all = store.load_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_gradient_increase_helps() {
        let file = NamedTempFile::new().unwrap();
        let mut store = LearningStore::new(file.path()).unwrap();
        // Increasing ball_conf helps
        store.append(&make_learning("ball_conf", 0.25, 0.28, 0.03)).unwrap();
        store.append(&make_learning("ball_conf", 0.28, 0.30, 0.02)).unwrap();
        let grad = store.get_param_gradient("ball_conf", "f1").unwrap();
        assert_eq!(grad.num_observations, 2);
        assert_eq!(grad.best_direction, GradientDirection::IncreaseHelps);
        assert!(grad.avg_metric_delta > 0.0);
        assert_eq!(grad.confidence, Confidence::Medium);
    }

    #[test]
    fn test_gradient_empty() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let grad = store.get_param_gradient("ball_conf", "f1").unwrap();
        assert_eq!(grad.num_observations, 0);
        assert_eq!(grad.best_direction, GradientDirection::Inconclusive);
    }

    #[test]
    fn test_untried_dimensions() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut params1 = HashMap::new();
        params1.insert("ball_conf".into(), serde_json::json!(0.28));
        params1.insert("dedup_window".into(), serde_json::json!(6.0));

        let mut params2 = HashMap::new();
        params2.insert("ball_conf".into(), serde_json::json!(0.25));
        params2.insert("dedup_window".into(), serde_json::json!(6.0));

        let history = vec![
            ExperimentResult {
                id: "1".into(),
                timestamp: chrono::Utc::now(),
                config: mobius_core::experiment::ExperimentConfig {
                    parameters: params1,
                    metadata: HashMap::new(),
                },
                metrics: HashMap::new(),
                per_segment: HashMap::new(),
                duration_secs: 0.0,
                cost_usd: None,
                status: mobius_core::experiment::ExperimentStatus::Success,
                error: None,
            },
            ExperimentResult {
                id: "2".into(),
                timestamp: chrono::Utc::now(),
                config: mobius_core::experiment::ExperimentConfig {
                    parameters: params2,
                    metadata: HashMap::new(),
                },
                metrics: HashMap::new(),
                per_segment: HashMap::new(),
                duration_secs: 0.0,
                cost_usd: None,
                status: mobius_core::experiment::ExperimentStatus::Success,
                error: None,
            },
        ];

        let mut sweep_space = HashMap::new();
        sweep_space.insert("ball_conf".into(), vec![serde_json::json!(0.25)]);
        sweep_space.insert("dedup_window".into(), vec![serde_json::json!(5.0)]);
        sweep_space.insert("vlm_temp".into(), vec![serde_json::json!(0.5)]);

        let untried = store.get_untried_dimensions(&history, &sweep_space);
        // ball_conf was varied (0.28 → 0.25), dedup_window and vlm_temp were not
        assert!(!untried.contains(&"ball_conf".to_string()));
        assert!(untried.contains(&"dedup_window".to_string()));
        assert!(untried.contains(&"vlm_temp".to_string()));
    }
}
