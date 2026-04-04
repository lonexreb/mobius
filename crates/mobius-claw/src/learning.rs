use mobius_core::experiment::ExperimentResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A learned signal from comparing two experiments.
///
/// Mirrors `experiment_learnings.py:append_learning`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Learning {
    pub changed_params: Vec<ParamDelta>,
    pub metric_deltas: HashMap<String, f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDelta {
    pub param: String,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
}

/// Gradient signal for a single parameter.
///
/// Mirrors `experiment_learnings.py:get_param_gradient`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamGradient {
    pub param: String,
    pub num_observations: usize,
    pub avg_metric_delta: f64,
    pub best_direction: GradientDirection,
    pub best_value: Option<serde_json::Value>,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientDirection {
    IncreaseHelps,
    DecreaseHelps,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    High,   // >= 5 observations
    Medium, // >= 2 observations
    Low,    // < 2 observations
}

/// Extract a learning from two experiment results.
pub fn extract_learning(
    prev: &ExperimentResult,
    curr: &ExperimentResult,
    primary_metric: &str,
) -> Option<Learning> {
    let mut changed_params = Vec::new();

    for (key, new_val) in &curr.config.parameters {
        if let Some(old_val) = prev.config.parameters.get(key)
            && old_val != new_val
        {
            changed_params.push(ParamDelta {
                param: key.clone(),
                old_value: old_val.clone(),
                new_value: new_val.clone(),
            });
        }
    }

    if changed_params.is_empty() {
        return None;
    }

    let mut metric_deltas = HashMap::new();
    for metric in [primary_metric, "precision", "recall"] {
        let prev_val = prev.metrics.get(metric).unwrap_or(&0.0);
        let curr_val = curr.metrics.get(metric).unwrap_or(&0.0);
        metric_deltas.insert(metric.to_string(), curr_val - prev_val);
    }

    Some(Learning {
        changed_params,
        metric_deltas,
        timestamp: chrono::Utc::now(),
    })
}
