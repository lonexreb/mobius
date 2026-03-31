use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A complete experiment configuration.
///
/// Generalizes pipeline_optimizer.py's config dicts + env var overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub parameters: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result of a single experiment run.
///
/// Generalizes the AUTORESEARCH_RESULT JSON and pipeline_optimizer.py's result format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub config: ExperimentConfig,
    /// Aggregate metrics (e.g., {"f1": 0.786, "precision": 0.71}).
    pub metrics: HashMap<String, f64>,
    /// Per-segment metrics for cross-validation.
    #[serde(default)]
    pub per_segment: HashMap<String, HashMap<String, f64>>,
    pub duration_secs: f64,
    pub cost_usd: Option<f64>,
    pub status: ExperimentStatus,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    Success,
    Error,
    Timeout,
}

/// Trait for experiment storage backends.
///
/// Generalizes experiment_history.py (JSONL append-only log).
pub trait ExperimentStore: Send + Sync {
    fn append(&mut self, result: &ExperimentResult) -> anyhow::Result<()>;
    fn load_all(&self) -> anyhow::Result<Vec<ExperimentResult>>;
    fn get_best(&self, metric: &str) -> anyhow::Result<Option<ExperimentResult>>;
    fn get_recent(&self, n: usize) -> anyhow::Result<Vec<ExperimentResult>>;
    fn count(&self) -> anyhow::Result<usize>;
}
