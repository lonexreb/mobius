use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Project configuration loaded from mobius.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobiusConfig {
    pub project: ProjectConfig,
    #[serde(default)]
    pub experiment: ExperimentSection,
    #[serde(default)]
    pub bench: BenchSection,
    #[serde(default)]
    pub compute: ComputeSection,
    #[serde(default)]
    pub agent: AgentSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "0.1.0".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentSection {
    #[serde(default = "default_budget")]
    pub budget_usd: f64,
    #[serde(default = "default_cost")]
    pub cost_per_run: f64,
    #[serde(default)]
    pub targets: HashMap<String, f64>,
    #[serde(default)]
    pub sweep_space: HashMap<String, Vec<serde_json::Value>>,
}

fn default_budget() -> f64 {
    20.0
}
fn default_cost() -> f64 {
    1.0
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchSection {
    #[serde(default = "default_match_window")]
    pub match_window: f64,
    #[serde(default)]
    pub dimensions: Vec<DimensionConfig>,
    #[serde(default = "default_store_path")]
    pub store_path: String,
}

fn default_match_window() -> f64 {
    6.0
}
fn default_store_path() -> String {
    "runs/".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionConfig {
    pub name: String,
    pub weight: f64,
    pub scorer: String,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComputeSection {
    #[serde(default = "default_backend")]
    pub backend: String,
    pub gpu_type: Option<String>,
}

fn default_backend() -> String {
    "local".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentSection {
    #[serde(default)]
    pub strategies: Vec<String>,
    #[serde(default = "default_plateau_window")]
    pub plateau_window: usize,
    #[serde(default = "default_plateau_threshold")]
    pub plateau_threshold: f64,
}

fn default_plateau_window() -> usize {
    5
}
fn default_plateau_threshold() -> f64 {
    0.02
}

impl MobiusConfig {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}
