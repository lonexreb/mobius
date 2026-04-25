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
    /// Shell command to execute for each experiment.
    #[serde(default)]
    pub command: Option<String>,
    /// Map parameter names to environment variables passed to the command.
    #[serde(default)]
    pub env_map: HashMap<String, String>,
    /// Timeout per experiment run in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_budget() -> f64 {
    20.0
}
fn default_cost() -> f64 {
    1.0
}
fn default_timeout() -> u64 {
    600
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
    /// SSH backend configuration (used when `backend = "ssh"`).
    #[serde(default)]
    pub ssh: Option<crate::ssh_backend::SshConfig>,
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
    /// Maximum retries for failed experiments (default: 1).
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    /// Strategy-specific hyperparameters.
    #[serde(default)]
    pub strategy_params: StrategyParams,
}

/// Strategy-specific hyperparameters configurable via mobius.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    /// TPE good/bad split quantile (default: 0.25).
    #[serde(default = "default_tpe_gamma")]
    pub tpe_gamma: f64,
    /// CMA-ES population size (None = auto from dimensionality).
    #[serde(default)]
    pub cmaes_population_size: Option<usize>,
    /// CMA-ES exploration constant (default: sqrt(2)).
    #[serde(default = "default_ucb1_exploration")]
    pub ucb1_exploration_constant: f64,
    /// PBT population size (default: 8).
    #[serde(default = "default_pbt_population")]
    pub pbt_population_size: usize,
    /// NSGA-II objectives (default: ["f1", "precision"]).
    #[serde(default = "default_nsga_objectives")]
    pub nsga_objectives: Vec<String>,
    /// NSGA-II population size (default: 20).
    #[serde(default = "default_nsga_population")]
    pub nsga_population_size: usize,
    /// Hyperband max resource (default: 81).
    #[serde(default = "default_hyperband_max_resource")]
    pub hyperband_max_resource: usize,
    /// Hyperband reduction factor (default: 3).
    #[serde(default = "default_hyperband_eta")]
    pub hyperband_eta: usize,
    /// Outcome constraints: upper bounds on metrics (e.g., latency < 100).
    #[serde(default)]
    pub constraints_upper: HashMap<String, f64>,
    /// Outcome constraints: lower bounds on metrics.
    #[serde(default)]
    pub constraints_lower: HashMap<String, f64>,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            tpe_gamma: default_tpe_gamma(),
            cmaes_population_size: None,
            ucb1_exploration_constant: default_ucb1_exploration(),
            pbt_population_size: default_pbt_population(),
            nsga_objectives: default_nsga_objectives(),
            nsga_population_size: default_nsga_population(),
            hyperband_max_resource: default_hyperband_max_resource(),
            hyperband_eta: default_hyperband_eta(),
            constraints_upper: HashMap::new(),
            constraints_lower: HashMap::new(),
        }
    }
}

fn default_tpe_gamma() -> f64 {
    0.25
}
fn default_ucb1_exploration() -> f64 {
    std::f64::consts::SQRT_2
}
fn default_pbt_population() -> usize {
    8
}
fn default_nsga_objectives() -> Vec<String> {
    vec!["f1".into(), "precision".into()]
}
fn default_nsga_population() -> usize {
    20
}
fn default_hyperband_max_resource() -> usize {
    81
}
fn default_hyperband_eta() -> usize {
    3
}

fn default_plateau_window() -> usize {
    5
}
fn default_plateau_threshold() -> f64 {
    0.02
}
fn default_max_retries() -> usize {
    1
}

impl MobiusConfig {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}
