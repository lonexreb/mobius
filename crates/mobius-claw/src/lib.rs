pub mod agent;
pub mod hooks;
pub mod learning;
pub mod learning_store;
pub mod strategy;

use mobius_core::experiment::ExperimentResult;

/// Decision after evaluating an experiment result.
pub enum Decision {
    Continue,
    SwitchStrategy(String),
    Stop(StopReason),
}

pub enum StopReason {
    TargetsMet,
    BudgetExhausted,
    MaxIterationsReached,
    Plateau,
    UserInterrupted,
}

/// Strategy phase for experiment planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyPhase {
    ParameterTuning,
    AlgorithmTuning,
    StructuralChanges,
    Custom(String),
}

/// Detects if recent experiments show plateau behavior.
pub fn is_plateaued(recent: &[ExperimentResult], metric: &str, threshold: f64) -> bool {
    if recent.len() < 2 {
        return false;
    }
    let values: Vec<f64> = recent
        .iter()
        .filter_map(|r| r.metrics.get(metric).copied())
        .collect();
    if values.len() < 2 {
        return false;
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (max - min) < threshold
}
