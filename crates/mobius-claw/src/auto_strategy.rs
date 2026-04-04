//! Auto-strategy selection based on problem characteristics.
//!
//! Inspired by Optuna's AutoSampler, `AutoStrategy` inspects the sweep space,
//! history length, and optimization targets to pick the best delegate strategy
//! for the current problem without any manual configuration.

use crate::cmaes::CmaEs;
use crate::nsga::NsgaTwo;
use crate::strategy::{
    GradientGuidedTuning, RandomSearch, Strategy, StrategyContext, Suggestion, TpeSearch,
};
use std::collections::HashMap;

/// Meta-strategy that auto-selects the best optimization algorithm based on
/// problem characteristics.
///
/// Classification logic:
/// 1. Cold start (< 4 history entries) -> `RandomSearch`
/// 2. Multi-objective (> 1 target) -> `NsgaTwo` with objectives from target keys, population 20
/// 3. All categorical parameters (no numeric values) -> `TpeSearch(0.25)`
/// 4. High dimensionality (> 10 params) -> `TpeSearch(0.25)`
/// 5. Low-dimensional all-numeric (<= 5 params) -> `CmaEs(None)`
/// 6. Default -> `GradientGuidedTuning(plateau_window, plateau_threshold)`
pub struct AutoStrategy {
    /// Window size for plateau detection, forwarded to `GradientGuidedTuning`.
    plateau_window: usize,
    /// Threshold for plateau detection, forwarded to `GradientGuidedTuning`.
    plateau_threshold: f64,
}

impl AutoStrategy {
    /// Create a new `AutoStrategy` with the given plateau detection parameters.
    ///
    /// These parameters are forwarded to `GradientGuidedTuning` when it is
    /// selected as the delegate strategy.
    pub fn new(plateau_window: usize, plateau_threshold: f64) -> Self {
        Self {
            plateau_window,
            plateau_threshold,
        }
    }

    /// Select the best strategy for the current problem characteristics.
    ///
    /// Returns a boxed strategy that will handle the actual `suggest()` call.
    fn select_strategy(&self, ctx: &StrategyContext) -> Box<dyn Strategy> {
        // 1. Cold start: not enough history for informed decisions
        if ctx.history.len() < 4 {
            return Box::new(RandomSearch);
        }

        // 2. Multi-objective: use NSGA-II with objectives from targets keys
        if ctx.targets.len() > 1 {
            let objectives: Vec<String> = ctx.targets.keys().cloned().collect();
            return Box::new(NsgaTwo::new(objectives, 20));
        }

        // 3. All categorical (no numeric values in sweep space)
        if !is_all_numeric(ctx.sweep_space) && count_numeric_params(ctx.sweep_space) == 0 {
            return Box::new(TpeSearch::new(0.25));
        }

        // 4. High dimensionality
        if ctx.sweep_space.len() > 10 {
            return Box::new(TpeSearch::new(0.25));
        }

        // 5. Low-dimensional, all numeric -> CMA-ES
        if ctx.sweep_space.len() <= 5 && is_all_numeric(ctx.sweep_space) {
            return Box::new(CmaEs::new(None));
        }

        // 6. Default: gradient-guided tuning
        Box::new(GradientGuidedTuning::new(
            self.plateau_window,
            self.plateau_threshold,
        ))
    }
}

impl Strategy for AutoStrategy {
    fn name(&self) -> &str {
        "auto"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        let delegate = self.select_strategy(ctx);
        delegate.suggest(ctx)
    }
}

/// Check if all parameter values in the sweep space are numeric.
///
/// Returns `true` when every value in every parameter list can be parsed as
/// an `f64`. Returns `false` if the sweep space is empty or any value is
/// non-numeric (e.g., strings, booleans, objects).
fn is_all_numeric(sweep_space: &HashMap<String, Vec<serde_json::Value>>) -> bool {
    if sweep_space.is_empty() {
        return false;
    }
    sweep_space
        .values()
        .all(|vals| !vals.is_empty() && vals.iter().all(|v| v.as_f64().is_some()))
}

/// Count the number of parameters in the sweep space that have only numeric
/// values.
///
/// A parameter is considered numeric if every value in its list can be
/// interpreted as an `f64`.
fn count_numeric_params(sweep_space: &HashMap<String, Vec<serde_json::Value>>) -> usize {
    sweep_space
        .values()
        .filter(|vals| !vals.is_empty() && vals.iter().all(|v| v.as_f64().is_some()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_store::LearningStore;
    use crate::strategy::StrategyContext;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use tempfile::NamedTempFile;

    fn make_result(id: &str, f1: f64, params: Vec<(&str, serde_json::Value)>) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut parameters = HashMap::new();
        for (k, v) in params {
            parameters.insert(k.to_string(), v);
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
    fn test_auto_cold_start() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let auto = AutoStrategy::new(5, 0.02);

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );

        // Only 1 history entry — should fall back to RandomSearch
        let ctx = StrategyContext {
            history: &[make_result("1", 0.5, vec![("lr", serde_json::json!(0.01))])],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = auto.suggest(&ctx).unwrap();
        assert!(
            suggestion.rationale.contains("Random"),
            "Expected random fallback for cold start, got: {}",
            suggestion.rationale
        );
    }

    #[test]
    fn test_auto_selects_strategy() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let auto = AutoStrategy::new(5, 0.02);

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![
                serde_json::json!(0.01),
                serde_json::json!(0.05),
                serde_json::json!(0.1),
                serde_json::json!(0.5),
                serde_json::json!(1.0),
            ],
        );

        let history: Vec<ExperimentResult> = (0..5)
            .map(|i| {
                make_result(
                    &format!("exp-{i}"),
                    0.5 + i as f64 * 0.05,
                    vec![("lr", serde_json::json!(0.01 + i as f64 * 0.02))],
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

        let suggestion = auto.suggest(&ctx).unwrap();
        assert!(!suggestion.config.parameters.is_empty());
        assert!(!suggestion.rationale.is_empty());
    }

    #[test]
    fn test_auto_name() {
        let auto = AutoStrategy::new(5, 0.02);
        assert_eq!(auto.name(), "auto");
    }

    #[test]
    fn test_build_strategy_auto() {
        let s = crate::strategy::build_strategy("auto", 5, 0.02).unwrap();
        assert_eq!(s.name(), "auto");
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_all_numeric_true() {
        let mut space = HashMap::new();
        space.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );
        space.insert(
            "bs".into(),
            vec![serde_json::json!(16), serde_json::json!(32)],
        );
        assert!(is_all_numeric(&space));
    }

    #[test]
    fn test_is_all_numeric_false_with_strings() {
        let mut space = HashMap::new();
        space.insert(
            "model".into(),
            vec![serde_json::json!("gpt-4"), serde_json::json!("claude-3")],
        );
        assert!(!is_all_numeric(&space));
    }

    #[test]
    fn test_is_all_numeric_empty() {
        let space: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
        assert!(!is_all_numeric(&space));
    }

    #[test]
    fn test_count_numeric_params() {
        let mut space = HashMap::new();
        space.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );
        space.insert(
            "model".into(),
            vec![serde_json::json!("resnet"), serde_json::json!("vgg")],
        );
        space.insert(
            "bs".into(),
            vec![serde_json::json!(16), serde_json::json!(32)],
        );
        assert_eq!(count_numeric_params(&space), 2);
    }

    #[test]
    fn test_auto_multi_objective_delegates() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();
        let auto = AutoStrategy::new(5, 0.02);

        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![
                serde_json::json!(0.01),
                serde_json::json!(0.05),
                serde_json::json!(0.10),
                serde_json::json!(0.50),
                serde_json::json!(1.00),
            ],
        );

        let mut history = Vec::new();
        for i in 0..5 {
            let mut r = make_result(
                &format!("r{i}"),
                0.5 + i as f64 * 0.05,
                vec![("lr", serde_json::json!(0.01 + i as f64 * 0.1))],
            );
            r.metrics.insert("precision".into(), 0.6 + i as f64 * 0.03);
            history.push(r);
        }

        let mut targets = HashMap::new();
        targets.insert("f1".into(), 0.90);
        targets.insert("precision".into(), 0.85);

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &targets,
            primary_metric: "f1",
            learning_store: &store,
        };

        // With > 1 target and enough history, auto should delegate to NSGA-II
        let suggestion = auto.suggest(&ctx).unwrap();
        assert!(!suggestion.config.parameters.is_empty());
    }
}
