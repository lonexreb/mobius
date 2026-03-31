use mobius_core::budget::BudgetGuard;
use mobius_core::experiment::ExperimentResult;

/// Action a hook can take.
#[derive(Debug, Clone)]
pub enum HookAction {
    Proceed,
    Warn(String),
    Block(String),
}

/// Pre-execution hook — runs before each experiment.
pub trait PreExecuteHook: Send + Sync {
    fn name(&self) -> &str;
    fn check(
        &self,
        budget: &BudgetGuard,
        cost: f64,
    ) -> anyhow::Result<HookAction>;
}

/// Post-evaluation hook — runs after each experiment is scored.
pub trait PostEvaluateHook: Send + Sync {
    fn name(&self) -> &str;
    fn check(
        &self,
        result: &ExperimentResult,
        history: &[ExperimentResult],
        primary_metric: &str,
    ) -> anyhow::Result<HookAction>;
}

/// Blocks execution if budget is insufficient.
pub struct BudgetCheckHook;

impl PreExecuteHook for BudgetCheckHook {
    fn name(&self) -> &str {
        "budget_check"
    }

    fn check(&self, budget: &BudgetGuard, cost: f64) -> anyhow::Result<HookAction> {
        if budget.can_run(cost) {
            Ok(HookAction::Proceed)
        } else {
            Ok(HookAction::Block(format!(
                "Budget exhausted: ${:.2} remaining, need ${:.2}",
                budget.remaining(),
                cost
            )))
        }
    }
}

/// Detects regression when bench_score drops significantly.
pub struct RegressionDetectionHook {
    pub threshold: f64,
}

impl RegressionDetectionHook {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for RegressionDetectionHook {
    fn default() -> Self {
        Self { threshold: 5.0 }
    }
}

impl PostEvaluateHook for RegressionDetectionHook {
    fn name(&self) -> &str {
        "regression_detection"
    }

    fn check(
        &self,
        result: &ExperimentResult,
        history: &[ExperimentResult],
        primary_metric: &str,
    ) -> anyhow::Result<HookAction> {
        let current = result.metrics.get(primary_metric).copied().unwrap_or(0.0);

        let prev_best = history
            .iter()
            .filter_map(|r| r.metrics.get(primary_metric).copied())
            .fold(f64::NEG_INFINITY, f64::max);

        if prev_best == f64::NEG_INFINITY {
            return Ok(HookAction::Proceed);
        }

        let delta = current - prev_best;
        if delta < -self.threshold {
            Ok(HookAction::Warn(format!(
                "Regression: {} dropped {:.1} points ({:.4} → {:.4}). REVERT recommended.",
                primary_metric,
                -delta,
                prev_best,
                current
            )))
        } else {
            Ok(HookAction::Proceed)
        }
    }
}

/// Detects overfitting when per-segment variance is too high.
pub struct OverfittingDetectionHook {
    pub max_spread: f64,
}

impl OverfittingDetectionHook {
    pub fn new(max_spread: f64) -> Self {
        Self { max_spread }
    }
}

impl Default for OverfittingDetectionHook {
    fn default() -> Self {
        Self { max_spread: 15.0 }
    }
}

impl PostEvaluateHook for OverfittingDetectionHook {
    fn name(&self) -> &str {
        "overfitting_detection"
    }

    fn check(
        &self,
        result: &ExperimentResult,
        _history: &[ExperimentResult],
        primary_metric: &str,
    ) -> anyhow::Result<HookAction> {
        let scores: Vec<f64> = result
            .per_segment
            .values()
            .filter_map(|seg| seg.get(primary_metric).copied())
            .collect();

        if scores.len() < 2 {
            return Ok(HookAction::Proceed);
        }

        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread = (max - min) * 100.0; // Convert to percentage points

        if spread > self.max_spread {
            Ok(HookAction::Warn(format!(
                "Possible overfitting: per-segment {} spread is {:.1}pp (max: {:.1}pp)",
                primary_metric, spread, self.max_spread
            )))
        } else {
            Ok(HookAction::Proceed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::{ExperimentConfig, ExperimentStatus};
    use std::collections::HashMap;

    fn make_result(f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        ExperimentResult {
            id: "test".into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: HashMap::new(),
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 10.0,
            cost_usd: None,
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    #[test]
    fn test_budget_check_pass() {
        let budget = BudgetGuard::new(20.0);
        let hook = BudgetCheckHook;
        match hook.check(&budget, 1.20).unwrap() {
            HookAction::Proceed => {}
            other => panic!("Expected Proceed, got {:?}", other),
        }
    }

    #[test]
    fn test_budget_check_block() {
        let mut budget = BudgetGuard::new(20.0);
        budget.record_spend(19.5).unwrap();
        let hook = BudgetCheckHook;
        match hook.check(&budget, 1.20).unwrap() {
            HookAction::Block(msg) => assert!(msg.contains("Budget exhausted")),
            other => panic!("Expected Block, got {:?}", other),
        }
    }

    #[test]
    fn test_regression_detection() {
        let hook = RegressionDetectionHook::new(0.05); // 5% threshold for 0-1 scale
        let history = vec![make_result(0.80), make_result(0.85)];
        let current = make_result(0.70); // 15% drop, exceeds 5% threshold

        match hook.check(&current, &history, "f1").unwrap() {
            HookAction::Warn(msg) => assert!(msg.contains("Regression")),
            other => panic!("Expected Warn, got {:?}", other),
        }
    }

    #[test]
    fn test_overfitting_detection() {
        let hook = OverfittingDetectionHook::default();
        let mut result = make_result(0.75);
        let mut seg1 = HashMap::new();
        seg1.insert("f1".into(), 0.90);
        let mut seg2 = HashMap::new();
        seg2.insert("f1".into(), 0.60);
        result.per_segment.insert("seg1".into(), seg1);
        result.per_segment.insert("seg2".into(), seg2);

        match hook.check(&result, &[], "f1").unwrap() {
            HookAction::Warn(msg) => assert!(msg.contains("overfitting")),
            other => panic!("Expected Warn, got {:?}", other),
        }
    }
}
