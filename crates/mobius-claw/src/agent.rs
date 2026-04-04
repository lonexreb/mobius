use crate::hooks::{HookAction, PostEvaluateHook, PreExecuteHook};
use crate::learning::extract_learning;
use crate::learning_store::LearningStore;
use crate::strategy::{Strategy, StrategyContext, build_strategy};
use crate::{Decision, StopReason, StrategyPhase};
use mobius_core::budget::BudgetGuard;
use mobius_core::compute::{ComputeBackend, OutputParser};
use mobius_core::experiment::{ExperimentResult, ExperimentStatus, ExperimentStore};
use std::collections::HashMap;

/// Configuration for the autonomous agent.
pub struct AgentConfig {
    pub max_iterations: usize,
    pub targets: HashMap<String, f64>,
    pub cost_per_run: f64,
    pub primary_metric: String,
    pub command_template: String,
    pub env_map: HashMap<String, String>,
    pub production_config: HashMap<String, serde_json::Value>,
    pub sweep_space: HashMap<String, Vec<serde_json::Value>>,
    pub timeout_secs: u64,
    /// Available strategy names for rotation on consecutive reverts.
    pub strategies: Vec<String>,
    /// Plateau detection window size.
    pub plateau_window: usize,
    /// Plateau detection threshold.
    pub plateau_threshold: f64,
}

/// Mutable state of the agent across iterations.
pub struct AgentState {
    pub iteration: usize,
    pub strategy_phase: StrategyPhase,
    pub consecutive_reverts: usize,
    pub best_result: Option<ExperimentResult>,
    pub strategy_index: usize,
}

/// Final report after the agent loop completes.
pub struct AgentReport {
    pub iterations: usize,
    pub stop_reason: StopReason,
    pub best_result: Option<ExperimentResult>,
    pub total_cost: f64,
}

/// The autonomous experiment loop.
///
/// Implements the 7-step ORIENT→DECIDE cycle from `paloa-claw/AGENT.md`.
pub struct AgentLoop {
    pub config: AgentConfig,
    strategy: Box<dyn Strategy>,
    compute: Box<dyn ComputeBackend>,
    experiment_store: Box<dyn ExperimentStore>,
    learning_store: LearningStore,
    budget: BudgetGuard,
    pre_hooks: Vec<Box<dyn PreExecuteHook>>,
    post_hooks: Vec<Box<dyn PostEvaluateHook>>,
    state: AgentState,
}

impl AgentLoop {
    pub fn new(
        config: AgentConfig,
        strategy: Box<dyn Strategy>,
        compute: Box<dyn ComputeBackend>,
        experiment_store: Box<dyn ExperimentStore>,
        learning_store: LearningStore,
        budget: BudgetGuard,
    ) -> Self {
        Self {
            config,
            strategy,
            compute,
            experiment_store,
            learning_store,
            budget,
            pre_hooks: Vec::new(),
            post_hooks: Vec::new(),
            state: AgentState {
                iteration: 0,
                strategy_phase: StrategyPhase::ParameterTuning,
                consecutive_reverts: 0,
                best_result: None,
                strategy_index: 0,
            },
        }
    }

    pub fn add_pre_hook(&mut self, hook: Box<dyn PreExecuteHook>) {
        self.pre_hooks.push(hook);
    }

    pub fn add_post_hook(&mut self, hook: Box<dyn PostEvaluateHook>) {
        self.post_hooks.push(hook);
    }

    /// Run the autonomous loop until a stop condition is met.
    pub fn run(&mut self) -> anyhow::Result<AgentReport> {
        loop {
            let decision = self.step()?;
            match decision {
                Decision::Continue => continue,
                Decision::SwitchStrategy(_) => {
                    if self.config.strategies.len() <= 1 {
                        return Ok(AgentReport {
                            iterations: self.state.iteration,
                            stop_reason: StopReason::Plateau,
                            best_result: self.state.best_result.clone(),
                            total_cost: self.budget.spent,
                        });
                    }
                    self.state.strategy_index =
                        (self.state.strategy_index + 1) % self.config.strategies.len();
                    let next = &self.config.strategies[self.state.strategy_index];
                    tracing::info!("Switching strategy to: {}", next);
                    self.strategy = build_strategy(
                        next,
                        self.config.plateau_window,
                        self.config.plateau_threshold,
                    )?;
                    self.state.consecutive_reverts = 0;
                    continue;
                }
                Decision::Stop(reason) => {
                    return Ok(AgentReport {
                        iterations: self.state.iteration,
                        stop_reason: reason,
                        best_result: self.state.best_result.clone(),
                        total_cost: self.budget.remaining(),
                    });
                }
            }
        }
    }

    /// Execute a single iteration of the 7-step cycle.
    pub fn step(&mut self) -> anyhow::Result<Decision> {
        self.state.iteration += 1;
        let iter = self.state.iteration;
        tracing::info!("--- Iteration {} ---", iter);

        // Step 1: ORIENT
        let history = self.experiment_store.load_all()?;
        let best = history
            .iter()
            .filter(|r| r.metrics.contains_key(&self.config.primary_metric))
            .max_by(|a, b| {
                let va = a.metrics.get(&self.config.primary_metric).unwrap_or(&0.0);
                let vb = b.metrics.get(&self.config.primary_metric).unwrap_or(&0.0);
                va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
        self.state.best_result = best;

        tracing::info!(
            "Budget: ${:.2} remaining ({} experiments left)",
            self.budget.remaining(),
            self.budget.experiments_remaining(self.config.cost_per_run)
        );

        // Step 2: RESEARCH (no-op, Phase 3 MCP)

        // Step 3: PROPOSE
        let ctx = StrategyContext {
            history: &history,
            production_config: &self.config.production_config,
            sweep_space: &self.config.sweep_space,
            targets: &self.config.targets,
            primary_metric: &self.config.primary_metric,
            learning_store: &self.learning_store,
        };
        let suggestion = self.strategy.suggest(&ctx)?;
        tracing::info!("Proposed: {}", suggestion.rationale);

        // Step 4: EXECUTE — pre-hooks
        for hook in &self.pre_hooks {
            match hook.check(&self.budget, self.config.cost_per_run)? {
                HookAction::Proceed => {}
                HookAction::Warn(msg) => tracing::warn!("[{}] {}", hook.name(), msg),
                HookAction::Block(msg) => {
                    tracing::error!("[{}] Blocked: {}", hook.name(), msg);
                    return Ok(Decision::Stop(StopReason::BudgetExhausted));
                }
            }
        }

        // Build env vars from config
        let env = mobius_core::compute::config_to_env(
            &suggestion.config.parameters,
            &self.config.env_map,
        );

        let output = self.compute.submit(
            &self.config.command_template,
            &env,
            self.config.timeout_secs,
        )?;

        let parsed = OutputParser::parse(&output.stdout, &output.stderr);

        let result_id = format!("exp-{:04}", iter);
        let mut metrics = parsed.metrics;
        if let Some(bs) = parsed.bench_score {
            metrics.insert("bench_score".into(), bs);
        }

        let result = ExperimentResult {
            id: result_id,
            timestamp: chrono::Utc::now(),
            config: suggestion.config,
            metrics,
            per_segment: HashMap::new(),
            duration_secs: output.duration_secs,
            cost_usd: Some(self.config.cost_per_run),
            status: if output.exit_code == 0 {
                ExperimentStatus::Success
            } else {
                ExperimentStatus::Error
            },
            error: if output.exit_code != 0 {
                Some(output.stderr.chars().take(200).collect())
            } else {
                None
            },
        };

        self.experiment_store.append(&result)?;

        // Step 5: EVALUATE — post-hooks
        let updated_history = self.experiment_store.load_all()?;
        let mut should_revert = false;
        for hook in &self.post_hooks {
            match hook.check(&result, &updated_history, &self.config.primary_metric)? {
                HookAction::Proceed => {}
                HookAction::Warn(msg) => {
                    tracing::warn!("[{}] {}", hook.name(), msg);
                    if msg.contains("Regression") {
                        should_revert = true;
                    }
                }
                HookAction::Block(msg) => {
                    tracing::error!("[{}] {}", hook.name(), msg);
                    should_revert = true;
                }
            }
        }

        // Step 6: LEARN
        self.budget.record_spend(self.config.cost_per_run)?;

        if let Some(ref prev_best) = self.state.best_result
            && let Some(learning) =
                extract_learning(prev_best, &result, &self.config.primary_metric)
        {
            self.learning_store.append(&learning)?;
            let delta = learning
                .metric_deltas
                .get(&self.config.primary_metric)
                .unwrap_or(&0.0);
            tracing::info!("{} delta: {:+.4}", self.config.primary_metric, delta);
        }

        if should_revert {
            self.state.consecutive_reverts += 1;
            tracing::info!("REVERT (consecutive: {})", self.state.consecutive_reverts);
        } else {
            self.state.consecutive_reverts = 0;
            let current_metric = result
                .metrics
                .get(&self.config.primary_metric)
                .copied()
                .unwrap_or(0.0);
            let best_metric = self
                .state
                .best_result
                .as_ref()
                .and_then(|b| b.metrics.get(&self.config.primary_metric).copied())
                .unwrap_or(0.0);
            if current_metric > best_metric {
                tracing::info!("KEEP — new best: {:.4}", current_metric);
                self.state.best_result = Some(result.clone());
            }
        }

        // Step 7: DECIDE
        self.decide(&result)
    }

    fn decide(&self, _result: &ExperimentResult) -> anyhow::Result<Decision> {
        // Check targets met
        if let Some(ref best) = self.state.best_result {
            let all_met = self.config.targets.iter().all(|(metric, target)| {
                best.metrics.get(metric).copied().unwrap_or(0.0) >= *target
            });
            if all_met && !self.config.targets.is_empty() {
                return Ok(Decision::Stop(StopReason::TargetsMet));
            }
        }

        // Check budget
        if !self.budget.can_run(self.config.cost_per_run) {
            return Ok(Decision::Stop(StopReason::BudgetExhausted));
        }

        // Check max iterations
        if self.state.iteration >= self.config.max_iterations {
            return Ok(Decision::Stop(StopReason::MaxIterationsReached));
        }

        // Check consecutive reverts → switch strategy
        if self.state.consecutive_reverts >= 3 {
            return Ok(Decision::SwitchStrategy("next_phase".into()));
        }

        Ok(Decision::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::compute::{ComputeBackend, RawOutput};
    use mobius_core::store::JsonlStore;
    use tempfile::TempDir;

    struct MockBackend {
        results: Vec<String>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockBackend {
        fn new(results: Vec<f64>) -> Self {
            let outputs: Vec<String> = results
                .iter()
                .map(|f1| format!("{{\"f1\": {}}}", f1))
                .collect();
            Self {
                results: outputs,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl ComputeBackend for MockBackend {
        fn submit(
            &self,
            _command: &str,
            _env: &HashMap<String, String>,
            _timeout: u64,
        ) -> anyhow::Result<RawOutput> {
            let idx = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let stdout = self
                .results
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "{\"f1\": 0.5}".into());
            Ok(RawOutput {
                stdout,
                stderr: String::new(),
                exit_code: 0,
                duration_secs: 1.0,
            })
        }
    }

    fn make_agent(
        dir: &TempDir,
        mock: MockBackend,
        max_iter: usize,
        budget: f64,
        targets: HashMap<String, f64>,
    ) -> AgentLoop {
        let history_path = dir.path().join("history.jsonl");
        let learning_path = dir.path().join("learnings.jsonl");

        let mut sweep = HashMap::new();
        sweep.insert(
            "x".into(),
            vec![serde_json::json!(1.0), serde_json::json!(2.0)],
        );

        let mut prod = HashMap::new();
        prod.insert("x".into(), serde_json::json!(1.5));

        let config = AgentConfig {
            max_iterations: max_iter,
            targets,
            cost_per_run: 1.0,
            primary_metric: "f1".into(),
            command_template: "echo test".into(),
            env_map: HashMap::new(),
            production_config: prod,
            sweep_space: sweep,
            timeout_secs: 10,
            strategies: vec!["gradient_guided".into(), "random".into()],
            plateau_window: 5,
            plateau_threshold: 0.02,
        };

        AgentLoop::new(
            config,
            Box::new(crate::strategy::GradientGuidedTuning::new(5, 0.02)),
            Box::new(mock),
            Box::new(JsonlStore::new(&history_path).unwrap()),
            LearningStore::new(&learning_path).unwrap(),
            BudgetGuard::new(budget),
        )
    }

    #[test]
    fn test_stops_at_max_iterations() {
        let dir = TempDir::new().unwrap();
        let mock = MockBackend::new(vec![0.5, 0.6, 0.7]);
        let mut agent = make_agent(&dir, mock, 3, 100.0, HashMap::new());
        let report = agent.run().unwrap();
        assert_eq!(report.iterations, 3);
        assert!(matches!(
            report.stop_reason,
            StopReason::MaxIterationsReached
        ));
    }

    #[test]
    fn test_stops_at_budget() {
        let dir = TempDir::new().unwrap();
        let mock = MockBackend::new(vec![0.5, 0.6]);
        let mut agent = make_agent(&dir, mock, 100, 1.5, HashMap::new());
        // Budget = 1.5, cost per run = 1.0, so should run 1 then stop
        let report = agent.run().unwrap();
        assert!(report.iterations <= 2);
        assert!(matches!(report.stop_reason, StopReason::BudgetExhausted));
    }

    #[test]
    fn test_stops_when_targets_met() {
        let dir = TempDir::new().unwrap();
        let mock = MockBackend::new(vec![0.5, 0.9]); // Second run meets target
        let mut targets = HashMap::new();
        targets.insert("f1".into(), 0.85);
        let mut agent = make_agent(&dir, mock, 100, 100.0, targets);
        let report = agent.run().unwrap();
        assert!(matches!(report.stop_reason, StopReason::TargetsMet));
    }
}
