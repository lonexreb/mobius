//! Tool parameter structs and handler helper functions.
//!
//! Each MCP tool has a parameter struct (with `schemars::JsonSchema` for
//! automatic schema generation) and a handler function that operates on
//! [`SharedState`](crate::state::SharedState).

use crate::error::{invalid_params, to_mcp_error};
use crate::state::State;
use mobius_bench::evaluator::Evaluator;
use mobius_bench::matcher::GreedyTimestampMatcher;
use mobius_bench::scorers::{
    ClassificationAccuracyScorer, DimensionScorer, F1Scorer, TimestampMaeScorer,
};
use mobius_claw::agent::{AgentConfig, AgentLoop};
use mobius_claw::hooks::{BudgetCheckHook, OverfittingDetectionHook, RegressionDetectionHook};
use mobius_claw::learning_store::LearningStore;
use mobius_claw::strategy::{StrategyContext, build_strategy};
use mobius_core::budget::BudgetGuard;
use mobius_core::compute::{ComputeBackend, OutputParser, SubprocessBackend};
use mobius_core::config::DimensionConfig;
use mobius_core::experiment::{
    ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore,
};
use mobius_core::loader;
use mobius_core::pareto;
use mobius_core::store::JsonlStore;
use rmcp::model::ErrorData;
use rmcp::schemars;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

/// Parameters for the `mobius_status` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StatusParams {
    /// Primary metric to rank by (default: "f1").
    pub metric: Option<String>,
}

/// Parameters for the `mobius_history` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HistoryParams {
    /// Number of recent experiments to return (default: 10, max: 100).
    pub last: Option<usize>,
}

/// Parameters for the `mobius_run` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunParams {
    /// JSON object of parameter overrides, e.g. `{"learning_rate": 0.01}`.
    pub config_overrides: Option<serde_json::Value>,
    /// Command to execute (overrides mobius.toml default).
    pub command: Option<String>,
    /// Timeout in seconds (default: 600).
    pub timeout_secs: Option<u64>,
}

/// Parameters for the `mobius_evaluate` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EvaluateParams {
    /// Path to predictions JSON file.
    pub predictions_path: String,
    /// Path to ground truth JSON file.
    pub ground_truth_path: String,
    /// Match window in seconds (default from config or 6.0).
    pub match_window: Option<f64>,
}

/// Parameters for the `mobius_suggest` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SuggestParams {
    /// Primary metric to optimize (default: "f1").
    pub metric: Option<String>,
    /// Strategy to use: "gradient_guided", "random", "grid" (default: from config).
    pub strategy: Option<String>,
}

/// Parameters for the `mobius_sweep` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SweepParams {
    /// Sweep spec mapping param names to value arrays.
    /// Example: `{"learning_rate": [0.01, 0.1], "batch_size": [16, 32]}`
    pub spec: serde_json::Value,
    /// Command to execute for each config (overrides mobius.toml default).
    pub command: Option<String>,
    /// Timeout per run in seconds (default: 600).
    pub timeout_secs: Option<u64>,
}

/// Parameters for the `mobius_agent` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AgentParams {
    /// Budget limit in USD (default from config or 20.0).
    pub budget: Option<f64>,
    /// Maximum iterations (default: 50).
    pub max_iterations: Option<usize>,
    /// Primary metric to optimize (default: "f1").
    pub metric: Option<String>,
    /// Strategy to use: "gradient_guided", "random", "grid", "tpe" (default: from config).
    pub strategy: Option<String>,
    /// Enable ASHA trial pruning to stop bad experiments early.
    pub pruning: Option<bool>,
}

/// Parameters for the `mobius_pareto` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ParetoParams {
    /// First metric for Pareto analysis (e.g. "f1").
    pub x_metric: String,
    /// Second metric for Pareto analysis (e.g. "precision").
    pub y_metric: String,
}

/// Parameters for the `mobius_budget` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BudgetParams {
    /// New budget limit to set (optional — omit to just read).
    pub set_limit: Option<f64>,
    /// Amount to record as spent (optional — for manual tracking).
    pub record_spend: Option<f64>,
}

/// Parameters for the `mobius_learnings` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LearningsParams {
    /// Specific parameter to get gradient for (omit for all).
    pub param: Option<String>,
    /// Primary metric for gradient computation (default: "f1").
    pub metric: Option<String>,
}

// ---------------------------------------------------------------------------
// Handler helpers
// ---------------------------------------------------------------------------

/// Handler for `mobius_status`.
pub async fn handle_status(state: &State, params: StatusParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let metric = params.metric.as_deref().unwrap_or("f1");
    let count = s.store.count().map_err(to_mcp_error)?;
    let best = s.store.get_best(metric).map_err(to_mcp_error)?;

    let mut targets_json = serde_json::Map::new();
    if let Some(cfg) = &s.config {
        for (m, target) in &cfg.experiment.targets {
            let current = s
                .store
                .get_best(m)
                .ok()
                .flatten()
                .and_then(|r| r.metrics.get(m).copied())
                .unwrap_or(0.0);
            targets_json.insert(
                m.clone(),
                serde_json::json!({
                    "current": current,
                    "target": target,
                    "gap": (target - current).max(0.0),
                    "met": current >= *target,
                }),
            );
        }
    }

    let result = serde_json::json!({
        "experiment_count": count,
        "targets": targets_json,
        "best_config": best.as_ref().map(|b| &b.config.parameters),
        "best_metrics": best.as_ref().map(|b| &b.metrics),
        "budget": {
            "spent": s.budget.spent,
            "limit": s.budget.limit,
            "remaining": s.budget.remaining(),
        },
    });
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_history`.
pub async fn handle_history(state: &State, params: HistoryParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let last = params.last.unwrap_or(10).min(100);
    let recent = s.store.get_recent(last).map_err(to_mcp_error)?;

    let experiments: Vec<serde_json::Value> = recent
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "timestamp": r.timestamp.to_rfc3339(),
                "config": r.config.parameters,
                "metrics": r.metrics,
                "duration_secs": r.duration_secs,
                "status": format!("{:?}", r.status),
                "cost_usd": r.cost_usd,
            })
        })
        .collect();

    let result = serde_json::json!({
        "count": experiments.len(),
        "experiments": experiments,
    });
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_run`.
pub async fn handle_run(state: &State, params: RunParams) -> Result<String, ErrorData> {
    let (config, mobius_dir) = {
        let s = state.read().await;
        (s.config.clone(), s.mobius_dir.clone())
    };
    let cfg =
        config.ok_or_else(|| invalid_params("No mobius.toml found. Run 'mobius init' first."))?;

    let overrides = parse_overrides(&params.config_overrides)?;
    let merged = merge_params(&cfg.experiment.sweep_space, &overrides);
    let command = params
        .command
        .or_else(|| cfg.experiment.command.clone())
        .unwrap_or_else(|| "echo '{\"f1\": 0.0}'".into());
    let timeout = params.timeout_secs.unwrap_or(cfg.experiment.timeout_secs);
    let env = mobius_core::compute::config_to_env(&merged, &cfg.experiment.env_map);

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ExperimentResult> {
        let backend = SubprocessBackend;
        let output = backend.submit(&command, &env, timeout)?;
        let parsed = OutputParser::parse(&output.stdout, &output.stderr);

        let mut metrics = parsed.metrics;
        if let Some(bs) = parsed.bench_score {
            metrics.insert("bench_score".into(), bs);
        }

        let store_path = mobius_dir.join("history.jsonl");
        let mut store = JsonlStore::new(&store_path)?;
        let count = store.count()? + 1;

        let result = ExperimentResult {
            id: format!("exp-{count:04}"),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: merged,
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: output.duration_secs,
            cost_usd: Some(cfg.experiment.cost_per_run),
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
        store.append(&result)?;
        Ok(result)
    })
    .await
    .map_err(to_mcp_error)?
    .map_err(to_mcp_error)?;

    // Record spend
    {
        let mut s = state.write().await;
        let _ = s.budget.record_spend(result.cost_usd.unwrap_or(0.0));
    }

    let out = serde_json::json!({
        "id": result.id,
        "status": format!("{:?}", result.status),
        "metrics": result.metrics,
        "duration_secs": result.duration_secs,
        "cost_usd": result.cost_usd,
        "config": result.config.parameters,
    });
    serde_json::to_string_pretty(&out).map_err(to_mcp_error)
}

/// Handler for `mobius_evaluate`.
pub async fn handle_evaluate(state: &State, params: EvaluateParams) -> Result<String, ErrorData> {
    let match_window = {
        let s = state.read().await;
        params.match_window.unwrap_or_else(|| {
            s.config
                .as_ref()
                .map(|c| c.bench.match_window)
                .unwrap_or(6.0)
        })
    };
    let dimensions = {
        let s = state.read().await;
        s.config
            .as_ref()
            .map(|c| c.bench.dimensions.clone())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(default_dimensions)
    };

    let pred_path = params.predictions_path;
    let gt_path = params.ground_truth_path;

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let predictions = loader::load_predictions(Path::new(&pred_path))?;
        let ground_truth = loader::load_ground_truth(Path::new(&gt_path))?;

        let mut evaluator = Evaluator::new(Box::new(GreedyTimestampMatcher), match_window);
        for dim in &dimensions {
            evaluator.add_dimension(dim.clone(), build_scorer(&dim.scorer));
        }
        let bench = evaluator.evaluate(&predictions, &ground_truth);

        Ok(serde_json::json!({
            "bench_score": bench.bench_score,
            "grade": format!("{:?}", bench.grade),
            "dimensions": bench.dimensions,
            "match_count": bench.match_count,
            "false_positive_count": bench.false_positive_count,
            "missed_count": bench.missed_count,
        }))
    })
    .await
    .map_err(to_mcp_error)?
    .map_err(to_mcp_error)?;

    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_suggest`.
pub async fn handle_suggest(state: &State, params: SuggestParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let cfg = s
        .config
        .as_ref()
        .ok_or_else(|| invalid_params("No mobius.toml found. Run 'mobius init' first."))?;

    let metric = params.metric.as_deref().unwrap_or("f1");
    let history = s.store.load_all().map_err(to_mcp_error)?;

    let strategy_name = params
        .strategy
        .as_deref()
        .or_else(|| cfg.agent.strategies.first().map(|s| s.as_str()))
        .unwrap_or("gradient_guided");
    let strategy = build_strategy(
        strategy_name,
        cfg.agent.plateau_window,
        cfg.agent.plateau_threshold,
    )
    .map_err(to_mcp_error)?;

    let production_config = build_production_config(&cfg.experiment.sweep_space);

    let ctx = StrategyContext {
        history: &history,
        production_config: &production_config,
        sweep_space: &cfg.experiment.sweep_space,
        targets: &cfg.experiment.targets,
        primary_metric: metric,
        learning_store: &s.learning_store,
    };

    let suggestion = strategy.suggest(&ctx).map_err(to_mcp_error)?;

    let result = serde_json::json!({
        "rationale": suggestion.rationale,
        "changed_params": suggestion.changed_params,
        "full_config": suggestion.config.parameters,
    });
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_sweep`.
pub async fn handle_sweep(state: &State, params: SweepParams) -> Result<String, ErrorData> {
    let (mobius_dir, cfg_command, cfg_env_map, cfg_timeout) = {
        let s = state.read().await;
        let cmd = s.config.as_ref().and_then(|c| c.experiment.command.clone());
        let env = s
            .config
            .as_ref()
            .map(|c| c.experiment.env_map.clone())
            .unwrap_or_default();
        let t = s
            .config
            .as_ref()
            .map(|c| c.experiment.timeout_secs)
            .unwrap_or(600);
        (s.mobius_dir.clone(), cmd, env, t)
    };

    let spec: HashMap<String, Vec<serde_json::Value>> = serde_json::from_value(params.spec)
        .map_err(|e| invalid_params(format!("Invalid sweep spec: {e}")))?;
    let command = params
        .command
        .or(cfg_command)
        .unwrap_or_else(|| "echo '{\"f1\": 0.0}'".into());
    let timeout = params.timeout_secs.unwrap_or(cfg_timeout);

    let results = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<serde_json::Value>> {
        let param_names: Vec<String> = spec.keys().cloned().collect();
        let param_values: Vec<&Vec<serde_json::Value>> =
            param_names.iter().map(|k| &spec[k]).collect();
        let combos = cartesian_product(&param_values);

        let mut store = JsonlStore::new(mobius_dir.join("history.jsonl"))?;
        let backend = SubprocessBackend;
        let mut all = Vec::new();

        for combo in &combos {
            let mut params = HashMap::new();
            for (j, name) in param_names.iter().enumerate() {
                params.insert(name.clone(), combo[j].clone());
            }
            let env = mobius_core::compute::config_to_env(&params, &cfg_env_map);
            let output = backend.submit(&command, &env, timeout)?;
            let parsed = OutputParser::parse(&output.stdout, &output.stderr);
            let count = store.count()? + 1;

            let result = ExperimentResult {
                id: format!("sweep-{count:04}"),
                timestamp: chrono::Utc::now(),
                config: ExperimentConfig {
                    parameters: params.clone(),
                    metadata: HashMap::new(),
                },
                metrics: parsed.metrics.clone(),
                per_segment: HashMap::new(),
                duration_secs: output.duration_secs,
                cost_usd: None,
                status: ExperimentStatus::Success,
                error: None,
            };
            store.append(&result)?;
            all.push(serde_json::json!({
                "id": result.id,
                "config": params,
                "metrics": parsed.metrics,
            }));
        }

        // Sort by f1 descending
        all.sort_by(|a, b| {
            let fa = a["metrics"]["f1"].as_f64().unwrap_or(0.0);
            let fb = b["metrics"]["f1"].as_f64().unwrap_or(0.0);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, entry) in all.iter_mut().enumerate() {
            entry["rank"] = serde_json::json!(i + 1);
        }
        Ok(all)
    })
    .await
    .map_err(to_mcp_error)?
    .map_err(to_mcp_error)?;

    let out = serde_json::json!({
        "total_configs": results.len(),
        "results": results,
        "best": results.first(),
    });
    serde_json::to_string_pretty(&out).map_err(to_mcp_error)
}

/// Handler for `mobius_agent`.
pub async fn handle_agent(state: &State, params: AgentParams) -> Result<String, ErrorData> {
    let (cfg, mobius_dir) = {
        let s = state.read().await;
        let cfg = s
            .config
            .clone()
            .ok_or_else(|| invalid_params("No mobius.toml found. Run 'mobius init' first."))?;
        (cfg, s.mobius_dir.clone())
    };

    let budget_limit = params.budget.unwrap_or(cfg.experiment.budget_usd);
    let max_iterations = params.max_iterations.unwrap_or(50);
    let metric = params.metric.unwrap_or_else(|| "f1".into());
    let production_config = build_production_config(&cfg.experiment.sweep_space);

    let report = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let store = JsonlStore::new(mobius_dir.join("history.jsonl"))?;
        let learning_store = LearningStore::new(mobius_dir.join("learnings.jsonl"))?;
        let budget =
            BudgetGuard::new(budget_limit).with_state_file(mobius_dir.join("budget.json"))?;

        let agent_config = AgentConfig {
            max_iterations,
            targets: cfg.experiment.targets.clone(),
            cost_per_run: cfg.experiment.cost_per_run,
            primary_metric: metric,
            command_template: cfg
                .experiment
                .command
                .clone()
                .unwrap_or_else(|| "echo '{\"f1\": 0.0}'".into()),
            env_map: cfg.experiment.env_map.clone(),
            production_config,
            sweep_space: cfg.experiment.sweep_space.clone(),
            timeout_secs: cfg.experiment.timeout_secs,
            strategies: cfg.agent.strategies.clone(),
            plateau_window: cfg.agent.plateau_window,
            plateau_threshold: cfg.agent.plateau_threshold,
        };

        let strategy_name = params
            .strategy
            .as_deref()
            .or_else(|| cfg.agent.strategies.first().map(|s| s.as_str()))
            .unwrap_or("gradient_guided");
        let strategy = build_strategy(
            strategy_name,
            cfg.agent.plateau_window,
            cfg.agent.plateau_threshold,
        )?;

        let mut agent = AgentLoop::new(
            agent_config,
            strategy,
            Box::new(SubprocessBackend),
            Box::new(store),
            learning_store,
            budget,
        );
        agent.add_pre_hook(Box::new(BudgetCheckHook));
        agent.add_post_hook(Box::new(RegressionDetectionHook::new(0.05)));
        agent.add_post_hook(Box::new(OverfittingDetectionHook::default()));
        if params.pruning.unwrap_or(false) {
            agent.set_pruner(mobius_claw::pruning::AshaPruner::new(3, 3));
        }

        let report = agent.run()?;

        Ok(serde_json::json!({
            "iterations": report.iterations,
            "stop_reason": format!("{:?}", report.stop_reason),
            "pruned_count": report.pruned_count,
            "best_result": report.best_result.as_ref().map(|b| serde_json::json!({
                "id": b.id,
                "metrics": b.metrics,
                "config": b.config.parameters,
            })),
            "total_cost": report.total_cost,
        }))
    })
    .await
    .map_err(to_mcp_error)?
    .map_err(to_mcp_error)?;

    // Reload budget from disk after agent loop
    {
        let mut s = state.write().await;
        let budget_path = s.mobius_dir.join("budget.json");
        if budget_path.exists()
            && let Ok(b) = BudgetGuard::new(0.0).with_state_file(&budget_path)
        {
            s.budget = b;
        }
    }

    serde_json::to_string_pretty(&report).map_err(to_mcp_error)
}

/// Handler for `mobius_pareto`.
pub async fn handle_pareto(state: &State, params: ParetoParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let all = s.store.load_all().map_err(to_mcp_error)?;
    let front = pareto::pareto_front(&all, &params.x_metric, &params.y_metric);

    let experiments: Vec<serde_json::Value> = front
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                params.x_metric.clone(): r.metrics.get(&params.x_metric),
                params.y_metric.clone(): r.metrics.get(&params.y_metric),
                "config": r.config.parameters,
            })
        })
        .collect();

    let result = serde_json::json!({
        "front_size": experiments.len(),
        "experiments": experiments,
    });
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_budget`.
pub async fn handle_budget(state: &State, params: BudgetParams) -> Result<String, ErrorData> {
    let cost_per_run;
    if params.set_limit.is_some() || params.record_spend.is_some() {
        let mut s = state.write().await;
        if let Some(limit) = params.set_limit {
            s.budget.limit = limit;
        }
        if let Some(spend) = params.record_spend {
            s.budget.record_spend(spend).map_err(to_mcp_error)?;
        }
        cost_per_run = s
            .config
            .as_ref()
            .map(|c| c.experiment.cost_per_run)
            .unwrap_or(1.0);
        let result = budget_json(&s.budget, cost_per_run);
        return serde_json::to_string_pretty(&result).map_err(to_mcp_error);
    }

    let s = state.read().await;
    cost_per_run = s
        .config
        .as_ref()
        .map(|c| c.experiment.cost_per_run)
        .unwrap_or(1.0);
    let result = budget_json(&s.budget, cost_per_run);
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_learnings`.
pub async fn handle_learnings(state: &State, params: LearningsParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let metric = params.metric.as_deref().unwrap_or("f1");

    let gradients = if let Some(param) = &params.param {
        let g = s
            .learning_store
            .get_param_gradient(param, metric)
            .map_err(to_mcp_error)?;
        vec![g]
    } else if let Some(cfg) = &s.config {
        let mut gs = Vec::new();
        for param in cfg.experiment.sweep_space.keys() {
            let g = s
                .learning_store
                .get_param_gradient(param, metric)
                .map_err(to_mcp_error)?;
            gs.push(g);
        }
        gs
    } else {
        Vec::new()
    };

    let history = s.store.load_all().map_err(to_mcp_error)?;
    let sweep_space = s.config.as_ref().map(|c| &c.experiment.sweep_space);
    let untried = sweep_space
        .map(|ss| s.learning_store.get_untried_dimensions(&history, ss))
        .unwrap_or_default();

    let result = serde_json::json!({
        "gradients": gradients,
        "untried_dimensions": untried,
    });
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn budget_json(budget: &BudgetGuard, cost_per_run: f64) -> serde_json::Value {
    serde_json::json!({
        "spent": budget.spent,
        "limit": budget.limit,
        "remaining": budget.remaining(),
        "experiments_remaining": budget.experiments_remaining(cost_per_run),
        "cost_per_run": cost_per_run,
    })
}

fn parse_overrides(
    value: &Option<serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>, ErrorData> {
    match value {
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| invalid_params(format!("Invalid config_overrides: {e}"))),
        None => Ok(HashMap::new()),
    }
}

fn merge_params(
    sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    overrides: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut params = HashMap::new();
    for (k, values) in sweep_space {
        if let Some(first) = values.first() {
            params.insert(k.clone(), first.clone());
        }
    }
    for (k, v) in overrides {
        params.insert(k.clone(), v.clone());
    }
    params
}

fn build_production_config(
    sweep_space: &HashMap<String, Vec<serde_json::Value>>,
) -> HashMap<String, serde_json::Value> {
    sweep_space
        .iter()
        .filter_map(|(k, v)| v.first().map(|f| (k.clone(), f.clone())))
        .collect()
}

fn default_dimensions() -> Vec<DimensionConfig> {
    vec![
        DimensionConfig {
            name: "Detection".into(),
            weight: 0.50,
            scorer: "f1".into(),
            options: Default::default(),
        },
        DimensionConfig {
            name: "Classification".into(),
            weight: 0.30,
            scorer: "accuracy".into(),
            options: Default::default(),
        },
        DimensionConfig {
            name: "Timestamp".into(),
            weight: 0.20,
            scorer: "timestamp_mae".into(),
            options: Default::default(),
        },
    ]
}

fn build_scorer(name: &str) -> Box<dyn DimensionScorer> {
    match name {
        "f1" => Box::new(F1Scorer),
        "accuracy" | "classification_accuracy" => {
            Box::new(ClassificationAccuracyScorer::on_primary_label())
        }
        "timestamp_mae" => Box::new(TimestampMaeScorer),
        _ => Box::new(F1Scorer),
    }
}

fn cartesian_product(lists: &[&Vec<serde_json::Value>]) -> Vec<Vec<serde_json::Value>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let rest = cartesian_product(&lists[1..]);
    for item in lists[0] {
        for combo in &rest {
            let mut new_combo = vec![item.clone()];
            new_combo.extend(combo.iter().cloned());
            result.push(new_combo);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SharedState;
    use mobius_claw::learning::Learning;
    use mobius_claw::learning::ParamDelta;
    use mobius_core::config::{
        AgentSection, BenchSection, ComputeSection, ExperimentSection, MobiusConfig, ProjectConfig,
    };
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use tempfile::TempDir;

    fn test_config() -> MobiusConfig {
        let mut targets = HashMap::new();
        targets.insert("f1".into(), 0.85);

        let mut sweep_space = HashMap::new();
        sweep_space.insert(
            "learning_rate".into(),
            vec![
                serde_json::json!(0.001),
                serde_json::json!(0.01),
                serde_json::json!(0.1),
            ],
        );
        sweep_space.insert(
            "batch_size".into(),
            vec![serde_json::json!(16), serde_json::json!(32)],
        );

        MobiusConfig {
            project: ProjectConfig {
                name: "test".into(),
                version: "0.1.0".into(),
            },
            experiment: ExperimentSection {
                budget_usd: 20.0,
                cost_per_run: 1.0,
                targets,
                sweep_space,
                ..Default::default()
            },
            bench: BenchSection::default(),
            compute: ComputeSection::default(),
            agent: AgentSection::default(),
        }
    }

    fn test_state(dir: &TempDir) -> State {
        SharedState::with_dir(dir.path().to_path_buf(), Some(test_config())).unwrap()
    }

    fn test_state_no_config(dir: &TempDir) -> State {
        SharedState::with_dir(dir.path().to_path_buf(), None).unwrap()
    }

    fn make_result(id: &str, f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        metrics.insert("precision".into(), f1 + 0.05);
        ExperimentResult {
            id: id.to_string(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("learning_rate".into(), serde_json::json!(0.01));
                    p.insert("batch_size".into(), serde_json::json!(32));
                    p
                },
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 10.0,
            cost_usd: Some(1.0),
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    async fn seed_history(state: &State, results: &[ExperimentResult]) {
        let mut s = state.write().await;
        for r in results {
            s.store.append(r).unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // handle_status tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_status_empty() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let result = handle_status(&state, StatusParams { metric: None })
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["experiment_count"], 0);
        assert!(v["best_config"].is_null());
    }

    #[tokio::test]
    async fn test_handle_status_with_experiments() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        seed_history(
            &state,
            &[
                make_result("exp-1", 0.60),
                make_result("exp-2", 0.75),
                make_result("exp-3", 0.70),
            ],
        )
        .await;

        let result = handle_status(&state, StatusParams { metric: None })
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["experiment_count"], 3);
        assert!(v["best_metrics"]["f1"].as_f64().unwrap() > 0.74);
    }

    #[tokio::test]
    async fn test_handle_status_no_config() {
        let dir = TempDir::new().unwrap();
        let state = test_state_no_config(&dir);
        let result = handle_status(&state, StatusParams { metric: None })
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["targets"], serde_json::json!({}));
    }

    // -----------------------------------------------------------------------
    // handle_history tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_history_default() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let results: Vec<_> = (0..15)
            .map(|i| make_result(&format!("exp-{i}"), 0.5 + i as f64 * 0.01))
            .collect();
        seed_history(&state, &results).await;

        let result = handle_history(&state, HistoryParams { last: None })
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["count"], 10);
    }

    #[tokio::test]
    async fn test_handle_history_custom_limit() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let results: Vec<_> = (0..15)
            .map(|i| make_result(&format!("exp-{i}"), 0.5))
            .collect();
        seed_history(&state, &results).await;

        let result = handle_history(&state, HistoryParams { last: Some(5) })
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["count"], 5);
    }

    // -----------------------------------------------------------------------
    // handle_suggest tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_suggest_insufficient() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        seed_history(&state, &[make_result("exp-1", 0.5)]).await;

        let result = handle_suggest(
            &state,
            SuggestParams {
                metric: None,
                strategy: None,
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(v["rationale"].as_str().unwrap().contains("Insufficient"));
    }

    #[tokio::test]
    async fn test_handle_suggest_with_history() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);

        let mut r1 = make_result("exp-1", 0.50);
        r1.config
            .parameters
            .insert("learning_rate".into(), serde_json::json!(0.001));
        let mut r2 = make_result("exp-2", 0.65);
        r2.config
            .parameters
            .insert("learning_rate".into(), serde_json::json!(0.01));
        let mut r3 = make_result("exp-3", 0.70);
        r3.config
            .parameters
            .insert("learning_rate".into(), serde_json::json!(0.1));
        seed_history(&state, &[r1, r2, r3]).await;

        let result = handle_suggest(
            &state,
            SuggestParams {
                metric: None,
                strategy: None,
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(!v["full_config"].as_object().unwrap().is_empty());
        assert!(!v["rationale"].as_str().unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // handle_pareto tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_pareto_empty() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let result = handle_pareto(
            &state,
            ParetoParams {
                x_metric: "f1".into(),
                y_metric: "precision".into(),
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["front_size"], 0);
    }

    #[tokio::test]
    async fn test_handle_pareto_with_data() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        seed_history(
            &state,
            &[
                make_result("exp-1", 0.80),
                make_result("exp-2", 0.70),
                make_result("exp-3", 0.90),
            ],
        )
        .await;

        let result = handle_pareto(
            &state,
            ParetoParams {
                x_metric: "f1".into(),
                y_metric: "precision".into(),
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(v["front_size"].as_u64().unwrap() >= 1);
    }

    // -----------------------------------------------------------------------
    // handle_budget tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_budget_read() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let result = handle_budget(
            &state,
            BudgetParams {
                set_limit: None,
                record_spend: None,
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["limit"], 20.0);
        assert_eq!(v["spent"], 0.0);
    }

    #[tokio::test]
    async fn test_handle_budget_set_limit() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let result = handle_budget(
            &state,
            BudgetParams {
                set_limit: Some(50.0),
                record_spend: None,
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["limit"], 50.0);
    }

    #[tokio::test]
    async fn test_handle_budget_record_spend() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let result = handle_budget(
            &state,
            BudgetParams {
                set_limit: None,
                record_spend: Some(5.0),
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["spent"], 5.0);
        assert_eq!(v["remaining"], 15.0);
    }

    // -----------------------------------------------------------------------
    // handle_learnings tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_learnings_empty() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let result = handle_learnings(
            &state,
            LearningsParams {
                param: None,
                metric: None,
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(!v["gradients"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_learnings_with_data() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);

        // Seed a learning entry
        {
            let mut s = state.write().await;
            let learning = Learning {
                changed_params: vec![ParamDelta {
                    param: "learning_rate".into(),
                    old_value: serde_json::json!(0.001),
                    new_value: serde_json::json!(0.01),
                }],
                metric_deltas: {
                    let mut m = HashMap::new();
                    m.insert("f1".into(), 0.1);
                    m
                },
                timestamp: chrono::Utc::now(),
            };
            s.learning_store.append(&learning).unwrap();
        }

        let result = handle_learnings(
            &state,
            LearningsParams {
                param: Some("learning_rate".into()),
                metric: None,
            },
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["gradients"].as_array().unwrap().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cartesian_product_empty() {
        let result = cartesian_product(&[]);
        assert_eq!(result, vec![Vec::<serde_json::Value>::new()]);
    }

    #[test]
    fn test_cartesian_product_single() {
        let vals = vec![
            serde_json::json!(1),
            serde_json::json!(2),
            serde_json::json!(3),
        ];
        let result = cartesian_product(&[&vals]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cartesian_product_multi() {
        let a = vec![serde_json::json!(1), serde_json::json!(2)];
        let b = vec![
            serde_json::json!("x"),
            serde_json::json!("y"),
            serde_json::json!("z"),
        ];
        let result = cartesian_product(&[&a, &b]);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_merge_params_defaults_and_overrides() {
        let mut sweep = HashMap::new();
        sweep.insert(
            "lr".into(),
            vec![serde_json::json!(0.01), serde_json::json!(0.1)],
        );
        sweep.insert("bs".into(), vec![serde_json::json!(16)]);

        let mut overrides = HashMap::new();
        overrides.insert("lr".into(), serde_json::json!(0.5));

        let merged = merge_params(&sweep, &overrides);
        assert_eq!(merged["lr"], serde_json::json!(0.5));
        assert_eq!(merged["bs"], serde_json::json!(16));
    }

    #[test]
    fn test_parse_overrides_valid() {
        let v = Some(serde_json::json!({"a": 1}));
        let result = parse_overrides(&v).unwrap();
        assert_eq!(result["a"], serde_json::json!(1));
    }

    #[test]
    fn test_parse_overrides_invalid() {
        let v = Some(serde_json::json!("not an object"));
        assert!(parse_overrides(&v).is_err());
    }

    #[test]
    fn test_parse_overrides_none() {
        let result = parse_overrides(&None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_production_config() {
        let mut sweep = HashMap::new();
        sweep.insert("a".into(), vec![serde_json::json!(1), serde_json::json!(2)]);
        sweep.insert("b".into(), vec![serde_json::json!("x")]);
        let config = build_production_config(&sweep);
        assert_eq!(config["a"], serde_json::json!(1));
        assert_eq!(config["b"], serde_json::json!("x"));
    }

    #[test]
    fn test_default_dimensions_weights_sum() {
        let dims = default_dimensions();
        assert_eq!(dims.len(), 3);
        let total: f64 = dims.iter().map(|d| d.weight).sum();
        assert!((total - 1.0).abs() < 0.001);
    }
}
