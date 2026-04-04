//! Tool parameter structs and handler helper functions.
//!
//! Each MCP tool has a parameter struct (with `schemars::JsonSchema` for
//! automatic schema generation) and a handler function that operates on
//! [`SharedState`](crate::state::SharedState).

use crate::error::{invalid_params, to_mcp_error};
use crate::state::State;
use mobius_bench::evaluator::Evaluator;
use mobius_bench::matcher::GreedyTimestampMatcher;
use mobius_bench::scorers::{ClassificationAccuracyScorer, DimensionScorer, F1Scorer, TimestampMaeScorer};
use mobius_claw::agent::{AgentConfig, AgentLoop};
use mobius_claw::hooks::{BudgetCheckHook, OverfittingDetectionHook, RegressionDetectionHook};
use mobius_claw::learning_store::LearningStore;
use mobius_claw::strategy::{GradientGuidedTuning, Strategy, StrategyContext};
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
            let current = s.store.get_best(m).ok()
                .flatten()
                .and_then(|r| r.metrics.get(m).copied())
                .unwrap_or(0.0);
            targets_json.insert(m.clone(), serde_json::json!({
                "current": current,
                "target": target,
                "gap": (target - current).max(0.0),
                "met": current >= *target,
            }));
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
        .map(|r| serde_json::json!({
            "id": r.id,
            "timestamp": r.timestamp.to_rfc3339(),
            "config": r.config.parameters,
            "metrics": r.metrics,
            "duration_secs": r.duration_secs,
            "status": format!("{:?}", r.status),
            "cost_usd": r.cost_usd,
        }))
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
    let cfg = config.ok_or_else(|| invalid_params("No mobius.toml found. Run 'mobius init' first."))?;

    let overrides = parse_overrides(&params.config_overrides)?;
    let merged = merge_params(&cfg.experiment.sweep_space, &overrides);
    let command = params.command.unwrap_or_else(|| "echo '{\"f1\": 0.0}'".into());
    let timeout = params.timeout_secs.unwrap_or(600);

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ExperimentResult> {
        let backend = SubprocessBackend;
        let output = backend.submit(&command, &HashMap::new(), timeout)?;
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
            config: ExperimentConfig { parameters: merged, metadata: HashMap::new() },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: output.duration_secs,
            cost_usd: Some(cfg.experiment.cost_per_run),
            status: if output.exit_code == 0 { ExperimentStatus::Success } else { ExperimentStatus::Error },
            error: if output.exit_code != 0 { Some(output.stderr.chars().take(200).collect()) } else { None },
        };
        store.append(&result)?;
        Ok(result)
    }).await.map_err(to_mcp_error)?.map_err(to_mcp_error)?;

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
            s.config.as_ref().map(|c| c.bench.match_window).unwrap_or(6.0)
        })
    };
    let dimensions = {
        let s = state.read().await;
        s.config.as_ref()
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
    }).await.map_err(to_mcp_error)?.map_err(to_mcp_error)?;

    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_suggest`.
pub async fn handle_suggest(state: &State, params: SuggestParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let cfg = s.config.as_ref()
        .ok_or_else(|| invalid_params("No mobius.toml found. Run 'mobius init' first."))?;

    let metric = params.metric.as_deref().unwrap_or("f1");
    let history = s.store.load_all().map_err(to_mcp_error)?;

    let strategy = GradientGuidedTuning::new(
        cfg.agent.plateau_window,
        cfg.agent.plateau_threshold,
    );

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
    let mobius_dir = {
        let s = state.read().await;
        s.mobius_dir.clone()
    };

    let spec: HashMap<String, Vec<serde_json::Value>> = serde_json::from_value(params.spec)
        .map_err(|e| invalid_params(format!("Invalid sweep spec: {e}")))?;
    let command = params.command.unwrap_or_else(|| "echo '{\"f1\": 0.0}'".into());
    let timeout = params.timeout_secs.unwrap_or(600);

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
            let output = backend.submit(&command, &HashMap::new(), timeout)?;
            let parsed = OutputParser::parse(&output.stdout, &output.stderr);
            let count = store.count()? + 1;

            let result = ExperimentResult {
                id: format!("sweep-{count:04}"),
                timestamp: chrono::Utc::now(),
                config: ExperimentConfig { parameters: params.clone(), metadata: HashMap::new() },
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
    }).await.map_err(to_mcp_error)?.map_err(to_mcp_error)?;

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
        let cfg = s.config.clone()
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
        let budget = BudgetGuard::new(budget_limit)
            .with_state_file(mobius_dir.join("budget.json"))?;

        let agent_config = AgentConfig {
            max_iterations,
            targets: cfg.experiment.targets.clone(),
            cost_per_run: cfg.experiment.cost_per_run,
            primary_metric: metric,
            command_template: "echo '{\"f1\": 0.0}'".into(),
            env_map: HashMap::new(),
            production_config,
            sweep_space: cfg.experiment.sweep_space.clone(),
            timeout_secs: 600,
        };

        let strategy = GradientGuidedTuning::new(
            cfg.agent.plateau_window,
            cfg.agent.plateau_threshold,
        );

        let mut agent = AgentLoop::new(
            agent_config,
            Box::new(strategy),
            Box::new(SubprocessBackend),
            Box::new(store),
            learning_store,
            budget,
        );
        agent.add_pre_hook(Box::new(BudgetCheckHook));
        agent.add_post_hook(Box::new(RegressionDetectionHook::new(0.05)));
        agent.add_post_hook(Box::new(OverfittingDetectionHook::default()));

        let report = agent.run()?;

        Ok(serde_json::json!({
            "iterations": report.iterations,
            "stop_reason": format!("{:?}", report.stop_reason),
            "best_result": report.best_result.as_ref().map(|b| serde_json::json!({
                "id": b.id,
                "metrics": b.metrics,
                "config": b.config.parameters,
            })),
            "total_cost": report.total_cost,
        }))
    }).await.map_err(to_mcp_error)?.map_err(to_mcp_error)?;

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
        .map(|r| serde_json::json!({
            "id": r.id,
            params.x_metric.clone(): r.metrics.get(&params.x_metric),
            params.y_metric.clone(): r.metrics.get(&params.y_metric),
            "config": r.config.parameters,
        }))
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
        cost_per_run = s.config.as_ref().map(|c| c.experiment.cost_per_run).unwrap_or(1.0);
        let result = budget_json(&s.budget, cost_per_run);
        return serde_json::to_string_pretty(&result).map_err(to_mcp_error);
    }

    let s = state.read().await;
    cost_per_run = s.config.as_ref().map(|c| c.experiment.cost_per_run).unwrap_or(1.0);
    let result = budget_json(&s.budget, cost_per_run);
    serde_json::to_string_pretty(&result).map_err(to_mcp_error)
}

/// Handler for `mobius_learnings`.
pub async fn handle_learnings(state: &State, params: LearningsParams) -> Result<String, ErrorData> {
    let s = state.read().await;
    let metric = params.metric.as_deref().unwrap_or("f1");

    let gradients = if let Some(param) = &params.param {
        let g = s.learning_store.get_param_gradient(param, metric).map_err(to_mcp_error)?;
        vec![g]
    } else if let Some(cfg) = &s.config {
        let mut gs = Vec::new();
        for param in cfg.experiment.sweep_space.keys() {
            let g = s.learning_store.get_param_gradient(param, metric).map_err(to_mcp_error)?;
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
