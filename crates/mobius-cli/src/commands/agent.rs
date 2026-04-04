use mobius_claw::StopReason;
use mobius_claw::agent::{AgentConfig, AgentLoop};
use mobius_claw::hooks::{BudgetCheckHook, OverfittingDetectionHook, RegressionDetectionHook};
use mobius_claw::learning_store::LearningStore;
use mobius_claw::pruning::AshaPruner;
use mobius_claw::strategy::build_strategy;
use mobius_core::budget::BudgetGuard;
use mobius_core::compute::SubprocessBackend;
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::ExperimentStore;
use mobius_core::store::{JsonlStore, SqliteStore};
use std::collections::HashMap;

pub fn run(
    budget_limit: f64,
    strategy_name: &str,
    pruning: bool,
    store_backend: &str,
) -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").map_err(|e| {
        anyhow::anyhow!(
            "Failed to load mobius.toml: {}. Run 'mobius init' first.",
            e
        )
    })?;

    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    let learning_path = mobius_dir.join("learnings.jsonl");
    let budget_path = mobius_dir.join("budget.json");

    let experiment_store: Box<dyn ExperimentStore> = match store_backend {
        "sqlite" => Box::new(SqliteStore::new(mobius_dir.join("history.db"))?),
        _ => Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?),
    };

    let budget = BudgetGuard::new(budget_limit).with_state_file(&budget_path)?;

    // Build production config
    let mut production_config: HashMap<String, serde_json::Value> = HashMap::new();
    for (k, values) in &config.experiment.sweep_space {
        if let Some(first) = values.first() {
            production_config.insert(k.clone(), first.clone());
        }
    }

    let agent_config = AgentConfig {
        max_iterations: 50,
        targets: config.experiment.targets.clone(),
        cost_per_run: config.experiment.cost_per_run,
        primary_metric: "f1".into(),
        command_template: config
            .experiment
            .command
            .clone()
            .unwrap_or_else(|| "echo '{\"f1\": 0.0}'".into()),
        env_map: config.experiment.env_map.clone(),
        production_config,
        sweep_space: config.experiment.sweep_space.clone(),
        timeout_secs: config.experiment.timeout_secs,
        strategies: config.agent.strategies.clone(),
        plateau_window: config.agent.plateau_window,
        plateau_threshold: config.agent.plateau_threshold,
    };

    let strategy = build_strategy(
        strategy_name,
        config.agent.plateau_window,
        config.agent.plateau_threshold,
    )?;

    let mut agent = AgentLoop::new(
        agent_config,
        strategy,
        Box::new(SubprocessBackend),
        experiment_store,
        LearningStore::new(&learning_path)?,
        budget,
    );

    agent.add_pre_hook(Box::new(BudgetCheckHook));
    agent.add_post_hook(Box::new(RegressionDetectionHook::new(0.05)));
    agent.add_post_hook(Box::new(OverfittingDetectionHook::default()));
    if pruning {
        agent.set_pruner(AshaPruner::new(3, 3));
    }

    println!(
        "Starting autonomous agent loop (budget: ${:.2})...\n",
        budget_limit
    );

    let report = agent.run()?;

    println!("\n{:=<60}", "");
    println!("  AGENT REPORT");
    println!("{:=<60}", "");
    println!("  Iterations: {}", report.iterations);
    println!(
        "  Stop reason: {}",
        match report.stop_reason {
            StopReason::TargetsMet => "Targets met!",
            StopReason::BudgetExhausted => "Budget exhausted",
            StopReason::MaxIterationsReached => "Max iterations reached",
            StopReason::Plateau => "Plateau detected",
            StopReason::UserInterrupted => "User interrupted",
        }
    );

    if let Some(best) = &report.best_result {
        println!("  Best result: {}", best.id);
        for (k, v) in &best.metrics {
            println!("    {}: {:.4}", k, v);
        }
    }

    println!("{:=<60}\n", "");
    Ok(())
}
