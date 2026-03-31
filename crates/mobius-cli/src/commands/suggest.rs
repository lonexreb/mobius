use mobius_claw::learning_store::LearningStore;
use mobius_claw::strategy::{GradientGuidedTuning, Strategy, StrategyContext};
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::ExperimentStore;
use mobius_core::store::JsonlStore;
use std::collections::HashMap;

pub fn run() -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml")
        .map_err(|e| anyhow::anyhow!("Failed to load mobius.toml: {}. Run 'mobius init' first.", e))?;

    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");
    let learning_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("learnings.jsonl");

    let store = JsonlStore::new(&store_path)?;
    let learning_store = LearningStore::new(&learning_path)?;
    let history = store.load_all()?;

    let strategy = GradientGuidedTuning::new(
        config.agent.plateau_window,
        config.agent.plateau_threshold,
    );

    // Build production config from sweep_space first values
    let mut production_config: HashMap<String, serde_json::Value> = HashMap::new();
    for (k, values) in &config.experiment.sweep_space {
        if let Some(first) = values.first() {
            production_config.insert(k.clone(), first.clone());
        }
    }

    let ctx = StrategyContext {
        history: &history,
        production_config: &production_config,
        sweep_space: &config.experiment.sweep_space,
        targets: &config.experiment.targets,
        primary_metric: "f1",
        learning_store: &learning_store,
    };

    let suggestion = strategy.suggest(&ctx)?;

    println!("\n{:=<60}", "");
    println!("  SUGGESTION");
    println!("{:=<60}", "");
    println!("  Rationale: {}", suggestion.rationale);

    if !suggestion.changed_params.is_empty() {
        println!("\n  Changed parameters:");
        for (k, v) in &suggestion.changed_params {
            println!("    {}: {}", k, v);
        }
    }

    println!("\n  Full config:");
    let json = serde_json::to_string_pretty(&suggestion.config.parameters)?;
    println!("{}", json);
    println!();

    Ok(())
}
