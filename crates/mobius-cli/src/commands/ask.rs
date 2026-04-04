//! Ask-and-tell: ask for a parameter suggestion.
//!
//! Prints ONLY valid JSON to stdout for machine-parseable output.
//! No decorative output, banners, or color codes.

use mobius_claw::learning_store::LearningStore;
use mobius_claw::strategy::{StrategyContext, build_strategy};
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::ExperimentStore;
use mobius_core::store::{JsonlStore, SqliteStore};
use std::collections::HashMap;

/// Build the appropriate experiment store based on the backend name.
fn build_store(store_backend: &str) -> anyhow::Result<Box<dyn ExperimentStore>> {
    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    match store_backend {
        "sqlite" => Ok(Box::new(SqliteStore::new(mobius_dir.join("history.db"))?)),
        _ => Ok(Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?)),
    }
}

/// Run the ask command, printing a JSON suggestion to stdout.
pub fn run(strategy_name: &str, metric: &str, store_backend: &str) -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").map_err(|e| {
        anyhow::anyhow!(
            "Failed to load mobius.toml: {}. Run 'mobius init' first.",
            e
        )
    })?;

    let store = build_store(store_backend)?;
    let history = store.load_all()?;

    let learning_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("learnings.jsonl");
    let learning_store = LearningStore::new(&learning_path)?;

    let strategy = build_strategy(
        strategy_name,
        config.agent.plateau_window,
        config.agent.plateau_threshold,
    )?;

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
        primary_metric: metric,
        learning_store: &learning_store,
    };

    let suggestion = strategy.suggest(&ctx)?;

    // Output ONLY valid JSON -- no decorative text
    let output = serde_json::json!({
        "config": suggestion.config.parameters,
        "rationale": suggestion.rationale,
    });
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_store_jsonl() {
        let store = build_store("jsonl");
        assert!(store.is_ok());
    }

    #[test]
    fn test_build_store_sqlite() {
        let store = build_store("sqlite");
        assert!(store.is_ok());
    }
}
