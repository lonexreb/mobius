use mobius_claw::learning_store::LearningStore;
use mobius_claw::strategy::{StrategyContext, Suggestion, build_strategy};
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::ExperimentStore;
use mobius_core::store::JsonlStore;
use std::collections::HashMap;

/// Compute a suggestion from the strategy without printing.
pub fn suggest_data(
    store: &dyn ExperimentStore,
    learning_store: &LearningStore,
    config: &MobiusConfig,
) -> anyhow::Result<Suggestion> {
    let history = store.load_all()?;

    let strategy_name = config
        .agent
        .strategies
        .first()
        .map(|s| s.as_str())
        .unwrap_or("gradient_guided");
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
        primary_metric: "f1",
        learning_store,
    };

    strategy.suggest(&ctx)
}

pub fn run() -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").map_err(|e| {
        anyhow::anyhow!(
            "Failed to load mobius.toml: {}. Run 'mobius init' first.",
            e
        )
    })?;

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

    let suggestion = suggest_data(&store, &learning_store, &config)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::config::{
        AgentSection, BenchSection, ComputeSection, ExperimentSection, ProjectConfig,
    };
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use tempfile::TempDir;

    fn test_config() -> MobiusConfig {
        let mut sweep_space = HashMap::new();
        sweep_space.insert(
            "lr".into(),
            vec![
                serde_json::json!(0.001),
                serde_json::json!(0.01),
                serde_json::json!(0.1),
            ],
        );
        MobiusConfig {
            project: ProjectConfig {
                name: "test".into(),
                version: "0.1.0".into(),
            },
            experiment: ExperimentSection {
                budget_usd: 20.0,
                cost_per_run: 1.0,
                targets: HashMap::new(),
                sweep_space,
            },
            bench: BenchSection::default(),
            compute: ComputeSection::default(),
            agent: AgentSection::default(),
        }
    }

    fn make_result(id: &str, f1: f64, lr: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut params = HashMap::new();
        params.insert("lr".into(), serde_json::json!(lr));
        ExperimentResult {
            id: id.into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: params,
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

    #[test]
    fn test_suggest_returns_suggestion() {
        let dir = TempDir::new().unwrap();
        let mut store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        store.append(&make_result("exp-1", 0.50, 0.001)).unwrap();
        store.append(&make_result("exp-2", 0.65, 0.01)).unwrap();
        store.append(&make_result("exp-3", 0.70, 0.1)).unwrap();

        let ls = LearningStore::new(dir.path().join("learnings.jsonl")).unwrap();
        let cfg = test_config();

        let suggestion = suggest_data(&store, &ls, &cfg).unwrap();
        assert!(!suggestion.rationale.is_empty());
        assert!(!suggestion.config.parameters.is_empty());
    }

    #[test]
    fn test_suggest_insufficient_history() {
        let dir = TempDir::new().unwrap();
        let mut store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        store.append(&make_result("exp-1", 0.50, 0.01)).unwrap();

        let ls = LearningStore::new(dir.path().join("learnings.jsonl")).unwrap();
        let cfg = test_config();

        let suggestion = suggest_data(&store, &ls, &cfg).unwrap();
        assert!(suggestion.rationale.contains("Insufficient"));
    }
}
