use mobius_core::experiment::{
    ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore,
};
use mobius_core::store::{JsonlStore, SqliteStore};
use std::collections::HashMap;

/// Enqueue a specific configuration for the agent to execute next.
pub fn run(config_json: &str, store_backend: &str) -> anyhow::Result<()> {
    let params: HashMap<String, serde_json::Value> = serde_json::from_str(config_json)?;

    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    let mut store: Box<dyn ExperimentStore> = match store_backend {
        "sqlite" => Box::new(SqliteStore::new(mobius_dir.join("history.db"))?),
        _ => Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?),
    };

    let count = store.count()? + 1;
    let result = ExperimentResult {
        id: format!("enqueue-{count:04}"),
        timestamp: chrono::Utc::now(),
        config: ExperimentConfig {
            parameters: params,
            metadata: HashMap::new(),
        },
        metrics: HashMap::new(),
        per_segment: HashMap::new(),
        duration_secs: 0.0,
        cost_usd: None,
        status: ExperimentStatus::Pending,
        error: None,
    };

    store.append(&result)?;
    println!(
        "{}",
        serde_json::json!({"id": result.id, "status": "pending", "enqueued": true})
    );
    Ok(())
}
