//! Ask-and-tell: tell the system about an experiment result.
//!
//! Accepts config and metrics as JSON, stores the result, and prints
//! a JSON confirmation to stdout.

use mobius_core::experiment::{
    ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore,
};
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

/// Run the tell command, storing an experiment result and printing JSON confirmation.
pub fn run(config_json: &str, metrics_json: &str, store_backend: &str) -> anyhow::Result<()> {
    let parameters: HashMap<String, serde_json::Value> = serde_json::from_str(config_json)
        .map_err(|e| anyhow::anyhow!("Invalid --config JSON: {}", e))?;

    let metrics: HashMap<String, f64> = serde_json::from_str(metrics_json)
        .map_err(|e| anyhow::anyhow!("Invalid --metrics JSON: {}", e))?;

    let mut store = build_store(store_backend)?;
    let count = store.count()? + 1;
    let id = format!("tell-{count:04}");

    let result = ExperimentResult {
        id: id.clone(),
        timestamp: chrono::Utc::now(),
        config: ExperimentConfig {
            parameters,
            metadata: HashMap::new(),
        },
        metrics,
        per_segment: HashMap::new(),
        duration_secs: 0.0,
        cost_usd: None,
        status: ExperimentStatus::Success,
        error: None,
    };

    store.append(&result)?;

    // Output ONLY valid JSON -- no decorative text
    let output = serde_json::json!({
        "id": id,
        "stored": true,
    });
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::store::JsonlStore;
    use tempfile::TempDir;

    #[test]
    fn test_tell_stores_result() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("history.jsonl");
        let mut store = JsonlStore::new(&store_path).unwrap();

        let params: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"lr": 0.01}"#).unwrap();
        let metrics: HashMap<String, f64> = serde_json::from_str(r#"{"f1": 0.85}"#).unwrap();

        let result = ExperimentResult {
            id: "tell-0001".into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: params,
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 0.0,
            cost_usd: None,
            status: ExperimentStatus::Success,
            error: None,
        };

        store.append(&result).unwrap();
        let all = store.load_all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "tell-0001");
        assert_eq!(all[0].metrics["f1"], 0.85);
    }

    #[test]
    fn test_build_store_jsonl() {
        let store = build_store("jsonl");
        assert!(store.is_ok());
    }
}
