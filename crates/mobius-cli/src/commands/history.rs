use mobius_core::experiment::{ExperimentResult, ExperimentStore};
use mobius_core::store::JsonlStore;

/// Fetch the N most recent experiments from a store.
pub fn history_data(
    store: &dyn ExperimentStore,
    last: usize,
) -> anyhow::Result<Vec<ExperimentResult>> {
    store.get_recent(last)
}

pub fn run(last: usize) -> anyhow::Result<()> {
    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");

    let store = JsonlStore::new(&store_path)?;
    let recent = history_data(&store, last)?;

    if recent.is_empty() {
        println!("No experiments yet.");
        return Ok(());
    }

    println!(
        "\n  {:<12} {:>8} {:>8} {:>8} {:>8}  Status",
        "ID", "F1", "Prec", "Recall", "Secs"
    );
    println!("  {}", "-".repeat(66));

    for entry in &recent {
        let f1 = entry
            .metrics
            .get("f1")
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "—".into());
        let p = entry
            .metrics
            .get("precision")
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "—".into());
        let r = entry
            .metrics
            .get("recall")
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "—".into());

        println!(
            "  {:<12} {:>8} {:>8} {:>8} {:>8.1}  {:?}",
            entry.id, f1, p, r, entry.duration_secs, entry.status
        );
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn make_result(id: &str) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), 0.5);
        ExperimentResult {
            id: id.into(),
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
    fn test_history_returns_recent() {
        let dir = TempDir::new().unwrap();
        let mut store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        for i in 0..10 {
            store.append(&make_result(&format!("exp-{i}"))).unwrap();
        }
        let results = history_data(&store, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_history_empty() {
        let dir = TempDir::new().unwrap();
        let store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        let results = history_data(&store, 10).unwrap();
        assert!(results.is_empty());
    }
}
