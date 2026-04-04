use mobius_core::experiment::ExperimentStore;
use mobius_core::store::{JsonlStore, SqliteStore};
use std::collections::BTreeSet;
use std::io::Write;

/// Export experiment history to CSV or JSON.
pub fn run(format: &str, store_backend: &str, output: Option<&str>) -> anyhow::Result<()> {
    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    let store: Box<dyn ExperimentStore> = match store_backend {
        "sqlite" => Box::new(SqliteStore::new(mobius_dir.join("history.db"))?),
        _ => Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?),
    };

    let history = store.load_all()?;
    if history.is_empty() {
        println!("No experiments to export.");
        return Ok(());
    }

    let content = match format {
        "json" => serde_json::to_string_pretty(&history)?,
        _ => {
            // CSV format
            let mut metric_keys = BTreeSet::new();
            let mut param_keys = BTreeSet::new();
            for r in &history {
                metric_keys.extend(r.metrics.keys().cloned());
                param_keys.extend(r.config.parameters.keys().cloned());
            }
            let metric_keys: Vec<String> = metric_keys.into_iter().collect();
            let param_keys: Vec<String> = param_keys.into_iter().collect();

            let mut csv = String::new();
            // Header
            csv.push_str("id,timestamp,status,duration_secs,cost_usd");
            for k in &metric_keys {
                csv.push(',');
                csv.push_str(k);
            }
            for k in &param_keys {
                csv.push(',');
                csv.push_str(&format!("param_{k}"));
            }
            csv.push('\n');

            // Rows
            for r in &history {
                csv.push_str(&format!(
                    "{},{},{:?},{:.2},{}",
                    r.id,
                    r.timestamp.to_rfc3339(),
                    r.status,
                    r.duration_secs,
                    r.cost_usd.map_or("".to_string(), |c| format!("{c:.4}")),
                ));
                for k in &metric_keys {
                    csv.push(',');
                    if let Some(v) = r.metrics.get(k) {
                        csv.push_str(&format!("{v:.6}"));
                    }
                }
                for k in &param_keys {
                    csv.push(',');
                    if let Some(v) = r.config.parameters.get(k) {
                        csv.push_str(&v.to_string());
                    }
                }
                csv.push('\n');
            }
            csv
        }
    };

    match output {
        Some(path) => {
            let mut file = std::fs::File::create(path)?;
            file.write_all(content.as_bytes())?;
            println!("Exported {} experiments to {path}", history.len());
        }
        None => print!("{content}"),
    }

    Ok(())
}
