//! Compare experiments side by side.
//!
//! Loads experiments by ID and displays their configs and metrics
//! in a tabular comparison with color-coded deltas.

use crate::style;
use mobius_core::experiment::{ExperimentResult, ExperimentStore};
use mobius_core::store::{JsonlStore, SqliteStore};
use std::collections::BTreeSet;

/// Build the appropriate experiment store based on the backend name.
fn build_store(store_backend: &str) -> anyhow::Result<Box<dyn ExperimentStore>> {
    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    match store_backend {
        "sqlite" => Ok(Box::new(SqliteStore::new(mobius_dir.join("history.db"))?)),
        _ => Ok(Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?)),
    }
}

/// Find experiments by IDs from the full history.
fn find_experiments(all: &[ExperimentResult], ids: &[String]) -> Vec<Option<ExperimentResult>> {
    ids.iter()
        .map(|id| all.iter().find(|r| r.id == *id).cloned())
        .collect()
}

/// Collect all unique parameter names across experiments.
fn collect_param_keys(experiments: &[&ExperimentResult]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for exp in experiments {
        for k in exp.config.parameters.keys() {
            keys.insert(k.clone());
        }
    }
    keys.into_iter().collect()
}

/// Collect all unique metric names across experiments.
fn collect_metric_keys(experiments: &[&ExperimentResult]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for exp in experiments {
        for k in exp.metrics.keys() {
            keys.insert(k.clone());
        }
    }
    keys.into_iter().collect()
}

/// Run the compare command, printing a side-by-side comparison table.
pub fn run(ids: &[String], store_backend: &str) -> anyhow::Result<()> {
    if ids.len() < 2 {
        anyhow::bail!("At least 2 experiment IDs are required for comparison.");
    }

    let store = build_store(store_backend)?;
    let all = store.load_all()?;
    let found = find_experiments(&all, ids);

    // Check for missing experiments
    let mut missing: Vec<&str> = Vec::new();
    let mut experiments: Vec<&ExperimentResult> = Vec::new();
    for (i, result) in found.iter().enumerate() {
        match result {
            Some(r) => experiments.push(r),
            None => missing.push(&ids[i]),
        }
    }

    if !missing.is_empty() {
        println!(
            "{}",
            style::warn(&format!("Experiments not found: {}", missing.join(", ")))
        );
        if experiments.len() < 2 {
            anyhow::bail!("Need at least 2 found experiments to compare.");
        }
    }

    let col_width = 16;

    // Header row
    println!("\n{}", style::section("EXPERIMENT COMPARISON"));
    print!("  {:<20}", "");
    for exp in &experiments {
        print!("{:>width$}", exp.id, width = col_width);
    }
    // Delta column (last vs first)
    if experiments.len() == 2 {
        print!("{:>width$}", "Delta", width = col_width);
    }
    println!();
    let total_width =
        20 + experiments.len() * col_width + if experiments.len() == 2 { col_width } else { 0 };
    println!("  {}", "-".repeat(total_width));

    // Config parameters
    let param_keys = collect_param_keys(&experiments);
    if !param_keys.is_empty() {
        println!("  PARAMETERS");
        for key in &param_keys {
            print!("  {:<20}", key);
            for exp in &experiments {
                let val = exp
                    .config
                    .parameters
                    .get(key)
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|| "-".into());
                print!("{:>width$}", val, width = col_width);
            }
            println!();
        }
        println!();
    }

    // Metrics
    let metric_keys = collect_metric_keys(&experiments);
    if !metric_keys.is_empty() {
        println!("  METRICS");
        for key in &metric_keys {
            print!("  {:<20}", key);
            let mut values: Vec<Option<f64>> = Vec::new();
            for exp in &experiments {
                let val = exp.metrics.get(key).copied();
                values.push(val);
                match val {
                    Some(v) => print!("{:>width$.4}", v, width = col_width),
                    None => print!("{:>width$}", "-", width = col_width),
                }
            }
            // Delta column for 2-experiment comparison
            if experiments.len() == 2 {
                if let (Some(v1), Some(v2)) = (values[0], values[1]) {
                    let delta = v2 - v1;
                    let delta_str = style::improvement(delta);
                    print!("{:>width$}", delta_str, width = col_width);
                } else {
                    print!("{:>width$}", "-", width = col_width);
                }
            }
            println!();
        }
        println!();
    }

    // Duration and cost
    println!("  META");
    print!("  {:<20}", "duration_secs");
    for exp in &experiments {
        print!("{:>width$.1}", exp.duration_secs, width = col_width);
    }
    println!();

    print!("  {:<20}", "cost_usd");
    for exp in &experiments {
        let cost_str = exp
            .cost_usd
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "-".into());
        print!("{:>width$}", cost_str, width = col_width);
    }
    println!();

    print!("  {:<20}", "status");
    for exp in &experiments {
        let is_success = format!("{:?}", exp.status) == "Success";
        let label = style::target_status(is_success);
        print!("{:>width$}", label, width = col_width);
    }
    println!("\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use std::collections::HashMap;

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
    fn test_find_experiments() {
        let all = vec![
            make_result("exp-1", 0.7, 0.01),
            make_result("exp-2", 0.8, 0.1),
            make_result("exp-3", 0.9, 0.001),
        ];

        let found = find_experiments(&all, &["exp-1".into(), "exp-3".into()]);
        assert_eq!(found.len(), 2);
        assert!(found[0].is_some());
        assert!(found[1].is_some());
        assert_eq!(found[0].as_ref().unwrap().id, "exp-1");
        assert_eq!(found[1].as_ref().unwrap().id, "exp-3");
    }

    #[test]
    fn test_find_experiments_missing() {
        let all = vec![make_result("exp-1", 0.7, 0.01)];
        let found = find_experiments(&all, &["exp-1".into(), "exp-99".into()]);
        assert!(found[0].is_some());
        assert!(found[1].is_none());
    }

    #[test]
    fn test_collect_param_keys() {
        let r1 = make_result("exp-1", 0.7, 0.01);
        let mut r2 = make_result("exp-2", 0.8, 0.1);
        r2.config
            .parameters
            .insert("batch_size".into(), serde_json::json!(32));

        let experiments = vec![&r1, &r2];
        let keys = collect_param_keys(&experiments);
        assert!(keys.contains(&"lr".to_string()));
        assert!(keys.contains(&"batch_size".to_string()));
    }

    #[test]
    fn test_collect_metric_keys() {
        let mut r1 = make_result("exp-1", 0.7, 0.01);
        r1.metrics.insert("precision".into(), 0.75);
        let r2 = make_result("exp-2", 0.8, 0.1);

        let experiments = vec![&r1, &r2];
        let keys = collect_metric_keys(&experiments);
        assert!(keys.contains(&"f1".to_string()));
        assert!(keys.contains(&"precision".to_string()));
    }
}
