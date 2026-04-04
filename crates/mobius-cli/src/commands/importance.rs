//! Parameter importance analysis command.
//!
//! Ranks parameters by their impact on the primary metric using
//! fANOVA-style between-group variance analysis.

use crate::style;
use mobius_claw::importance::compute_importance;
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::ExperimentStore;
use mobius_core::store::{JsonlStore, SqliteStore};

/// Build the appropriate experiment store based on the backend name.
fn build_store(store_backend: &str) -> anyhow::Result<Box<dyn ExperimentStore>> {
    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    match store_backend {
        "sqlite" => Ok(Box::new(SqliteStore::new(mobius_dir.join("history.db"))?)),
        _ => Ok(Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?)),
    }
}

/// Run the parameter importance analysis command.
pub fn run(metric: &str, store_backend: &str) -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").map_err(|e| {
        anyhow::anyhow!(
            "Failed to load mobius.toml: {}. Run 'mobius init' first.",
            e
        )
    })?;

    let store = build_store(store_backend)?;
    let history = store.load_all()?;

    if history.is_empty() {
        println!(
            "{}",
            style::warn("No experiments yet. Run some experiments first.")
        );
        return Ok(());
    }

    let importances = compute_importance(&history, &config.experiment.sweep_space, metric);

    if importances.is_empty() {
        println!(
            "{}",
            style::warn("Not enough data to compute parameter importance.")
        );
        return Ok(());
    }

    let best_val = history
        .iter()
        .filter_map(|r| r.metrics.get(metric).copied())
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n{}", style::section("PARAMETER IMPORTANCE"));
    println!(
        "  {}  |  Experiments: {}\n",
        style::metric(metric, best_val),
        history.len(),
    );
    println!(
        "  {:<20} {:>12} {:>8} {:>24}",
        "Parameter", "Importance", "Values", "Metric Range"
    );
    println!("  {}", "-".repeat(68));

    for pi in &importances {
        let bar_len = (pi.importance * 20.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        let range_str = format!("[{:.4}, {:.4}]", pi.metric_range.0, pi.metric_range.1);
        println!(
            "  {:<20} {:>10.1}%  {:>8} {:>24}  {}",
            pi.param,
            pi.importance * 100.0,
            pi.num_unique_values,
            range_str,
            style::improvement(pi.metric_range.1 - pi.metric_range.0),
        );
        if !bar.is_empty() {
            println!("  {:<20} {}", "", bar);
        }
    }

    println!();
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
