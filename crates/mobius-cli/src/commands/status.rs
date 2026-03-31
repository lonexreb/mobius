use mobius_core::config::MobiusConfig;
use mobius_core::experiment::ExperimentStore;
use mobius_core::store::JsonlStore;

pub fn run() -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").ok();

    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");

    let store = JsonlStore::new(&store_path)?;
    let count = store.count()?;

    println!("\n{:=<60}", "");
    println!("  MOBIUS STATUS");
    println!("{:=<60}", "");
    println!("  History entries: {}", count);

    if let Some(config) = &config {
        println!("\n  TARGETS:");
        for (metric, target) in &config.experiment.targets {
            let current = store
                .get_best(metric)?
                .and_then(|r| r.metrics.get(metric).copied())
                .unwrap_or(0.0);
            let gap = target - current;
            let status = if gap <= 0.0 { "MET" } else { "GAP" };
            println!(
                "    {:<20} {:.4} / {:.4}  [{}{}]",
                metric,
                current,
                target,
                status,
                if gap > 0.0 {
                    format!(": {:.4}", gap)
                } else {
                    String::new()
                }
            );
        }
    }

    if let Some(best) = store.get_best("f1")? {
        println!("\n  BEST CONFIG (by f1):");
        for (k, v) in &best.config.parameters {
            println!("    {}: {}", k, v);
        }
        println!("\n  BEST METRICS:");
        for (k, v) in &best.metrics {
            println!("    {}: {:.4}", k, v);
        }
    } else {
        println!("\n  No experiments yet. Run: mobius run --config '{{}}'");
    }

    println!("{:=<60}\n", "");
    Ok(())
}
