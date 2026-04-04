use mobius_core::experiment::ExperimentStore;
use mobius_core::store::JsonlStore;

pub fn run(last: usize) -> anyhow::Result<()> {
    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");

    let store = JsonlStore::new(&store_path)?;
    let recent = store.get_recent(last)?;

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
        let f1 = entry.metrics.get("f1").map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
        let p = entry.metrics.get("precision").map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
        let r = entry.metrics.get("recall").map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());

        println!(
            "  {:<12} {:>8} {:>8} {:>8} {:>8.1}  {:?}",
            entry.id,
            f1,
            p,
            r,
            entry.duration_secs,
            entry.status
        );
    }

    println!();
    Ok(())
}
