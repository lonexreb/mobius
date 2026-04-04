use mobius_core::config::MobiusConfig;
use mobius_core::experiment::{ExperimentResult, ExperimentStore};
use mobius_core::store::JsonlStore;

/// Computed status for a single target metric.
pub struct TargetStatus {
    pub metric: String,
    pub current: f64,
    pub target: f64,
    pub gap: f64,
    pub met: bool,
}

/// Aggregate status data computed from the store and config.
pub struct StatusReport {
    pub count: usize,
    pub targets: Vec<TargetStatus>,
    pub best: Option<ExperimentResult>,
}

/// Compute status data from a store and optional config.
pub fn status_data(
    store: &dyn ExperimentStore,
    config: Option<&MobiusConfig>,
) -> anyhow::Result<StatusReport> {
    let count = store.count()?;

    let mut targets = Vec::new();
    if let Some(cfg) = config {
        for (metric, &target) in &cfg.experiment.targets {
            let current = store
                .get_best(metric)?
                .and_then(|r| r.metrics.get(metric).copied())
                .unwrap_or(0.0);
            let gap = (target - current).max(0.0);
            targets.push(TargetStatus {
                metric: metric.clone(),
                current,
                target,
                gap,
                met: current >= target,
            });
        }
    }

    let best = store.get_best("f1")?;

    Ok(StatusReport {
        count,
        targets,
        best,
    })
}

pub fn run() -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").ok();

    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");

    let store = JsonlStore::new(&store_path)?;
    let report = status_data(&store, config.as_ref())?;

    println!("\n{:=<60}", "");
    println!("  MOBIUS STATUS");
    println!("{:=<60}", "");
    println!("  History entries: {}", report.count);

    if !report.targets.is_empty() {
        println!("\n  TARGETS:");
        for t in &report.targets {
            let label = if t.met { "MET" } else { "GAP" };
            println!(
                "    {:<20} {:.4} / {:.4}  [{}{}]",
                t.metric,
                t.current,
                t.target,
                label,
                if t.gap > 0.0 {
                    format!(": {:.4}", t.gap)
                } else {
                    String::new()
                }
            );
        }
    }

    if let Some(best) = &report.best {
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

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::config::{
        AgentSection, BenchSection, ComputeSection, ExperimentSection, ProjectConfig,
    };
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn make_config_with_targets() -> MobiusConfig {
        let mut targets = HashMap::new();
        targets.insert("f1".into(), 0.85);
        MobiusConfig {
            project: ProjectConfig {
                name: "test".into(),
                version: "0.1.0".into(),
            },
            experiment: ExperimentSection {
                budget_usd: 20.0,
                cost_per_run: 1.0,
                targets,
                sweep_space: HashMap::new(),
                ..Default::default()
            },
            bench: BenchSection::default(),
            compute: ComputeSection::default(),
            agent: AgentSection::default(),
        }
    }

    fn make_result(id: &str, f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
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
            cost_usd: Some(1.0),
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    #[test]
    fn test_status_empty_store() {
        let dir = TempDir::new().unwrap();
        let store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        let report = status_data(&store, None).unwrap();
        assert_eq!(report.count, 0);
        assert!(report.best.is_none());
        assert!(report.targets.is_empty());
    }

    #[test]
    fn test_status_with_experiments() {
        let dir = TempDir::new().unwrap();
        let mut store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        store.append(&make_result("exp-1", 0.60)).unwrap();
        store.append(&make_result("exp-2", 0.75)).unwrap();
        store.append(&make_result("exp-3", 0.70)).unwrap();

        let report = status_data(&store, None).unwrap();
        assert_eq!(report.count, 3);
        assert!(report.best.is_some());
        assert!(report.best.unwrap().metrics["f1"] > 0.74);
    }

    #[test]
    fn test_status_targets_gap() {
        let dir = TempDir::new().unwrap();
        let mut store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        store.append(&make_result("exp-1", 0.70)).unwrap();

        let cfg = make_config_with_targets();
        let report = status_data(&store, Some(&cfg)).unwrap();
        assert_eq!(report.targets.len(), 1);
        let t = &report.targets[0];
        assert!(!t.met);
        assert!(t.gap > 0.14);
    }

    #[test]
    fn test_status_no_config() {
        let dir = TempDir::new().unwrap();
        let store = JsonlStore::new(dir.path().join("history.jsonl")).unwrap();
        let report = status_data(&store, None).unwrap();
        assert!(report.targets.is_empty());
    }
}
