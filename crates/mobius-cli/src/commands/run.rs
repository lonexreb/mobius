use mobius_core::compute::{ComputeBackend, OutputParser, SubprocessBackend};
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::{
    ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore,
};
use mobius_core::store::JsonlStore;
use std::collections::HashMap;

pub fn run(config_json: &str) -> anyhow::Result<()> {
    let mobius_config = MobiusConfig::load("mobius.toml").map_err(|e| {
        anyhow::anyhow!(
            "Failed to load mobius.toml: {}. Run 'mobius init' first.",
            e
        )
    })?;

    let overrides: HashMap<String, serde_json::Value> = serde_json::from_str(config_json)?;

    // Merge with defaults from sweep_space (use first value as default)
    let mut params = HashMap::new();
    for (k, values) in &mobius_config.experiment.sweep_space {
        if let Some(first) = values.first() {
            params.insert(k.clone(), first.clone());
        }
    }
    // Apply overrides
    for (k, v) in &overrides {
        params.insert(k.clone(), v.clone());
    }

    println!("Running experiment...");
    println!("Config overrides: {}", config_json);

    let backend = SubprocessBackend;
    let env: HashMap<String, String> = HashMap::new(); // TODO: use env_map from config

    let command = "echo '{\"f1\": 0.0}'".to_string(); // Placeholder — user configures in mobius.toml
    let output = backend.submit(&command, &env, 600)?;
    let parsed = OutputParser::parse(&output.stdout, &output.stderr);

    let mut metrics = parsed.metrics;
    if let Some(bs) = parsed.bench_score {
        metrics.insert("bench_score".into(), bs);
    }

    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");
    let mut store = JsonlStore::new(&store_path)?;
    let count = store.count()? + 1;

    let result = ExperimentResult {
        id: format!("exp-{:04}", count),
        timestamp: chrono::Utc::now(),
        config: ExperimentConfig {
            parameters: params,
            metadata: HashMap::new(),
        },
        metrics: metrics.clone(),
        per_segment: HashMap::new(),
        duration_secs: output.duration_secs,
        cost_usd: Some(mobius_config.experiment.cost_per_run),
        status: if output.exit_code == 0 {
            ExperimentStatus::Success
        } else {
            ExperimentStatus::Error
        },
        error: if output.exit_code != 0 {
            Some(output.stderr.chars().take(200).collect())
        } else {
            None
        },
    };

    store.append(&result)?;

    println!("\nResult: {}", result.id);
    for (k, v) in &result.metrics {
        println!("  {}: {:.4}", k, v);
    }
    println!("  Duration: {:.1}s", result.duration_secs);
    println!("  Status: {:?}", result.status);

    Ok(())
}
