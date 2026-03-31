use mobius_core::compute::{ComputeBackend, OutputParser, SubprocessBackend};
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore};
use mobius_core::store::JsonlStore;
use std::collections::HashMap;

pub fn run(spec_json: &str) -> anyhow::Result<()> {
    let _config = MobiusConfig::load("mobius.toml")
        .map_err(|e| anyhow::anyhow!("Failed to load mobius.toml: {}. Run 'mobius init' first.", e))?;

    let spec: HashMap<String, Vec<serde_json::Value>> = serde_json::from_str(spec_json)?;

    // Generate cartesian product
    let param_names: Vec<String> = spec.keys().cloned().collect();
    let param_values: Vec<&Vec<serde_json::Value>> = param_names.iter().map(|k| &spec[k]).collect();

    let combos = cartesian_product(&param_values);
    println!("Sweep: {} configs", combos.len());

    let store_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".mobius")
        .join("history.jsonl");
    let mut store = JsonlStore::new(&store_path)?;
    let backend = SubprocessBackend;

    let mut results: Vec<(HashMap<String, serde_json::Value>, HashMap<String, f64>)> = Vec::new();

    for (i, combo) in combos.iter().enumerate() {
        let mut params = HashMap::new();
        for (j, name) in param_names.iter().enumerate() {
            params.insert(name.clone(), combo[j].clone());
        }

        println!("[{}/{}] {:?}", i + 1, combos.len(), params);

        let output = backend.submit("echo '{\"f1\": 0.0}'", &HashMap::new(), 600)?;
        let parsed = OutputParser::parse(&output.stdout, &output.stderr);

        let count = store.count()? + 1;
        let result = ExperimentResult {
            id: format!("sweep-{:04}", count),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: params.clone(),
                metadata: HashMap::new(),
            },
            metrics: parsed.metrics.clone(),
            per_segment: HashMap::new(),
            duration_secs: output.duration_secs,
            cost_usd: None,
            status: ExperimentStatus::Success,
            error: None,
        };
        store.append(&result)?;
        results.push((params, parsed.metrics));
    }

    // Rank by f1
    results.sort_by(|a, b| {
        let fa = a.1.get("f1").unwrap_or(&0.0);
        let fb = b.1.get("f1").unwrap_or(&0.0);
        fb.partial_cmp(fa).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\nRanked results:");
    for (i, (params, metrics)) in results.iter().enumerate() {
        let f1 = metrics.get("f1").unwrap_or(&0.0);
        println!("  {}. f1={:.4}  {:?}", i + 1, f1, params);
    }

    Ok(())
}

fn cartesian_product(lists: &[&Vec<serde_json::Value>]) -> Vec<Vec<serde_json::Value>> {
    if lists.is_empty() {
        return vec![vec![]];
    }

    let mut result = Vec::new();
    let rest = cartesian_product(&lists[1..]);

    for item in lists[0] {
        for combo in &rest {
            let mut new_combo = vec![item.clone()];
            new_combo.extend(combo.iter().cloned());
            result.push(new_combo);
        }
    }

    result
}
