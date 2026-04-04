use mobius_core::compute::{
    ComputeBackend, OutputParser, ParallelBackend, SubprocessBackend, SyncAdapter, config_to_env,
};
use mobius_core::config::MobiusConfig;
use mobius_core::experiment::{
    ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore,
};
use mobius_core::store::{JsonlStore, SqliteStore};
use std::collections::HashMap;
use std::sync::Arc;

pub fn run(
    spec_json: &str,
    parallel: bool,
    max_concurrency: usize,
    store_backend: &str,
) -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").map_err(|e| {
        anyhow::anyhow!(
            "Failed to load mobius.toml: {}. Run 'mobius init' first.",
            e
        )
    })?;

    let spec: HashMap<String, Vec<serde_json::Value>> = serde_json::from_str(spec_json)?;

    let param_names: Vec<String> = spec.keys().cloned().collect();
    let param_values: Vec<&Vec<serde_json::Value>> = param_names.iter().map(|k| &spec[k]).collect();
    let combos = cartesian_product(&param_values);
    println!(
        "Sweep: {} configs{}",
        combos.len(),
        if parallel { " (parallel)" } else { "" }
    );

    let command = config
        .experiment
        .command
        .as_deref()
        .unwrap_or("echo '{\"f1\": 0.0}'");
    let timeout = config.experiment.timeout_secs;

    // Build all param sets
    let all_params: Vec<HashMap<String, serde_json::Value>> = combos
        .iter()
        .map(|combo| {
            let mut params = HashMap::new();
            for (j, name) in param_names.iter().enumerate() {
                params.insert(name.clone(), combo[j].clone());
            }
            params
        })
        .collect();

    let outputs = if parallel {
        run_parallel(
            &all_params,
            command,
            &config.experiment.env_map,
            timeout,
            max_concurrency,
        )?
    } else {
        run_sequential(&all_params, command, &config.experiment.env_map, timeout)?
    };

    // Store results and rank
    let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
    let mut store: Box<dyn ExperimentStore> = match store_backend {
        "sqlite" => Box::new(SqliteStore::new(mobius_dir.join("history.db"))?),
        _ => Box::new(JsonlStore::new(mobius_dir.join("history.jsonl"))?),
    };
    let mut results: Vec<(HashMap<String, serde_json::Value>, HashMap<String, f64>)> = Vec::new();

    for (params, metrics) in all_params.into_iter().zip(outputs) {
        let count = store.count()? + 1;
        let result = ExperimentResult {
            id: format!("sweep-{count:04}"),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: params.clone(),
                metadata: HashMap::new(),
            },
            metrics: metrics.clone(),
            per_segment: HashMap::new(),
            duration_secs: 0.0,
            cost_usd: None,
            status: ExperimentStatus::Success,
            error: None,
        };
        store.append(&result)?;
        results.push((params, metrics));
    }

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

fn run_sequential(
    all_params: &[HashMap<String, serde_json::Value>],
    command: &str,
    env_map: &HashMap<String, String>,
    timeout: u64,
) -> anyhow::Result<Vec<HashMap<String, f64>>> {
    let backend = SubprocessBackend;
    let mut outputs = Vec::new();
    for (i, params) in all_params.iter().enumerate() {
        println!("[{}/{}] {:?}", i + 1, all_params.len(), params);
        let env = config_to_env(params, env_map);
        let output = backend.submit(command, &env, timeout)?;
        let parsed = OutputParser::parse(&output.stdout, &output.stderr);
        outputs.push(parsed.metrics);
    }
    Ok(outputs)
}

fn run_parallel(
    all_params: &[HashMap<String, serde_json::Value>],
    command: &str,
    env_map: &HashMap<String, String>,
    timeout: u64,
    max_concurrency: usize,
) -> anyhow::Result<Vec<HashMap<String, f64>>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let adapter = Arc::new(SyncAdapter::new(SubprocessBackend));
        let parallel = ParallelBackend::new(adapter, max_concurrency);

        let jobs: Vec<_> = all_params
            .iter()
            .map(|params| {
                let env = config_to_env(params, env_map);
                (command.to_string(), env, timeout)
            })
            .collect();

        let results = parallel.submit_batch(jobs).await;
        let mut outputs = Vec::new();
        for (i, r) in results.into_iter().enumerate() {
            match r {
                Ok(output) => {
                    let parsed = OutputParser::parse(&output.stdout, &output.stderr);
                    outputs.push(parsed.metrics);
                }
                Err(e) => {
                    eprintln!("[{}/{}] Error: {}", i + 1, all_params.len(), e);
                    outputs.push(HashMap::new());
                }
            }
        }
        Ok(outputs)
    })
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
