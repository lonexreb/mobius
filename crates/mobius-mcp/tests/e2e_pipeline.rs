//! End-to-end integration tests that wire together all Mobius crates
//! through the MCP handler layer.

use mobius_claw::learning::extract_learning;
use mobius_core::config::{
    AgentSection, BenchSection, ComputeSection, ExperimentSection, MobiusConfig, ProjectConfig,
};
use mobius_core::experiment::{
    ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore,
};
use mobius_mcp::state::SharedState;
use mobius_mcp::tools::{
    BudgetParams, HistoryParams, ParetoParams, StatusParams, SuggestParams, handle_budget,
    handle_history, handle_pareto, handle_status, handle_suggest,
};
use std::collections::HashMap;
use tempfile::TempDir;

fn e2e_config() -> MobiusConfig {
    let mut targets = HashMap::new();
    targets.insert("f1".into(), 0.85);
    targets.insert("precision".into(), 0.90);

    let mut sweep_space = HashMap::new();
    sweep_space.insert(
        "learning_rate".into(),
        vec![
            serde_json::json!(0.001),
            serde_json::json!(0.01),
            serde_json::json!(0.1),
        ],
    );
    sweep_space.insert(
        "batch_size".into(),
        vec![
            serde_json::json!(16),
            serde_json::json!(32),
            serde_json::json!(64),
        ],
    );

    MobiusConfig {
        project: ProjectConfig {
            name: "e2e-test".into(),
            version: "0.1.0".into(),
        },
        experiment: ExperimentSection {
            budget_usd: 20.0,
            cost_per_run: 1.0,
            targets,
            sweep_space,
        },
        bench: BenchSection::default(),
        compute: ComputeSection::default(),
        agent: AgentSection::default(),
    }
}

fn make_result(id: &str, f1: f64, precision: f64, lr: f64, bs: i64) -> ExperimentResult {
    let mut metrics = HashMap::new();
    metrics.insert("f1".into(), f1);
    metrics.insert("precision".into(), precision);
    metrics.insert("recall".into(), f1 - 0.05);

    let mut params = HashMap::new();
    params.insert("learning_rate".into(), serde_json::json!(lr));
    params.insert("batch_size".into(), serde_json::json!(bs));

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

/// Full pipeline: seed experiments -> learn -> suggest.
#[tokio::test]
async fn test_full_pipeline_config_to_suggest() {
    let dir = TempDir::new().unwrap();
    let state = SharedState::with_dir(dir.path().to_path_buf(), Some(e2e_config())).unwrap();

    // Phase 1: Seed 3 experiments with improving metrics
    let r1 = make_result("exp-0001", 0.60, 0.65, 0.001, 16);
    let r2 = make_result("exp-0002", 0.72, 0.78, 0.01, 32);
    let r3 = make_result("exp-0003", 0.81, 0.85, 0.1, 32);
    {
        let mut s = state.write().await;
        s.store.append(&r1).unwrap();
        s.store.append(&r2).unwrap();
        s.store.append(&r3).unwrap();
    }

    // Phase 2: Verify status reflects 3 experiments
    let status = handle_status(&state, StatusParams { metric: None })
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&status).unwrap();
    assert_eq!(v["experiment_count"], 3);
    assert!(v["best_metrics"]["f1"].as_f64().unwrap() > 0.80);

    // Phase 3: Extract learnings from consecutive pairs
    let learning_1_2 = extract_learning(&r1, &r2, "f1");
    assert!(learning_1_2.is_some());
    let learning = learning_1_2.unwrap();
    assert!(learning.metric_deltas["f1"] > 0.0); // f1 improved

    let learning_2_3 = extract_learning(&r2, &r3, "f1");
    assert!(learning_2_3.is_some());

    // Seed learnings into the store
    {
        let mut s = state.write().await;
        s.learning_store.append(&learning).unwrap();
        s.learning_store.append(&learning_2_3.unwrap()).unwrap();
    }

    // Phase 4: Suggest next config using accumulated history + learnings
    let suggestion = handle_suggest(
        &state,
        SuggestParams {
            metric: None,
            strategy: None,
        },
    )
    .await
    .unwrap();
    let sv: serde_json::Value = serde_json::from_str(&suggestion).unwrap();
    assert!(!sv["rationale"].as_str().unwrap().is_empty());
    assert!(!sv["full_config"].as_object().unwrap().is_empty());
}

/// Budget exhaustion: low budget, spend until blocked.
#[tokio::test]
async fn test_e2e_budget_exhaustion() {
    let dir = TempDir::new().unwrap();
    let state = SharedState::with_dir(dir.path().to_path_buf(), Some(e2e_config())).unwrap();

    // Set budget to 3.0 and spend 3.5
    let _ = handle_budget(
        &state,
        BudgetParams {
            set_limit: Some(3.0),
            record_spend: None,
        },
    )
    .await
    .unwrap();

    let _ = handle_budget(
        &state,
        BudgetParams {
            set_limit: None,
            record_spend: Some(3.5),
        },
    )
    .await
    .unwrap();

    let result = handle_budget(
        &state,
        BudgetParams {
            set_limit: None,
            record_spend: None,
        },
    )
    .await
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(v["remaining"], 0.0); // clamped to 0
    assert_eq!(v["experiments_remaining"], 0);
}

/// Pareto front after seeding experiments with tradeoffs.
#[tokio::test]
async fn test_e2e_pareto_after_experiments() {
    let dir = TempDir::new().unwrap();
    let state = SharedState::with_dir(dir.path().to_path_buf(), Some(e2e_config())).unwrap();

    // Seed experiments with f1/precision tradeoffs
    {
        let mut s = state.write().await;
        s.store
            .append(&make_result("a", 0.90, 0.70, 0.01, 16))
            .unwrap(); // high f1, low precision
        s.store
            .append(&make_result("b", 0.70, 0.95, 0.1, 32))
            .unwrap(); // low f1, high precision
        s.store
            .append(&make_result("c", 0.60, 0.60, 0.001, 64))
            .unwrap(); // dominated
    }

    let result = handle_pareto(
        &state,
        ParetoParams {
            x_metric: "f1".into(),
            y_metric: "precision".into(),
        },
    )
    .await
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&result).unwrap();
    // a and b should be on the front; c is dominated
    assert!(v["front_size"].as_u64().unwrap() >= 2);
}

/// History retrieval after seeding experiments.
#[tokio::test]
async fn test_e2e_history_with_seeded_data() {
    let dir = TempDir::new().unwrap();
    let state = SharedState::with_dir(dir.path().to_path_buf(), Some(e2e_config())).unwrap();

    {
        let mut s = state.write().await;
        for i in 0..5 {
            s.store
                .append(&make_result(
                    &format!("exp-{i}"),
                    0.5 + i as f64 * 0.05,
                    0.6 + i as f64 * 0.05,
                    0.01,
                    32,
                ))
                .unwrap();
        }
    }

    let result = handle_history(&state, HistoryParams { last: Some(3) })
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(v["count"], 3);
}
