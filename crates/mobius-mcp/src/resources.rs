//! MCP resource handlers for read-only access to Mobius state.
//!
//! Exposes project configuration, budget status, and best experiment
//! result as MCP resources that agents can read without calling tools.

use crate::error::to_mcp_error;
use crate::state::State;
use mobius_core::experiment::ExperimentStore;
use rmcp::model::{
    AnnotateAble, ErrorData, ListResourcesResult, RawResource, ReadResourceRequestParams,
    ReadResourceResult, ResourceContents,
};

/// The set of resource URIs this server exposes.
pub const RESOURCE_URIS: &[(&str, &str)] = &[
    ("mobius://config", "Project Configuration"),
    ("mobius://budget", "Budget Status"),
    ("mobius://best", "Best Experiment Result"),
];

/// Build the resource list for `list_resources`.
pub fn list_resources() -> ListResourcesResult {
    let resources = RESOURCE_URIS
        .iter()
        .map(|(uri, name)| RawResource::new(uri.to_string(), name.to_string()).no_annotation())
        .collect();
    ListResourcesResult {
        resources,
        next_cursor: None,
        meta: None,
    }
}

/// Read a single resource by URI.
pub async fn read_resource(
    state: &State,
    request: ReadResourceRequestParams,
) -> Result<ReadResourceResult, ErrorData> {
    let uri = request.uri.as_str();
    let s = state.read().await;

    let content = match uri {
        "mobius://config" => match &s.config {
            Some(cfg) => serde_json::to_string_pretty(cfg).map_err(to_mcp_error)?,
            None => r#"{"error": "No mobius.toml loaded"}"#.to_string(),
        },
        "mobius://budget" => {
            let cost = s
                .config
                .as_ref()
                .map(|c| c.experiment.cost_per_run)
                .unwrap_or(1.0);
            serde_json::to_string_pretty(&serde_json::json!({
                "spent": s.budget.spent,
                "limit": s.budget.limit,
                "remaining": s.budget.remaining(),
                "experiments_remaining": s.budget.experiments_remaining(cost),
            }))
            .map_err(to_mcp_error)?
        }
        "mobius://best" => match s.store.get_best("f1").map_err(to_mcp_error)? {
            Some(best) => serde_json::to_string_pretty(&serde_json::json!({
                "id": best.id,
                "metrics": best.metrics,
                "config": best.config.parameters,
                "timestamp": best.timestamp.to_rfc3339(),
            }))
            .map_err(to_mcp_error)?,
            None => r#"{"error": "No experiments yet"}"#.to_string(),
        },
        _ => {
            return Err(ErrorData::resource_not_found(
                "Unknown resource",
                Some(serde_json::json!({"uri": uri})),
            ));
        }
    };

    Ok(ReadResourceResult::new(vec![ResourceContents::text(
        content,
        request.uri.clone(),
    )]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SharedState;
    use mobius_core::config::{
        AgentSection, BenchSection, ComputeSection, ExperimentSection, MobiusConfig, ProjectConfig,
    };
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn test_config() -> MobiusConfig {
        MobiusConfig {
            project: ProjectConfig {
                name: "test".into(),
                version: "0.1.0".into(),
            },
            experiment: ExperimentSection::default(),
            bench: BenchSection::default(),
            compute: ComputeSection::default(),
            agent: AgentSection::default(),
        }
    }

    fn make_result(id: &str, f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        ExperimentResult {
            id: id.to_string(),
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
    fn test_list_resources() {
        let result = list_resources();
        assert_eq!(result.resources.len(), 3);
    }

    #[tokio::test]
    async fn test_read_config_with_config() {
        let dir = TempDir::new().unwrap();
        let state = SharedState::with_dir(dir.path().to_path_buf(), Some(test_config())).unwrap();
        let request = ReadResourceRequestParams::new("mobius://config");
        let result = read_resource(&state, request).await.unwrap();
        assert!(!result.contents.is_empty());
    }

    #[tokio::test]
    async fn test_read_config_no_config() {
        let dir = TempDir::new().unwrap();
        let state = SharedState::with_dir(dir.path().to_path_buf(), None).unwrap();
        let request = ReadResourceRequestParams::new("mobius://config");
        let result = read_resource(&state, request).await.unwrap();
        let text = format!("{:?}", result.contents);
        assert!(text.contains("No mobius.toml"));
    }

    #[tokio::test]
    async fn test_read_budget() {
        let dir = TempDir::new().unwrap();
        let state = SharedState::with_dir(dir.path().to_path_buf(), Some(test_config())).unwrap();
        let request = ReadResourceRequestParams::new("mobius://budget");
        let result = read_resource(&state, request).await.unwrap();
        assert!(!result.contents.is_empty());
    }

    #[tokio::test]
    async fn test_read_best_empty() {
        let dir = TempDir::new().unwrap();
        let state = SharedState::with_dir(dir.path().to_path_buf(), None).unwrap();
        let request = ReadResourceRequestParams::new("mobius://best");
        let result = read_resource(&state, request).await.unwrap();
        let text = format!("{:?}", result.contents);
        assert!(text.contains("No experiments"));
    }

    #[tokio::test]
    async fn test_read_best_with_data() {
        let dir = TempDir::new().unwrap();
        let state = SharedState::with_dir(dir.path().to_path_buf(), None).unwrap();
        {
            let mut s = state.write().await;
            s.store.append(&make_result("exp-1", 0.80)).unwrap();
        }
        let request = ReadResourceRequestParams::new("mobius://best");
        let result = read_resource(&state, request).await.unwrap();
        let text = format!("{:?}", result.contents);
        assert!(text.contains("exp-1"));
    }

    #[tokio::test]
    async fn test_read_unknown_uri() {
        let dir = TempDir::new().unwrap();
        let state = SharedState::with_dir(dir.path().to_path_buf(), None).unwrap();
        let request = ReadResourceRequestParams::new("mobius://unknown");
        let result = read_resource(&state, request).await;
        assert!(result.is_err());
    }
}
