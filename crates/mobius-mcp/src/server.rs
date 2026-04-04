//! MCP server struct and handler implementation.
//!
//! [`MobiusServer`] wires together the tool router (via `#[tool_router]` macro)
//! and resource handlers into a single `ServerHandler` implementation.

use crate::resources;
use crate::state::State;
use crate::tools;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, Content, ListResourcesResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, ServerCapabilities, ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler, tool, tool_handler, tool_router};

/// MCP server exposing Mobius experiment tools and resources.
///
/// All mutable state is shared via `Arc<RwLock<SharedState>>` to satisfy the
/// `Clone` requirement of `ServerHandler`. Tool handlers delegate to helper
/// functions in the [`tools`](crate::tools) module.
#[derive(Debug, Clone)]
pub struct MobiusServer {
    state: State,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MobiusServer {
    /// Create a new server with initialized shared state.
    pub fn new(state: State) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Get the current status of the Mobius experiment project including experiment count, target metric gaps, best configuration found so far, and budget remaining. Call this first to orient yourself."
    )]
    async fn mobius_status(
        &self,
        Parameters(params): Parameters<tools::StatusParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_status(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "List recent experiment results with their configs, metrics, duration, and status. Use to understand what has been tried and the trajectory of improvement."
    )]
    async fn mobius_history(
        &self,
        Parameters(params): Parameters<tools::HistoryParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_history(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Execute a single ML experiment with optional parameter overrides. Returns the experiment ID, metrics, and status. Each run costs budget."
    )]
    async fn mobius_run(
        &self,
        Parameters(params): Parameters<tools::RunParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_run(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Evaluate prediction results against ground truth using multi-dimensional scoring (detection F1, classification accuracy, timestamp precision). Returns bench score and per-dimension breakdown."
    )]
    async fn mobius_evaluate(
        &self,
        Parameters(params): Parameters<tools::EvaluateParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_evaluate(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Get the next experiment configuration suggestion from the gradient-guided tuning strategy. Uses accumulated learning signals to propose the most promising parameter change."
    )]
    async fn mobius_suggest(
        &self,
        Parameters(params): Parameters<tools::SuggestParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_suggest(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Run a cartesian product parameter sweep across multiple parameter values. Returns all results ranked by primary metric. Use for systematic exploration."
    )]
    async fn mobius_sweep(
        &self,
        Parameters(params): Parameters<tools::SweepParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_sweep(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Start the autonomous experiment agent loop. The agent iterates through ORIENT-PROPOSE-EXECUTE-EVALUATE-LEARN-DECIDE cycles until a stop condition (targets met, budget exhausted, max iterations). Long-running operation."
    )]
    async fn mobius_agent(
        &self,
        Parameters(params): Parameters<tools::AgentParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_agent(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Find Pareto-optimal experiments across two metrics. Returns experiments that are not dominated on both metrics simultaneously."
    )]
    async fn mobius_pareto(
        &self,
        Parameters(params): Parameters<tools::ParetoParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_pareto(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Check or modify the experiment budget. Returns current spend, limit, remaining budget, and estimated experiments remaining."
    )]
    async fn mobius_budget(
        &self,
        Parameters(params): Parameters<tools::BudgetParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_budget(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        description = "Query the learning store for gradient signals on parameters. Shows which parameters have been explored, their impact direction, and confidence level. Use to understand what the system has learned."
    )]
    async fn mobius_learnings(
        &self,
        Parameters(params): Parameters<tools::LearningsParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = tools::handle_learnings(&self.state, params).await?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for MobiusServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_instructions(
            "Mobius MCP server — drive ML experiments, evaluate results, \
             get parameter suggestions, and run autonomous optimization loops."
                .to_string(),
        )
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(resources::list_resources())
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        resources::read_resource(&self.state, request).await
    }
}
