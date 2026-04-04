//! Binary entry point for the Mobius MCP server.
//!
//! Starts an MCP server over stdio that exposes Mobius experiment tools
//! and resources to any MCP-compatible client.

use anyhow::Result;
use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Tracing MUST go to stderr — stdout is the MCP JSON-RPC transport.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting mobius-mcp server");

    let state = mobius_mcp::state::SharedState::new()?;
    let server = mobius_mcp::server::MobiusServer::new(state);

    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .inspect_err(|e| tracing::error!("Server error: {:?}", e))?;

    service.waiting().await?;
    Ok(())
}
