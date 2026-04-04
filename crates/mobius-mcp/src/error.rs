//! Error conversion helpers for MCP responses.

use rmcp::model::ErrorData;

/// Convert any error into an MCP internal error.
pub fn to_mcp_error(err: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(format!("{err}"), None)
}

/// Create an MCP error for invalid tool parameters.
pub fn invalid_params(msg: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(msg.into(), None)
}
