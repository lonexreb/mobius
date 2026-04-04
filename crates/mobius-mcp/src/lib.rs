//! MCP server and client integration for Mobius.
//!
//! Exposes Mobius tools via Model Context Protocol so any MCP-compatible
//! agent (Claude Code, Copilot, custom) can drive experiments, evaluate
//! results, get parameter suggestions, and run autonomous optimization loops.

pub mod error;
pub mod resources;
pub mod server;
pub mod state;
pub mod tools;
