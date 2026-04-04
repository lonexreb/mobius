# MCP Server Setup

Mobius exposes all experiment tools through the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), allowing AI assistants to run experiments, analyze results, and optimize parameters autonomously.

## Claude Code

Add to your project's `.mcp.json` (or `~/.claude/mcp_servers.json` for global access):

```json
{
  "mcpServers": {
    "mobius": {
      "command": "cargo",
      "args": ["run", "--release", "-p", "mobius-mcp"],
      "cwd": "/path/to/mobius"
    }
  }
}
```

Or if installed via `cargo install`:

```json
{
  "mcpServers": {
    "mobius": {
      "command": "mobius-mcp"
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `mobius_status` | Current project status: experiment count, target gaps, best config, budget |
| `mobius_history` | Recent experiment results with metrics and config |
| `mobius_run` | Execute a single experiment with optional parameter overrides |
| `mobius_evaluate` | Score predictions against ground truth (multi-dimensional) |
| `mobius_suggest` | Get next config suggestion using gradient-guided or other strategy |
| `mobius_sweep` | Run cartesian parameter sweep across configs |
| `mobius_agent` | Start autonomous experiment loop with budget and hooks |
| `mobius_pareto` | Pareto front analysis across two metrics |
| `mobius_budget` | Read or modify budget (set limit, record spend) |
| `mobius_learnings` | Query gradient signals and untried dimensions |

## Available Resources

| URI | Description |
|-----|-------------|
| `mobius://config` | Project configuration from `mobius.toml` |
| `mobius://budget` | Current budget status (spent, limit, remaining) |
| `mobius://best` | Best experiment result with metrics and config |

## Requirements

- A `mobius.toml` file must exist in the working directory for most tools
- State is stored in `~/.mobius/` (history, learnings, budget)
- The MCP server communicates over stdio (stdout = JSON-RPC, stderr = logs)

## Troubleshooting

**No output / tools not responding:**
- Ensure `mobius.toml` exists in the directory where the MCP server runs
- Check stderr for errors: `RUST_LOG=debug cargo run -p mobius-mcp 2>mcp.log`

**Tools return "No mobius.toml found":**
- The server's working directory must contain `mobius.toml`
- Use the `cwd` field in `.mcp.json` to set the correct directory

**Budget issues:**
- Budget state persists in `~/.mobius/budget.json`
- Use `mobius_budget` with `set_limit` to reset
