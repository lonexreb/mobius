//! Shared mutable state for the MCP server.
//!
//! All tool handlers share access to experiment storage, learning signals,
//! budget tracking, and project configuration through `Arc<RwLock<SharedState>>`.

use mobius_claw::learning_store::LearningStore;
use mobius_core::budget::BudgetGuard;
use mobius_core::config::MobiusConfig;
use mobius_core::store::JsonlStore;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type alias for the shared state handle passed to all tool handlers.
pub type State = Arc<RwLock<SharedState>>;

/// Mutable server state shared across all MCP tool and resource handlers.
///
/// Holds experiment storage, learning signals, budget tracking, and
/// the optional project configuration loaded from `mobius.toml`.
pub struct SharedState {
    /// Project configuration (absent if no `mobius.toml` found).
    pub config: Option<MobiusConfig>,
    /// Experiment result store (`~/.mobius/history.jsonl`).
    pub store: JsonlStore,
    /// Learning signal store (`~/.mobius/learnings.jsonl`).
    pub learning_store: LearningStore,
    /// Budget tracker (`~/.mobius/budget.json`).
    pub budget: BudgetGuard,
    /// Root directory for Mobius state files.
    pub mobius_dir: PathBuf,
}

impl fmt::Debug for SharedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedState")
            .field("config", &self.config.is_some())
            .field("budget", &self.budget)
            .field("mobius_dir", &self.mobius_dir)
            .finish_non_exhaustive()
    }
}

impl SharedState {
    /// Initialize shared state from disk.
    ///
    /// Creates `~/.mobius/` if it does not exist. Loads `mobius.toml` from the
    /// current directory if present (returns `config: None` otherwise).
    pub fn new() -> anyhow::Result<State> {
        let mobius_dir = dirs::home_dir().unwrap_or_default().join(".mobius");
        Self::with_dir(mobius_dir, MobiusConfig::load("mobius.toml").ok())
    }

    /// Initialize shared state rooted in a custom directory.
    ///
    /// Creates `mobius_dir` if it does not exist. Useful for tests,
    /// embeddings, and deployments where `~/.mobius/` is not appropriate.
    pub fn with_dir(mobius_dir: PathBuf, config: Option<MobiusConfig>) -> anyhow::Result<State> {
        std::fs::create_dir_all(&mobius_dir)?;

        let store = JsonlStore::new(mobius_dir.join("history.jsonl"))?;
        let learning_store = LearningStore::new(mobius_dir.join("learnings.jsonl"))?;

        let budget_path = mobius_dir.join("budget.json");
        let budget = if budget_path.exists() {
            BudgetGuard::new(20.0).with_state_file(&budget_path)?
        } else {
            BudgetGuard::new(20.0)
        };

        Ok(Arc::new(RwLock::new(SharedState {
            config,
            store,
            learning_store,
            budget,
            mobius_dir,
        })))
    }
}
