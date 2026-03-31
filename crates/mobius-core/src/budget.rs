use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Tracks experiment spending against a budget limit.
///
/// Mirrors `paloa-claw/.state/budget.json` + `post_experiment.py:track_cost`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetGuard {
    pub spent: f64,
    pub limit: f64,
    #[serde(skip)]
    state_path: Option<PathBuf>,
}

impl BudgetGuard {
    pub fn new(limit: f64) -> Self {
        Self {
            spent: 0.0,
            limit,
            state_path: None,
        }
    }

    pub fn with_state_file(mut self, path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let saved: BudgetGuard = serde_json::from_str(&content)?;
            self.spent = saved.spent;
            self.limit = saved.limit;
        }
        self.state_path = Some(path);
        Ok(self)
    }

    pub fn can_run(&self, cost: f64) -> bool {
        self.remaining() >= cost
    }

    pub fn remaining(&self) -> f64 {
        (self.limit - self.spent).max(0.0)
    }

    pub fn experiments_remaining(&self, cost_per_experiment: f64) -> usize {
        if cost_per_experiment <= 0.0 {
            return usize::MAX;
        }
        (self.remaining() / cost_per_experiment) as usize
    }

    pub fn record_spend(&mut self, amount: f64) -> anyhow::Result<()> {
        self.spent += amount;
        self.save()
    }

    fn save(&self) -> anyhow::Result<()> {
        if let Some(ref path) = self.state_path {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let content = serde_json::to_string_pretty(self)?;
            fs::write(path, content)?;
        }
        Ok(())
    }
}
