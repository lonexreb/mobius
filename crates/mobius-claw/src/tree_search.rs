//! UCB1 Tree-Search strategy for AIDE-style hypothesis exploration.
//!
//! Models the parameter sweep space as a tree where each level corresponds
//! to a parameter and each branch to a value choice. Uses the UCB1 formula
//! to balance exploration vs exploitation when traversing the tree.

use crate::strategy::{RandomSearch, Strategy, StrategyContext, Suggestion, config_already_tried};
use mobius_core::experiment::ExperimentConfig;
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

/// A node in the UCB1 search tree.
///
/// Each node tracks visit count and cumulative reward for computing UCB1 scores.
/// Children are keyed by the string representation of parameter values.
struct TreeNode {
    /// Number of experiments routed through this node.
    visits: usize,
    /// Sum of primary metric values from experiments through this node.
    total_reward: f64,
    /// Children keyed by parameter value (JSON string representation).
    children: HashMap<String, TreeNode>,
}

impl TreeNode {
    fn new() -> Self {
        Self {
            visits: 0,
            total_reward: 0.0,
            children: HashMap::new(),
        }
    }

    /// UCB1 score: exploitation + exploration bonus.
    ///
    /// Formula: Q(n)/N(n) + C * sqrt(ln(N_parent) / N(n))
    fn ucb1(&self, parent_visits: usize, c: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.total_reward / self.visits as f64;
        let exploration = c * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration
    }
}

/// UCB1 Tree-Search strategy.
///
/// Models the parameter sweep space as a tree where each level represents
/// a parameter and each branch a value choice. On each `suggest()` call,
/// rebuilds the tree from experiment history and walks from root to leaf
/// using UCB1 selection at each level.
///
/// Inspired by AIDE/WecoAI's tree-search approach to hypothesis exploration.
pub struct UcbTreeSearch {
    /// UCB1 exploration constant (default: sqrt(2)).
    pub exploration_constant: f64,
}

impl UcbTreeSearch {
    /// Create a new UCB1 tree search with the given exploration constant.
    pub fn new(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }

    /// Reconstruct the search tree from experiment history.
    ///
    /// Each experiment traces a path through the tree (one level per parameter).
    /// Visit counts and rewards are accumulated along the path.
    fn build_tree(
        &self,
        history: &[mobius_core::experiment::ExperimentResult],
        param_order: &[String],
        primary_metric: &str,
    ) -> TreeNode {
        let mut root = TreeNode::new();

        for result in history {
            let reward = result.metrics.get(primary_metric).copied().unwrap_or(0.0);
            let mut node = &mut root;
            node.visits += 1;
            node.total_reward += reward;

            for param in param_order {
                let value_key = result
                    .config
                    .parameters
                    .get(param)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "null".to_string());

                let child = node.children.entry(value_key).or_insert_with(TreeNode::new);
                child.visits += 1;
                child.total_reward += reward;
                node = child;
            }
        }

        root
    }

    /// Walk the tree from root to leaf using UCB1 selection at each level.
    ///
    /// When an unexplored node is reached (no children), remaining parameters
    /// are filled randomly from the sweep space.
    fn select_path(
        &self,
        root: &TreeNode,
        param_order: &[String],
        sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut rng = rand::rng();
        let mut config = HashMap::new();
        let mut node = root;
        let mut hit_unexplored = false;

        for param in param_order {
            let values = match sweep_space.get(param) {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };

            if hit_unexplored {
                if let Some(val) = values.choose(&mut rng) {
                    config.insert(param.clone(), val.clone());
                }
                continue;
            }

            let parent_visits = node.visits.max(1);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_value = values[0].clone();

            for value in values {
                let key = value.to_string();
                let score = match node.children.get(&key) {
                    Some(child) => child.ucb1(parent_visits, self.exploration_constant),
                    None => f64::INFINITY,
                };
                if score > best_score {
                    best_score = score;
                    best_value = value.clone();
                }
            }

            config.insert(param.clone(), best_value.clone());

            let key = best_value.to_string();
            match node.children.get(&key) {
                Some(child) => node = child,
                None => hit_unexplored = true,
            }
        }

        config
    }
}

impl Strategy for UcbTreeSearch {
    fn name(&self) -> &str {
        "ucb1"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        if ctx.history.len() < 2 {
            return RandomSearch.suggest(ctx);
        }

        // Deterministic parameter ordering for consistent tree structure
        let mut param_order: Vec<String> = ctx.sweep_space.keys().cloned().collect();
        param_order.sort();

        let root = self.build_tree(ctx.history, &param_order, ctx.primary_metric);
        let candidate = self.select_path(&root, &param_order, ctx.sweep_space);

        if config_already_tried(&candidate, ctx.history) {
            return RandomSearch.suggest(ctx);
        }

        let mut changed = HashMap::new();
        for (k, v) in &candidate {
            if ctx.production_config.get(k) != Some(v) {
                changed.insert(k.clone(), v.clone());
            }
        }

        Ok(Suggestion {
            config: ExperimentConfig {
                parameters: candidate,
                metadata: HashMap::new(),
            },
            changed_params: changed,
            rationale: format!(
                "UCB1 tree search: {} params, {} history, C={:.3}",
                param_order.len(),
                ctx.history.len(),
                self.exploration_constant
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_store::LearningStore;
    use crate::strategy::StrategyContext;
    use mobius_core::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus};
    use tempfile::NamedTempFile;

    fn make_result(id: &str, f1: f64, params: Vec<(&str, f64)>) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        let mut parameters = HashMap::new();
        for (k, v) in params {
            parameters.insert(k.to_string(), serde_json::json!(v));
        }
        ExperimentResult {
            id: id.into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters,
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 1.0,
            cost_usd: None,
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    fn sweep_space() -> HashMap<String, Vec<serde_json::Value>> {
        let mut space = HashMap::new();
        space.insert(
            "lr".into(),
            vec![
                serde_json::json!(0.01),
                serde_json::json!(0.05),
                serde_json::json!(0.10),
            ],
        );
        space.insert(
            "bs".into(),
            vec![serde_json::json!(16.0), serde_json::json!(32.0)],
        );
        space
    }

    #[test]
    fn test_ucb1_insufficient_history_fallback() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let ctx = StrategyContext {
            history: &[make_result("1", 0.5, vec![("lr", 0.01), ("bs", 16.0)])],
            production_config: &HashMap::new(),
            sweep_space: &sweep_space(),
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let ucb = UcbTreeSearch::new(std::f64::consts::SQRT_2);
        let suggestion = ucb.suggest(&ctx).unwrap();
        assert!(suggestion.rationale.contains("Random"));
    }

    #[test]
    fn test_ucb1_explores_unvisited_branches() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        // All history uses lr=0.01 — UCB1 should explore other lr values
        let history = vec![
            make_result("1", 0.5, vec![("lr", 0.01), ("bs", 16.0)]),
            make_result("2", 0.6, vec![("lr", 0.01), ("bs", 32.0)]),
        ];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep_space(),
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let ucb = UcbTreeSearch::new(std::f64::consts::SQRT_2);
        let suggestion = ucb.suggest(&ctx).unwrap();

        // Should prefer unvisited lr values (0.05 or 0.10) since they have infinite UCB1
        let lr = suggestion.config.parameters["lr"].as_f64().unwrap();
        assert!(
            (lr - 0.05).abs() < f64::EPSILON || (lr - 0.10).abs() < f64::EPSILON,
            "Expected unvisited lr (0.05 or 0.10), got {lr}"
        );
    }

    #[test]
    fn test_ucb1_prefers_high_reward_branch() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        // 3 parameters: sorted order is ["bs", "lr", "wd"]
        // bs=32 branch has high reward, bs=16 has low reward.
        // Under bs=32, lr is visited but wd has untried values,
        // so UCB1 can find an untried config in the high-reward branch.
        let mut space = sweep_space(); // lr=[0.01,0.05,0.10], bs=[16,32]
        space.insert(
            "wd".into(),
            vec![serde_json::json!(0.001), serde_json::json!(0.01)],
        );

        let history = vec![
            // bs=16 branch: low reward
            make_result("1", 0.3, vec![("bs", 16.0), ("lr", 0.01), ("wd", 0.001)]),
            make_result("2", 0.2, vec![("bs", 16.0), ("lr", 0.05), ("wd", 0.001)]),
            // bs=32 branch: high reward
            make_result("3", 0.9, vec![("bs", 32.0), ("lr", 0.05), ("wd", 0.001)]),
            make_result("4", 0.8, vec![("bs", 32.0), ("lr", 0.01), ("wd", 0.001)]),
        ];

        let ucb = UcbTreeSearch::new(0.1);

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &space,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let suggestion = ucb.suggest(&ctx).unwrap();
        let bs = suggestion.config.parameters["bs"].as_f64().unwrap();
        // With low C, exploitation dominates — should pick bs=32 (high-reward branch)
        assert!(
            (bs - 32.0).abs() < f64::EPSILON,
            "Expected bs=32 (best reward branch), got {bs}"
        );
    }

    #[test]
    fn test_ucb1_tree_node_scores() {
        let mut node = TreeNode::new();
        node.visits = 10;
        node.total_reward = 7.0;

        // Exploitation: 7/10 = 0.7
        // Exploration: sqrt(2) * sqrt(ln(100) / 10) ≈ 1.414 * sqrt(0.4605) ≈ 0.96
        let score = node.ucb1(100, std::f64::consts::SQRT_2);
        assert!(score > 0.7, "Score should exceed exploitation term");
        assert!(score < 2.0, "Score should be reasonable");

        // Unvisited node gets infinity
        let unvisited = TreeNode::new();
        assert!(unvisited.ucb1(100, 1.0).is_infinite());
    }

    #[test]
    fn test_ucb1_returns_valid_suggestion() {
        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let history = vec![
            make_result("1", 0.5, vec![("lr", 0.01), ("bs", 16.0)]),
            make_result("2", 0.7, vec![("lr", 0.05), ("bs", 32.0)]),
        ];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep_space(),
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let ucb = UcbTreeSearch::new(std::f64::consts::SQRT_2);
        let suggestion = ucb.suggest(&ctx).unwrap();

        assert!(suggestion.config.parameters.contains_key("lr"));
        assert!(suggestion.config.parameters.contains_key("bs"));
        assert!(suggestion.rationale.contains("UCB1"));
    }

    #[test]
    fn test_build_strategy_ucb1() {
        let s = crate::strategy::build_strategy("ucb1", 5, 0.02).unwrap();
        assert_eq!(s.name(), "ucb1");

        let s2 = crate::strategy::build_strategy("tree_search", 5, 0.02).unwrap();
        assert_eq!(s2.name(), "ucb1");
    }
}
