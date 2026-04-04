//! NSGA-II multi-objective optimization strategy.
//!
//! Implements non-dominated sorting, crowding distance, and tournament
//! selection to explore the Pareto front across multiple objectives.

use crate::strategy::{RandomSearch, Strategy, StrategyContext, Suggestion, config_already_tried};
use mobius_core::experiment::{ExperimentConfig, ExperimentResult};
use rand::prelude::IndexedRandom;
use rayon::prelude::*;
use std::collections::HashMap;

/// NSGA-II multi-objective optimization strategy.
///
/// Maintains internal population state across `suggest()` calls. Each call
/// evolves the population by one step and returns the best untried candidate.
pub struct NsgaTwo {
    /// Objective metric names (all maximized).
    pub objectives: Vec<String>,
    /// Population size per generation.
    pub population_size: usize,
}

impl NsgaTwo {
    pub fn new(objectives: Vec<String>, population_size: usize) -> Self {
        Self {
            objectives,
            population_size,
        }
    }

    /// Non-dominated sorting: returns fronts as vectors of indices into `results`.
    /// Front 0 = Pareto-optimal, Front 1 = next layer, etc.
    fn non_dominated_sort(&self, results: &[ExperimentResult]) -> Vec<Vec<usize>> {
        let n = results.len();
        if n == 0 {
            return vec![];
        }

        let mut domination_count = vec![0usize; n];
        let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];
        let mut fronts: Vec<Vec<usize>> = vec![];

        for i in 0..n {
            for j in (i + 1)..n {
                match self.dominance(&results[i], &results[j]) {
                    Dominance::Left => {
                        dominated_by[i].push(j);
                        domination_count[j] += 1;
                    }
                    Dominance::Right => {
                        dominated_by[j].push(i);
                        domination_count[i] += 1;
                    }
                    Dominance::Neither => {}
                }
            }
        }

        // Front 0: non-dominated solutions
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front.is_empty() {
            let mut next_front = vec![];
            for &i in &current_front {
                for &j in &dominated_by[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            fronts.push(current_front);
            current_front = next_front;
        }

        fronts
    }

    /// Check dominance between two solutions across all objectives (all maximized).
    fn dominance(&self, a: &ExperimentResult, b: &ExperimentResult) -> Dominance {
        let mut a_better = false;
        let mut b_better = false;

        for obj in &self.objectives {
            let va = a.metrics.get(obj).copied().unwrap_or(0.0);
            let vb = b.metrics.get(obj).copied().unwrap_or(0.0);
            if va > vb {
                a_better = true;
            } else if vb > va {
                b_better = true;
            }
        }

        if a_better && !b_better {
            Dominance::Left
        } else if b_better && !a_better {
            Dominance::Right
        } else {
            Dominance::Neither
        }
    }

    /// Crowding distance for solutions within a front.
    ///
    /// Per-objective contributions computed in parallel via rayon, then summed.
    fn crowding_distance(&self, front: &[usize], results: &[ExperimentResult]) -> Vec<f64> {
        let n = front.len();
        if n <= 2 {
            return vec![f64::INFINITY; n];
        }

        // Compute per-objective distance contributions in parallel
        let per_obj: Vec<Vec<f64>> = self
            .objectives
            .par_iter()
            .map(|obj| {
                let mut obj_dist = vec![0.0f64; n];

                let mut sorted_indices: Vec<usize> = (0..n).collect();
                sorted_indices.sort_by(|&a, &b| {
                    let va = results[front[a]].metrics.get(obj).copied().unwrap_or(0.0);
                    let vb = results[front[b]].metrics.get(obj).copied().unwrap_or(0.0);
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });

                obj_dist[sorted_indices[0]] = f64::INFINITY;
                obj_dist[sorted_indices[n - 1]] = f64::INFINITY;

                let min_val = results[front[sorted_indices[0]]]
                    .metrics
                    .get(obj)
                    .copied()
                    .unwrap_or(0.0);
                let max_val = results[front[sorted_indices[n - 1]]]
                    .metrics
                    .get(obj)
                    .copied()
                    .unwrap_or(0.0);
                let range = max_val - min_val;
                if range >= 1e-10 {
                    for i in 1..(n - 1) {
                        let prev = results[front[sorted_indices[i - 1]]]
                            .metrics
                            .get(obj)
                            .copied()
                            .unwrap_or(0.0);
                        let next = results[front[sorted_indices[i + 1]]]
                            .metrics
                            .get(obj)
                            .copied()
                            .unwrap_or(0.0);
                        obj_dist[sorted_indices[i]] += (next - prev) / range;
                    }
                }

                obj_dist
            })
            .collect();

        // Sum per-objective contributions
        (0..n)
            .map(|i| {
                let sum: f64 = per_obj.iter().map(|d| d[i]).sum();
                if per_obj.iter().any(|d| d[i].is_infinite()) {
                    f64::INFINITY
                } else {
                    sum
                }
            })
            .collect()
    }

    /// Generate a new candidate by crossover of two parents from the best front.
    fn generate_candidate(
        &self,
        results: &[ExperimentResult],
        fronts: &[Vec<usize>],
        sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut rng = rand::rng();

        // Select from best available front
        let front = if !fronts.is_empty() {
            &fronts[0]
        } else {
            // No fronts — random
            return self.random_config(sweep_space);
        };

        if front.len() < 2 {
            return self.random_config(sweep_space);
        }

        // Crowding-distance-weighted tournament selection
        let distances = self.crowding_distance(front, results);
        let parent1 = self.tournament(front, &distances);
        let parent2 = self.tournament(front, &distances);

        let p1 = &results[parent1].config.parameters;
        let p2 = &results[parent2].config.parameters;

        // Uniform crossover + mutation
        let mut child = HashMap::new();
        for (param, values) in sweep_space {
            let base = if rand::random::<bool>() {
                p1.get(param)
            } else {
                p2.get(param)
            };

            // 10% mutation: pick random value instead
            if rand::random::<f64>() < 0.1
                && let Some(val) = values.choose(&mut rng)
            {
                child.insert(param.clone(), val.clone());
                continue;
            }

            if let Some(val) = base {
                child.insert(param.clone(), val.clone());
            } else if let Some(val) = values.first() {
                child.insert(param.clone(), val.clone());
            }
        }

        child
    }

    fn tournament(&self, front: &[usize], distances: &[f64]) -> usize {
        let mut rng = rand::rng();
        let candidates: Vec<usize> = (0..front.len()).collect();
        let &a = candidates.choose(&mut rng).unwrap_or(&0);
        let &b = candidates.choose(&mut rng).unwrap_or(&0);
        // Prefer higher crowding distance
        if distances[a] >= distances[b] {
            front[a]
        } else {
            front[b]
        }
    }

    fn random_config(
        &self,
        sweep_space: &HashMap<String, Vec<serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut rng = rand::rng();
        sweep_space
            .iter()
            .filter_map(|(k, v)| v.choose(&mut rng).map(|val| (k.clone(), val.clone())))
            .collect()
    }
}

enum Dominance {
    Left,
    Right,
    Neither,
}

impl Strategy for NsgaTwo {
    fn name(&self) -> &str {
        "nsga2"
    }

    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion> {
        // Need enough history to form a population
        if ctx.history.len() < 4 {
            return RandomSearch.suggest(ctx);
        }

        let fronts = self.non_dominated_sort(ctx.history);

        // Generate candidates and pick the first untried one
        for _ in 0..self.population_size {
            let candidate = self.generate_candidate(ctx.history, &fronts, ctx.sweep_space);
            if !config_already_tried(&candidate, ctx.history) {
                let mut changed = HashMap::new();
                for (k, v) in &candidate {
                    if ctx.production_config.get(k) != Some(v) {
                        changed.insert(k.clone(), v.clone());
                    }
                }
                return Ok(Suggestion {
                    config: ExperimentConfig {
                        parameters: candidate,
                        metadata: HashMap::new(),
                    },
                    changed_params: changed,
                    rationale: format!(
                        "NSGA-II: multi-objective selection across {} objectives ({} fronts, {} history)",
                        self.objectives.len(),
                        fronts.len(),
                        ctx.history.len()
                    ),
                });
            }
        }

        // All candidates tried — fall back to random
        RandomSearch.suggest(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mobius_core::experiment::{ExperimentConfig, ExperimentStatus};

    fn make_result(id: &str, f1: f64, precision: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".into(), f1);
        metrics.insert("precision".into(), precision);
        ExperimentResult {
            id: id.into(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("x".into(), serde_json::json!(f1));
                    p
                },
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

    #[test]
    fn test_non_dominated_sort_basic() {
        let nsga = NsgaTwo::new(vec!["f1".into(), "precision".into()], 20);
        let results = vec![
            make_result("a", 0.9, 0.7), // Pareto-optimal (high f1)
            make_result("b", 0.7, 0.9), // Pareto-optimal (high precision)
            make_result("c", 0.5, 0.5), // Dominated by both a and b
        ];

        let fronts = nsga.non_dominated_sort(&results);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 2); // a and b are on front 0
        assert_eq!(fronts[1].len(), 1); // c is on front 1
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let nsga = NsgaTwo::new(vec!["f1".into(), "precision".into()], 20);
        let results = vec![
            make_result("a", 0.9, 0.5),
            make_result("b", 0.7, 0.7),
            make_result("c", 0.5, 0.9),
        ];
        let front = vec![0, 1, 2];
        let distances = nsga.crowding_distance(&front, &results);
        // Boundary points should have infinite distance
        assert!(distances[0].is_infinite() || distances[2].is_infinite());
    }

    #[test]
    fn test_nsga2_suggest_with_history() {
        use crate::learning_store::LearningStore;
        use tempfile::NamedTempFile;

        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert(
            "x".into(),
            vec![
                serde_json::json!(0.1),
                serde_json::json!(0.3),
                serde_json::json!(0.5),
                serde_json::json!(0.7),
                serde_json::json!(0.9),
            ],
        );

        let history = vec![
            make_result("a", 0.9, 0.5),
            make_result("b", 0.7, 0.7),
            make_result("c", 0.5, 0.9),
            make_result("d", 0.3, 0.3),
        ];

        let ctx = StrategyContext {
            history: &history,
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let nsga = NsgaTwo::new(vec!["f1".into(), "precision".into()], 20);
        let suggestion = nsga.suggest(&ctx).unwrap();
        assert!(!suggestion.config.parameters.is_empty());
    }

    #[test]
    fn test_nsga2_insufficient_history() {
        use crate::learning_store::LearningStore;
        use tempfile::NamedTempFile;

        let file = NamedTempFile::new().unwrap();
        let store = LearningStore::new(file.path()).unwrap();

        let mut sweep = HashMap::new();
        sweep.insert("x".into(), vec![serde_json::json!(1.0)]);

        let ctx = StrategyContext {
            history: &[make_result("a", 0.5, 0.5)],
            production_config: &HashMap::new(),
            sweep_space: &sweep,
            targets: &HashMap::new(),
            primary_metric: "f1",
            learning_store: &store,
        };

        let nsga = NsgaTwo::new(vec!["f1".into(), "precision".into()], 20);
        let suggestion = nsga.suggest(&ctx).unwrap();
        assert!(suggestion.rationale.contains("Random"));
    }
}
