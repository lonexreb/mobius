use crate::matcher::Matcher;
use crate::scorers::DimensionScorer;
use mobius_core::config::DimensionConfig;
use mobius_core::schema::{BenchResult, Grade, GroundTruth, Prediction};

/// Multi-dimension evaluation engine.
///
/// Generalizes `paloa_bench/evaluator.py:BenchEvaluator`.
pub struct Evaluator {
    matcher: Box<dyn Matcher>,
    scorers: Vec<(DimensionConfig, Box<dyn DimensionScorer>)>,
    match_window: f64,
}

impl Evaluator {
    pub fn new(matcher: Box<dyn Matcher>, match_window: f64) -> Self {
        Self {
            matcher,
            scorers: Vec::new(),
            match_window,
        }
    }

    pub fn add_dimension(&mut self, config: DimensionConfig, scorer: Box<dyn DimensionScorer>) {
        self.scorers.push((config, scorer));
    }

    pub fn evaluate(
        &self,
        predictions: &[Prediction],
        ground_truth: &[GroundTruth],
    ) -> BenchResult {
        let match_result =
            self.matcher
                .match_predictions(predictions, ground_truth, self.match_window);

        let mut dimensions = Vec::new();
        let mut total_weighted = 0.0;

        for (config, scorer) in &self.scorers {
            let mut dim_score = scorer.score(
                &match_result.matched,
                &match_result.false_positives,
                &match_result.missed,
            );
            dim_score.name = config.name.clone();
            dim_score.weight = config.weight;
            dim_score.weighted_score = dim_score.score * config.weight;
            total_weighted += dim_score.weighted_score;
            dimensions.push(dim_score);
        }

        let bench_score = total_weighted;
        let grade = Grade::from_score(bench_score);

        let mut flags = std::collections::HashMap::new();
        flags.insert(
            mobius_core::schema::Flag::Green,
            match_result.matched.len(),
        );
        flags.insert(
            mobius_core::schema::Flag::Red,
            match_result.false_positives.len() + match_result.missed.len(),
        );

        BenchResult {
            bench_score,
            grade,
            dimensions,
            match_count: match_result.matched.len(),
            false_positive_count: match_result.false_positives.len(),
            missed_count: match_result.missed.len(),
            flags,
            edge_case_breakdown: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}
