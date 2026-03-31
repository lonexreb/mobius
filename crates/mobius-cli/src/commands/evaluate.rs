use mobius_bench::evaluator::Evaluator;
use mobius_bench::matcher::GreedyTimestampMatcher;
use mobius_bench::scorers::{ClassificationAccuracyScorer, DimensionScorer, F1Scorer, TimestampMaeScorer};
use mobius_core::config::{DimensionConfig, MobiusConfig};
use mobius_core::loader;
use std::path::Path;

pub fn run(predictions_path: &str, gt_path: &str) -> anyhow::Result<()> {
    let config = MobiusConfig::load("mobius.toml").ok();
    let match_window = config
        .as_ref()
        .map(|c| c.bench.match_window)
        .unwrap_or(6.0);

    let predictions = loader::load_predictions(Path::new(predictions_path))?;
    let ground_truth = loader::load_ground_truth(Path::new(gt_path))?;

    println!("Loaded {} predictions, {} ground truth entries", predictions.len(), ground_truth.len());

    let mut evaluator = Evaluator::new(Box::new(GreedyTimestampMatcher), match_window);

    // Use dimensions from config or defaults
    let dimensions = config
        .as_ref()
        .map(|c| c.bench.dimensions.clone())
        .filter(|d| !d.is_empty())
        .unwrap_or_else(default_dimensions);

    for dim in &dimensions {
        let scorer = build_scorer(&dim.scorer);
        evaluator.add_dimension(dim.clone(), scorer);
    }

    let result = evaluator.evaluate(&predictions, &ground_truth);

    // Print results
    println!("\n{:=<60}", "");
    println!("  BENCH SCORE: {:.1} / 100  ({:?})", result.bench_score, result.grade);
    println!("{:=<60}", "");

    for dim in &result.dimensions {
        println!(
            "  {:<24} {:>5.1} x {:.2} = {:>5.1}",
            dim.name, dim.score, dim.weight, dim.weighted_score
        );
    }

    println!();
    println!(
        "  Matches: {} | FPs: {} | Missed: {}",
        result.match_count, result.false_positive_count, result.missed_count
    );
    println!();

    Ok(())
}

fn default_dimensions() -> Vec<DimensionConfig> {
    vec![
        DimensionConfig {
            name: "Detection".into(),
            weight: 0.50,
            scorer: "f1".into(),
            options: Default::default(),
        },
        DimensionConfig {
            name: "Classification".into(),
            weight: 0.30,
            scorer: "accuracy".into(),
            options: Default::default(),
        },
        DimensionConfig {
            name: "Timestamp".into(),
            weight: 0.20,
            scorer: "timestamp_mae".into(),
            options: Default::default(),
        },
    ]
}

fn build_scorer(name: &str) -> Box<dyn DimensionScorer> {
    match name {
        "f1" => Box::new(F1Scorer),
        "accuracy" | "classification_accuracy" => {
            Box::new(ClassificationAccuracyScorer::on_primary_label())
        }
        "timestamp_mae" => Box::new(TimestampMaeScorer),
        _ => {
            eprintln!("Unknown scorer '{}', defaulting to F1", name);
            Box::new(F1Scorer)
        }
    }
}
