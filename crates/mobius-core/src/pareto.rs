use crate::experiment::ExperimentResult;

/// Find Pareto-optimal experiments on two metrics.
///
/// Mirrors `experiment_history.py:get_pareto_front`.
pub fn pareto_front<'a>(
    results: &'a [ExperimentResult],
    x_metric: &str,
    y_metric: &str,
) -> Vec<&'a ExperimentResult> {
    let mut candidates: Vec<&ExperimentResult> = results
        .iter()
        .filter(|r| r.metrics.contains_key(x_metric) && r.metrics.contains_key(y_metric))
        .collect();

    candidates.sort_by(|a, b| {
        let ax = a.metrics.get(x_metric).unwrap_or(&0.0);
        let bx = b.metrics.get(x_metric).unwrap_or(&0.0);
        bx.partial_cmp(ax).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut front = Vec::new();
    let mut best_y = f64::NEG_INFINITY;

    for result in &candidates {
        let y = *result.metrics.get(y_metric).unwrap_or(&0.0);
        if y > best_y {
            front.push(*result);
            best_y = y;
        }
    }

    front
}
