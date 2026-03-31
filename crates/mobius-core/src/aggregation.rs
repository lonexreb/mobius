use std::collections::HashMap;

/// Aggregated metrics across multiple segments.
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub macro_avg: HashMap<String, f64>,
    pub micro_f1: f64,
    pub total_tp: usize,
    pub total_fp: usize,
    pub total_fn: usize,
    pub num_segments: usize,
}

/// Compute macro-average of metrics across segments (mean of per-segment values).
pub fn macro_average(
    segments: &HashMap<String, HashMap<String, f64>>,
    metrics: &[&str],
) -> HashMap<String, f64> {
    let mut result = HashMap::new();
    for metric in metrics {
        let values: Vec<f64> = segments
            .values()
            .filter_map(|seg| seg.get(*metric).copied())
            .collect();
        if !values.is_empty() {
            let avg = values.iter().sum::<f64>() / values.len() as f64;
            result.insert(metric.to_string(), (avg * 10000.0).round() / 10000.0);
        }
    }
    result
}

/// Compute micro-average F1 by aggregating TP/FP/FN across segments.
pub fn micro_average(segments: &HashMap<String, HashMap<String, f64>>) -> AggregateMetrics {
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;
    let mut num = 0usize;

    for seg in segments.values() {
        let tp = seg.get("true_positives").copied().unwrap_or(0.0) as usize;
        let fp = seg.get("false_positives").copied().unwrap_or(0.0) as usize;
        let r#fn = seg.get("false_negatives").copied().unwrap_or(0.0) as usize;
        total_tp += tp;
        total_fp += fp;
        total_fn += r#fn;
        num += 1;
    }

    let p = if total_tp + total_fp > 0 {
        total_tp as f64 / (total_tp + total_fp) as f64
    } else {
        0.0
    };
    let r = if total_tp + total_fn > 0 {
        total_tp as f64 / (total_tp + total_fn) as f64
    } else {
        0.0
    };
    let micro_f1 = if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    };

    let macro_avg = macro_average(segments, &["f1", "precision", "recall"]);

    AggregateMetrics {
        macro_avg,
        micro_f1: (micro_f1 * 10000.0).round() / 10000.0,
        total_tp,
        total_fp,
        total_fn,
        num_segments: num,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segments() -> HashMap<String, HashMap<String, f64>> {
        let mut segments = HashMap::new();

        let mut s1 = HashMap::new();
        s1.insert("f1".into(), 0.80);
        s1.insert("precision".into(), 0.85);
        s1.insert("recall".into(), 0.76);
        s1.insert("true_positives".into(), 10.0);
        s1.insert("false_positives".into(), 2.0);
        s1.insert("false_negatives".into(), 3.0);
        segments.insert("seg1".into(), s1);

        let mut s2 = HashMap::new();
        s2.insert("f1".into(), 0.70);
        s2.insert("precision".into(), 0.75);
        s2.insert("recall".into(), 0.66);
        s2.insert("true_positives".into(), 8.0);
        s2.insert("false_positives".into(), 3.0);
        s2.insert("false_negatives".into(), 4.0);
        segments.insert("seg2".into(), s2);

        segments
    }

    #[test]
    fn test_macro_average() {
        let segments = make_segments();
        let avg = macro_average(&segments, &["f1", "precision", "recall"]);
        assert!((avg["f1"] - 0.75).abs() < 0.01);
        assert!((avg["precision"] - 0.80).abs() < 0.01);
    }

    #[test]
    fn test_micro_average() {
        let segments = make_segments();
        let agg = micro_average(&segments);
        assert_eq!(agg.total_tp, 18);
        assert_eq!(agg.total_fp, 5);
        assert_eq!(agg.total_fn, 7);
        assert_eq!(agg.num_segments, 2);
        // micro_p = 18/23 ≈ 0.7826, micro_r = 18/25 = 0.72
        // micro_f1 = 2 * 0.7826 * 0.72 / (0.7826 + 0.72) ≈ 0.7499
        assert!(agg.micro_f1 > 0.74 && agg.micro_f1 < 0.76);
    }

    #[test]
    fn test_empty_segments() {
        let segments: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let avg = macro_average(&segments, &["f1"]);
        assert!(avg.is_empty());
    }
}
