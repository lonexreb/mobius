use crate::schema::{GroundTruth, Prediction};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Load predictions from a JSON file, auto-detecting format.
///
/// Supported formats:
/// - Bare array: `[{timestamp, label, ...}, ...]`
/// - Wrapped: `{"shots": [...]}`, `{"confirmed": [...]}`, `{"predictions": [...]}`,
///   `{"detected_shots": [...]}`, `{"merged_shots": [...]}`
pub fn load_predictions(path: &Path) -> anyhow::Result<Vec<Prediction>> {
    let content = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&content)?;

    let arr = if let Some(arr) = value.as_array() {
        arr.clone()
    } else if let Some(obj) = value.as_object() {
        let keys = [
            "shots",
            "confirmed",
            "predictions",
            "detected_shots",
            "merged_shots",
            "detections",
        ];
        let mut found = None;
        for key in &keys {
            if let Some(arr) = obj.get(*key).and_then(|v| v.as_array()) {
                found = Some(arr.clone());
                break;
            }
        }
        found.ok_or_else(|| anyhow::anyhow!("No recognized prediction array in JSON"))?
    } else {
        anyhow::bail!("Expected JSON array or object");
    };

    let predictions: Vec<Prediction> = arr
        .into_iter()
        .filter_map(|v| adapt_prediction(&v))
        .collect();

    Ok(predictions)
}

/// Load ground truth from a JSON file, auto-detecting format.
///
/// Maps common field names: `time_sec`→`timestamp`, `result`→`label`.
pub fn load_ground_truth(path: &Path) -> anyhow::Result<Vec<GroundTruth>> {
    let content = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&content)?;

    let arr = if let Some(arr) = value.as_array() {
        arr.clone()
    } else if let Some(obj) = value.as_object() {
        if let Some(arr) = obj.get("shots").and_then(|v| v.as_array()) {
            arr.clone()
        } else {
            anyhow::bail!("No 'shots' array in ground truth JSON");
        }
    } else {
        anyhow::bail!("Expected JSON array or object");
    };

    let ground_truth: Vec<GroundTruth> = arr
        .into_iter()
        .enumerate()
        .filter_map(|(i, v)| adapt_ground_truth(&v, i))
        .collect();

    Ok(ground_truth)
}

/// List available game datasets in a directory.
pub fn list_games(gt_dir: &Path) -> anyhow::Result<Vec<String>> {
    let mut games = Vec::new();
    if !gt_dir.exists() {
        return Ok(games);
    }
    for entry in fs::read_dir(gt_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            games.push(stem.to_string());
        }
    }
    games.sort();
    Ok(games)
}

fn adapt_prediction(v: &serde_json::Value) -> Option<Prediction> {
    let obj = v.as_object()?;

    let timestamp = obj
        .get("timestamp")
        .or_else(|| obj.get("time_sec"))
        .or_else(|| obj.get("time"))
        .and_then(|v| v.as_f64());

    let label = obj
        .get("label")
        .or_else(|| obj.get("result"))
        .or_else(|| obj.get("shot_result"))
        .and_then(|v| v.as_str())
        .map(String::from);

    let confidence = obj.get("confidence").and_then(|v| v.as_f64());
    let source = obj.get("source").and_then(|v| v.as_str()).map(String::from);

    let mut attributes = HashMap::new();
    for key in ["shot_type", "shooter_jersey", "shooter_team", "jersey", "team"] {
        if let Some(val) = obj.get(key) {
            attributes.insert(key.to_string(), val.clone());
        }
    }

    Some(Prediction {
        timestamp,
        label,
        confidence,
        source,
        attributes,
    })
}

fn adapt_ground_truth(v: &serde_json::Value, index: usize) -> Option<GroundTruth> {
    let obj = v.as_object()?;

    let id = obj
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| format!("gt-{}", index));

    let timestamp = obj
        .get("timestamp")
        .or_else(|| obj.get("time_sec"))
        .or_else(|| obj.get("time"))
        .and_then(|v| v.as_f64());

    let label = obj
        .get("label")
        .or_else(|| obj.get("result"))
        .or_else(|| obj.get("shot_result"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let tags = obj
        .get("edge_case_tags")
        .or_else(|| obj.get("tags"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let mut attributes = HashMap::new();
    for key in [
        "shot_type",
        "shooter_jersey",
        "shooter_team",
        "jersey",
        "team",
    ] {
        if let Some(val) = obj.get(key) {
            attributes.insert(key.to_string(), val.clone());
        }
    }

    Some(GroundTruth {
        id,
        timestamp,
        label,
        attributes,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_predictions_bare_array() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"[{{"timestamp": 10.5, "result": "make"}}, {{"timestamp": 20.0, "result": "miss"}}]"#
        )
        .unwrap();
        let preds = load_predictions(f.path()).unwrap();
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].label.as_deref(), Some("make"));
        assert!((preds[0].timestamp.unwrap() - 10.5).abs() < 0.01);
    }

    #[test]
    fn test_load_predictions_wrapped() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"{{"shots": [{{"time_sec": 5.0, "label": "make"}}]}}"#
        )
        .unwrap();
        let preds = load_predictions(f.path()).unwrap();
        assert_eq!(preds.len(), 1);
        assert!((preds[0].timestamp.unwrap() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_load_ground_truth() {
        let mut f = NamedTempFile::new().unwrap();
        write!(
            f,
            r#"[{{"time_sec": 10.0, "result": "MAKE", "shot_type": "2pt", "edge_case_tags": ["free_throw"]}}]"#
        )
        .unwrap();
        let gt = load_ground_truth(f.path()).unwrap();
        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0].label, "MAKE");
        assert_eq!(gt[0].tags, vec!["free_throw"]);
        assert!(gt[0].attributes.contains_key("shot_type"));
    }
}
