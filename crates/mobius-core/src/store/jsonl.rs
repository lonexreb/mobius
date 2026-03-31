use crate::experiment::{ExperimentResult, ExperimentStore};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Append-only JSONL experiment store.
///
/// Mirrors `~/.paloa/optimizer/history.jsonl` from experiment_history.py.
/// Each line is a JSON-serialized ExperimentResult.
pub struct JsonlStore {
    path: PathBuf,
}

impl JsonlStore {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ExperimentStore for JsonlStore {
    fn append(&mut self, result: &ExperimentResult) -> anyhow::Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(result)?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    fn load_all(&self) -> anyhow::Result<Vec<ExperimentResult>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut results = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<ExperimentResult>(trimmed) {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::warn!("Skipping malformed history line: {}", e);
                }
            }
        }
        Ok(results)
    }

    fn get_best(&self, metric: &str) -> anyhow::Result<Option<ExperimentResult>> {
        let all = self.load_all()?;
        Ok(all
            .into_iter()
            .filter(|r| r.metrics.contains_key(metric))
            .max_by(|a, b| {
                let va = a.metrics.get(metric).unwrap_or(&0.0);
                let vb = b.metrics.get(metric).unwrap_or(&0.0);
                va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal)
            }))
    }

    fn get_recent(&self, n: usize) -> anyhow::Result<Vec<ExperimentResult>> {
        let all = self.load_all()?;
        let start = all.len().saturating_sub(n);
        Ok(all[start..].to_vec())
    }

    fn count(&self) -> anyhow::Result<usize> {
        Ok(self.load_all()?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{ExperimentConfig, ExperimentStatus};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    fn make_result(id: &str, f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".to_string(), f1);
        ExperimentResult {
            id: id.to_string(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: HashMap::new(),
                metadata: HashMap::new(),
            },
            metrics,
            per_segment: HashMap::new(),
            duration_secs: 10.0,
            cost_usd: Some(1.20),
            status: ExperimentStatus::Success,
            error: None,
        }
    }

    #[test]
    fn test_append_and_load() {
        let file = NamedTempFile::new().unwrap();
        let mut store = JsonlStore::new(file.path()).unwrap();

        store.append(&make_result("exp-001", 0.75)).unwrap();
        store.append(&make_result("exp-002", 0.82)).unwrap();

        let all = store.load_all().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, "exp-001");
        assert_eq!(all[1].id, "exp-002");
    }

    #[test]
    fn test_get_best() {
        let file = NamedTempFile::new().unwrap();
        let mut store = JsonlStore::new(file.path()).unwrap();

        store.append(&make_result("exp-001", 0.75)).unwrap();
        store.append(&make_result("exp-002", 0.82)).unwrap();
        store.append(&make_result("exp-003", 0.78)).unwrap();

        let best = store.get_best("f1").unwrap().unwrap();
        assert_eq!(best.id, "exp-002");
    }

    #[test]
    fn test_get_recent() {
        let file = NamedTempFile::new().unwrap();
        let mut store = JsonlStore::new(file.path()).unwrap();

        for i in 0..10 {
            store
                .append(&make_result(&format!("exp-{:03}", i), 0.5 + i as f64 * 0.01))
                .unwrap();
        }

        let recent = store.get_recent(3).unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].id, "exp-007");
    }

    #[test]
    fn test_empty_store() {
        let file = NamedTempFile::new().unwrap();
        let store = JsonlStore::new(file.path()).unwrap();

        assert_eq!(store.load_all().unwrap().len(), 0);
        assert!(store.get_best("f1").unwrap().is_none());
        assert_eq!(store.count().unwrap(), 0);
    }
}
