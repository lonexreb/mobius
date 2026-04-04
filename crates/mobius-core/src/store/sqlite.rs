//! SQLite-backed experiment store.
//!
//! Production-grade alternative to `JsonlStore` with proper indexing,
//! atomic writes, and efficient queries for `get_best` and `get_recent`.

use crate::experiment::{ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentStore};
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Raw row tuple from SQLite: (id, timestamp, config, metrics, per_segment, duration, cost, status, error).
type RawRow = (
    String,
    String,
    String,
    String,
    String,
    f64,
    Option<f64>,
    String,
    Option<String>,
);

/// SQLite experiment storage backend.
///
/// Stores experiment results in a SQLite database with JSON columns for
/// nested data (parameters, metrics, per-segment). Provides indexed
/// lookups and atomic appends. Thread-safe via internal `Mutex`.
pub struct SqliteStore {
    path: PathBuf,
    conn: Mutex<Connection>,
}

impl SqliteStore {
    /// Open (or create) a SQLite store at the given path.
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=-64000;
             PRAGMA temp_store=MEMORY;
             CREATE TABLE IF NOT EXISTS experiments (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                per_segment TEXT NOT NULL DEFAULT '{}',
                duration_secs REAL NOT NULL,
                cost_usd REAL,
                status TEXT NOT NULL,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_id ON experiments(id);",
        )?;
        Ok(Self {
            path,
            conn: Mutex::new(conn),
        })
    }

    /// Return the database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Deserialize a raw row tuple into an `ExperimentResult`.
    fn parse_row(row: RawRow) -> anyhow::Result<ExperimentResult> {
        let timestamp = row.1.parse::<chrono::DateTime<chrono::Utc>>()?;
        let config: ExperimentConfig = serde_json::from_str(&row.2)?;
        let metrics: HashMap<String, f64> = serde_json::from_str(&row.3)?;
        let per_segment: HashMap<String, HashMap<String, f64>> = serde_json::from_str(&row.4)?;
        let status: ExperimentStatus = serde_json::from_str(&row.7)?;

        Ok(ExperimentResult {
            id: row.0,
            timestamp,
            config,
            metrics,
            per_segment,
            duration_secs: row.5,
            cost_usd: row.6,
            status,
            error: row.8,
        })
    }
}

impl ExperimentStore for SqliteStore {
    fn append(&mut self, result: &ExperimentResult) -> anyhow::Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut stmt = conn.prepare_cached(
            "INSERT INTO experiments (id, timestamp, config, metrics, per_segment, duration_secs, cost_usd, status, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        stmt.execute(params![
            result.id,
            result.timestamp.to_rfc3339(),
            serde_json::to_string(&result.config)?,
            serde_json::to_string(&result.metrics)?,
            serde_json::to_string(&result.per_segment)?,
            result.duration_secs,
            result.cost_usd,
            serde_json::to_string(&result.status)?,
            result.error,
        ])?;
        Ok(())
    }

    fn load_all(&self) -> anyhow::Result<Vec<ExperimentResult>> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut stmt = conn.prepare_cached(
            "SELECT id, timestamp, config, metrics, per_segment, duration_secs, cost_usd, status, error
             FROM experiments ORDER BY rowid",
        )?;
        let mut results = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, Option<f64>>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, Option<String>>(8)?,
            ))
        })?;
        for row in rows {
            results.push(Self::parse_row(row?)?);
        }
        Ok(results)
    }

    fn get_best(&self, metric: &str) -> anyhow::Result<Option<ExperimentResult>> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let json_path = format!("$.{metric}");
        let mut stmt = conn.prepare_cached(
            "SELECT id, timestamp, config, metrics, per_segment, duration_secs, cost_usd, status, error
             FROM experiments
             WHERE json_extract(metrics, ?1) IS NOT NULL
             ORDER BY CAST(json_extract(metrics, ?1) AS REAL) DESC
             LIMIT 1",
        )?;
        let mut rows = stmt.query_map(params![json_path], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, Option<f64>>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, Option<String>>(8)?,
            ))
        })?;
        match rows.next() {
            Some(row) => Ok(Some(Self::parse_row(row?)?)),
            None => Ok(None),
        }
    }

    fn get_recent(&self, n: usize) -> anyhow::Result<Vec<ExperimentResult>> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut stmt = conn.prepare_cached(
            "SELECT id, timestamp, config, metrics, per_segment, duration_secs, cost_usd, status, error
             FROM experiments ORDER BY rowid DESC LIMIT ?1",
        )?;
        let mut results = Vec::new();
        let rows = stmt.query_map(params![n as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, Option<f64>>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, Option<String>>(8)?,
            ))
        })?;
        for row in rows {
            results.push(Self::parse_row(row?)?);
        }
        // Reverse since we queried DESC to get most recent
        results.reverse();
        Ok(results)
    }

    fn count(&self) -> anyhow::Result<usize> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM experiments", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_result(id: &str, f1: f64) -> ExperimentResult {
        let mut metrics = HashMap::new();
        metrics.insert("f1".to_string(), f1);
        ExperimentResult {
            id: id.to_string(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("lr".to_string(), serde_json::json!(0.01));
                    p
                },
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
    fn test_sqlite_append_and_load() {
        let file = NamedTempFile::new().unwrap();
        let mut store = SqliteStore::new(file.path()).unwrap();

        store.append(&make_result("exp-001", 0.75)).unwrap();
        store.append(&make_result("exp-002", 0.82)).unwrap();

        let all = store.load_all().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, "exp-001");
        assert_eq!(all[1].id, "exp-002");
    }

    #[test]
    fn test_sqlite_get_best() {
        let file = NamedTempFile::new().unwrap();
        let mut store = SqliteStore::new(file.path()).unwrap();

        store.append(&make_result("exp-001", 0.75)).unwrap();
        store.append(&make_result("exp-002", 0.82)).unwrap();
        store.append(&make_result("exp-003", 0.78)).unwrap();

        let best = store.get_best("f1").unwrap().unwrap();
        assert_eq!(best.id, "exp-002");
    }

    #[test]
    fn test_sqlite_get_recent() {
        let file = NamedTempFile::new().unwrap();
        let mut store = SqliteStore::new(file.path()).unwrap();

        for i in 0..10 {
            store
                .append(&make_result(&format!("exp-{i:03}"), 0.5 + i as f64 * 0.01))
                .unwrap();
        }

        let recent = store.get_recent(3).unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].id, "exp-007");
        assert_eq!(recent[2].id, "exp-009");
    }

    #[test]
    fn test_sqlite_count() {
        let file = NamedTempFile::new().unwrap();
        let mut store = SqliteStore::new(file.path()).unwrap();

        assert_eq!(store.count().unwrap(), 0);
        store.append(&make_result("exp-001", 0.75)).unwrap();
        assert_eq!(store.count().unwrap(), 1);
        store.append(&make_result("exp-002", 0.82)).unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn test_sqlite_empty_store() {
        let file = NamedTempFile::new().unwrap();
        let store = SqliteStore::new(file.path()).unwrap();

        assert_eq!(store.load_all().unwrap().len(), 0);
        assert!(store.get_best("f1").unwrap().is_none());
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_sqlite_preserves_all_fields() {
        let file = NamedTempFile::new().unwrap();
        let mut store = SqliteStore::new(file.path()).unwrap();

        let mut per_seg = HashMap::new();
        let mut seg_metrics = HashMap::new();
        seg_metrics.insert("f1".to_string(), 0.80);
        per_seg.insert("segment_a".to_string(), seg_metrics);

        let result = ExperimentResult {
            id: "exp-full".to_string(),
            timestamp: chrono::Utc::now(),
            config: ExperimentConfig {
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("lr".to_string(), serde_json::json!(0.05));
                    p.insert("bs".to_string(), serde_json::json!(32));
                    p
                },
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("note".to_string(), serde_json::json!("test run"));
                    m
                },
            },
            metrics: {
                let mut m = HashMap::new();
                m.insert("f1".to_string(), 0.85);
                m.insert("precision".to_string(), 0.90);
                m
            },
            per_segment: per_seg,
            duration_secs: 42.5,
            cost_usd: Some(2.50),
            status: ExperimentStatus::Error,
            error: Some("OOM".to_string()),
        };

        store.append(&result).unwrap();
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);

        let r = &loaded[0];
        assert_eq!(r.id, "exp-full");
        assert_eq!(r.config.parameters["lr"], serde_json::json!(0.05));
        assert_eq!(r.config.parameters["bs"], serde_json::json!(32));
        assert_eq!(r.config.metadata["note"], serde_json::json!("test run"));
        assert_eq!(r.metrics["f1"], 0.85);
        assert_eq!(r.metrics["precision"], 0.90);
        assert_eq!(r.per_segment["segment_a"]["f1"], 0.80);
        assert_eq!(r.duration_secs, 42.5);
        assert_eq!(r.cost_usd, Some(2.50));
        assert_eq!(r.status, ExperimentStatus::Error);
        assert_eq!(r.error, Some("OOM".to_string()));
    }
}
