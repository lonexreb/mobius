use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

/// Raw output from a compute backend execution.
#[derive(Debug, Clone)]
pub struct RawOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration_secs: f64,
}

/// Parsed metrics extracted from pipeline output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedMetrics {
    pub metrics: HashMap<String, f64>,
    pub bench_score: Option<f64>,
    pub dimension_scores: HashMap<String, f64>,
    pub true_positives: Option<usize>,
    pub false_positives: Option<usize>,
    pub false_negatives: Option<usize>,
}

/// Trait for compute backends that execute experiments.
pub trait ComputeBackend: Send + Sync {
    fn submit(
        &self,
        command: &str,
        env: &HashMap<String, String>,
        timeout_secs: u64,
    ) -> anyhow::Result<RawOutput>;
}

/// Async trait for compute backends that perform I/O (HTTP, SSH, cloud APIs).
///
/// For local subprocess execution, wrap `SubprocessBackend` in a [`SyncAdapter`].
#[async_trait]
pub trait AsyncComputeBackend: Send + Sync {
    async fn submit(
        &self,
        command: &str,
        env: &HashMap<String, String>,
        timeout_secs: u64,
    ) -> anyhow::Result<RawOutput>;
}

/// Wraps a synchronous [`ComputeBackend`] for use in async contexts.
///
/// Delegates to `tokio::task::spawn_blocking` for non-blocking execution.
pub struct SyncAdapter<T: ComputeBackend + Clone + 'static> {
    inner: T,
}

impl<T: ComputeBackend + Clone + 'static> SyncAdapter<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<T: ComputeBackend + Clone + 'static> AsyncComputeBackend for SyncAdapter<T> {
    async fn submit(
        &self,
        command: &str,
        env: &HashMap<String, String>,
        timeout_secs: u64,
    ) -> anyhow::Result<RawOutput> {
        let backend = self.inner.clone();
        let cmd = command.to_string();
        let env = env.clone();
        tokio::task::spawn_blocking(move || backend.submit(&cmd, &env, timeout_secs)).await?
    }
}

/// Runs multiple experiments concurrently using an inner async backend.
pub struct ParallelBackend {
    inner: Arc<dyn AsyncComputeBackend>,
    max_concurrency: usize,
}

impl ParallelBackend {
    pub fn new(inner: Arc<dyn AsyncComputeBackend>, max_concurrency: usize) -> Self {
        Self {
            inner,
            max_concurrency,
        }
    }

    /// Submit multiple commands concurrently, respecting `max_concurrency`.
    pub async fn submit_batch(
        &self,
        jobs: Vec<(String, HashMap<String, String>, u64)>,
    ) -> Vec<anyhow::Result<RawOutput>> {
        let sem = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        let mut handles = Vec::new();

        for (cmd, env, timeout) in jobs {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let backend = self.inner.clone();
            handles.push(tokio::spawn(async move {
                let result = backend.submit(&cmd, &env, timeout).await;
                drop(permit);
                result
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(
                handle
                    .await
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("Task panicked: {e}"))),
            );
        }
        results
    }
}

/// Executes experiments as local subprocesses.
#[derive(Clone)]
pub struct SubprocessBackend;

impl ComputeBackend for SubprocessBackend {
    fn submit(
        &self,
        command: &str,
        env: &HashMap<String, String>,
        timeout_secs: u64,
    ) -> anyhow::Result<RawOutput> {
        let start = Instant::now();

        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            anyhow::bail!("Empty command");
        }

        let mut cmd = Command::new(parts[0]);
        if parts.len() > 1 {
            cmd.args(&parts[1..]);
        }
        for (k, v) in env {
            cmd.env(k, v);
        }

        let output = cmd
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute command '{}': {}", parts[0], e))?;

        let duration = start.elapsed().as_secs_f64();

        if duration > timeout_secs as f64 {
            anyhow::bail!("Command exceeded timeout of {}s", timeout_secs);
        }

        Ok(RawOutput {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            duration_secs: duration,
        })
    }
}

/// Extracts metrics from pipeline output text.
pub struct OutputParser;

impl OutputParser {
    /// Parse stdout/stderr for metrics. Tries JSON first, then regex patterns.
    pub fn parse(stdout: &str, stderr: &str) -> ParsedMetrics {
        // Try JSON block first (reverse scan for last JSON with "f1")
        if let Some(m) = Self::try_json(stdout) {
            return m;
        }
        // Fall back to regex-based extraction
        Self::parse_bench_format(stdout, stderr)
    }

    fn try_json(stdout: &str) -> Option<ParsedMetrics> {
        for line in stdout.lines().rev() {
            let trimmed = line.trim();
            if trimmed.starts_with('{')
                && trimmed.contains("\"f1\"")
                && let Ok(map) = serde_json::from_str::<HashMap<String, serde_json::Value>>(trimmed)
            {
                let mut metrics = HashMap::new();
                let mut pm = ParsedMetrics::default();

                for (k, v) in &map {
                    if let Some(n) = v.as_f64() {
                        metrics.insert(k.clone(), n);
                    }
                }

                pm.bench_score = metrics.remove("bench_score");
                if let Some(tp) = metrics.remove("true_positives") {
                    pm.true_positives = Some(tp as usize);
                }
                if let Some(fp) = metrics.remove("false_positives") {
                    pm.false_positives = Some(fp as usize);
                }
                if let Some(r#fn) = metrics.remove("false_negatives") {
                    pm.false_negatives = Some(r#fn as usize);
                }
                pm.metrics = metrics;
                return Some(pm);
            }
        }
        None
    }

    fn parse_bench_format(stdout: &str, _stderr: &str) -> ParsedMetrics {
        let mut pm = ParsedMetrics::default();

        let bench_re = Regex::new(r"BENCH SCORE:\s*([\d.]+)").unwrap();
        let dim_re =
            Regex::new(r"(\w[\w\s&/]+?)\s+([\d.]+)\s+[x×]\s+([\d.]+)\s+=\s+([\d.]+)").unwrap();
        let match_re = Regex::new(r"Matches:\s*(\d+).*FPs:\s*(\d+).*Missed:\s*(\d+)").unwrap();
        let mm_re = Regex::new(r"Make/miss:.*?(\d+)/(\d+).*?([\d.]+)%").unwrap();

        for line in stdout.lines() {
            let line = line.trim();

            if let Some(cap) = bench_re.captures(line)
                && let Ok(score) = cap[1].parse::<f64>()
            {
                pm.bench_score = Some(score);
            }

            if let Some(cap) = dim_re.captures(line) {
                let name = cap[1].trim().to_lowercase().replace([' ', '/'], "_");
                if let Ok(score) = cap[2].parse::<f64>() {
                    pm.dimension_scores.insert(name, score);
                }
            }

            if let Some(cap) = match_re.captures(line) {
                let tp: usize = cap[1].parse().unwrap_or(0);
                let fp: usize = cap[2].parse().unwrap_or(0);
                let r#fn: usize = cap[3].parse().unwrap_or(0);
                pm.true_positives = Some(tp);
                pm.false_positives = Some(fp);
                pm.false_negatives = Some(r#fn);

                if tp > 0 {
                    let p = tp as f64 / (tp + fp) as f64;
                    let r = tp as f64 / (tp + r#fn) as f64;
                    let f1 = if p + r > 0.0 {
                        2.0 * p * r / (p + r)
                    } else {
                        0.0
                    };
                    pm.metrics
                        .insert("precision".into(), (p * 10000.0).round() / 10000.0);
                    pm.metrics
                        .insert("recall".into(), (r * 10000.0).round() / 10000.0);
                    pm.metrics
                        .insert("f1".into(), (f1 * 10000.0).round() / 10000.0);
                }
            }

            if let Some(cap) = mm_re.captures(line)
                && let Ok(acc) = cap[3].parse::<f64>()
            {
                pm.metrics.insert("make_miss_accuracy".into(), acc / 100.0);
            }
        }

        pm
    }
}

/// Map experiment config parameters to environment variables.
pub fn config_to_env(
    params: &HashMap<String, serde_json::Value>,
    env_map: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut env = HashMap::new();
    for (param, env_var) in env_map {
        if let Some(val) = params.get(param) {
            let s = match val {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                other => other.to_string(),
            };
            env.insert(env_var.clone(), s);
        }
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_format() {
        let stdout = r#"
Some pipeline output...
{"f1": 0.786, "precision": 0.733, "recall": 0.846, "bench_score": 62.3}
Done.
"#;
        let result = OutputParser::parse(stdout, "");
        assert!((result.metrics["f1"] - 0.786).abs() < 0.001);
        assert!((result.metrics["precision"] - 0.733).abs() < 0.001);
        assert!((result.bench_score.unwrap() - 62.3).abs() < 0.1);
    }

    #[test]
    fn test_parse_bench_score_format() {
        let stdout = r#"
PALOA BENCH EVALUATION
  BENCH SCORE: 47.7 / 100  (FAIL)
  Shot Detection         59.3 × 0.40 =  23.7  (PASS)
  Make/Miss              62.5 × 0.25 =  15.6  (PASS)
  Matches: 8 | FPs: 6 | Missed: 5
  Make/miss:        6/11 (55%)
"#;
        let result = OutputParser::parse(stdout, "");
        assert!((result.bench_score.unwrap() - 47.7).abs() < 0.1);
        assert_eq!(result.true_positives, Some(8));
        assert_eq!(result.false_positives, Some(6));
        assert_eq!(result.false_negatives, Some(5));
        assert!(result.metrics.contains_key("f1"));
        assert!(result.metrics.contains_key("precision"));
        assert!(result.metrics.contains_key("recall"));
        assert!((result.metrics["make_miss_accuracy"] - 0.55).abs() < 0.01);
        assert!(result.dimension_scores.contains_key("shot_detection"));
    }

    #[test]
    fn test_parse_empty_output() {
        let result = OutputParser::parse("", "");
        assert!(result.metrics.is_empty());
        assert!(result.bench_score.is_none());
    }

    #[test]
    fn test_config_to_env() {
        let mut params = HashMap::new();
        params.insert("ball_conf".into(), serde_json::json!(0.25));
        params.insert("merge_strategy".into(), serde_json::json!("yolo_kimi"));

        let mut env_map = HashMap::new();
        env_map.insert("ball_conf".into(), "PALOA_BALL_CONF".into());
        env_map.insert("merge_strategy".into(), "PALOA_MERGE_STRATEGY".into());

        let env = config_to_env(&params, &env_map);
        assert_eq!(env["PALOA_BALL_CONF"], "0.25");
        assert_eq!(env["PALOA_MERGE_STRATEGY"], "yolo_kimi");
    }

    #[test]
    fn test_subprocess_echo() {
        let backend = SubprocessBackend;
        let result = backend.submit("echo hello", &HashMap::new(), 10).unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_sync_adapter_echo() {
        let adapter = SyncAdapter::new(SubprocessBackend);
        let result = adapter
            .submit("echo async_hello", &HashMap::new(), 10)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("async_hello"));
    }

    #[tokio::test]
    async fn test_parallel_batch() {
        let adapter = Arc::new(SyncAdapter::new(SubprocessBackend));
        let parallel = ParallelBackend::new(adapter, 2);

        let jobs = vec![
            ("echo job1".into(), HashMap::new(), 10),
            ("echo job2".into(), HashMap::new(), 10),
            ("echo job3".into(), HashMap::new(), 10),
        ];

        let results = parallel.submit_batch(jobs).await;
        assert_eq!(results.len(), 3);
        for (i, r) in results.iter().enumerate() {
            let output = r.as_ref().unwrap();
            assert_eq!(output.exit_code, 0);
            assert!(output.stdout.contains(&format!("job{}", i + 1)));
        }
    }
}
