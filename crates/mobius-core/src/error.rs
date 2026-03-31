use serde::{Deserialize, Serialize};

/// Classifies errors from experiment execution.
///
/// Mirrors `pipeline_optimizer.py:_classify_error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
    pub retry_after_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    GpuOom,
    QuotaExceeded,
    Timeout,
    ResourceNotFound,
    MissingDependency,
    AuthError,
    Custom(String),
}

pub struct ErrorPattern {
    pub pattern: String,
    pub code: ErrorCode,
    pub retryable: bool,
    pub retry_after_secs: u64,
}

pub struct ErrorClassifier {
    patterns: Vec<ErrorPattern>,
}

impl ErrorClassifier {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                ErrorPattern {
                    pattern: "CUDA out of memory".into(),
                    code: ErrorCode::GpuOom,
                    retryable: true,
                    retry_after_secs: 60,
                },
                ErrorPattern {
                    pattern: "quota".into(),
                    code: ErrorCode::QuotaExceeded,
                    retryable: true,
                    retry_after_secs: 300,
                },
                ErrorPattern {
                    pattern: "TimeoutError".into(),
                    code: ErrorCode::Timeout,
                    retryable: true,
                    retry_after_secs: 0,
                },
                ErrorPattern {
                    pattern: "not found".into(),
                    code: ErrorCode::ResourceNotFound,
                    retryable: false,
                    retry_after_secs: 0,
                },
                ErrorPattern {
                    pattern: "ModuleNotFoundError".into(),
                    code: ErrorCode::MissingDependency,
                    retryable: false,
                    retry_after_secs: 0,
                },
            ],
        }
    }

    pub fn add_pattern(&mut self, pattern: ErrorPattern) {
        self.patterns.push(pattern);
    }

    pub fn classify(&self, stderr: &str, stdout: &str) -> Option<ErrorInfo> {
        let combined = format!("{}\n{}", stderr, stdout);
        let lower = combined.to_lowercase();

        for pat in &self.patterns {
            if lower.contains(&pat.pattern.to_lowercase()) {
                return Some(ErrorInfo {
                    code: pat.code.clone(),
                    message: combined
                        .lines()
                        .find(|l| l.to_lowercase().contains(&pat.pattern.to_lowercase()))
                        .unwrap_or("")
                        .chars()
                        .take(200)
                        .collect(),
                    retryable: pat.retryable,
                    retry_after_secs: pat.retry_after_secs,
                });
            }
        }
        None
    }
}

impl Default for ErrorClassifier {
    fn default() -> Self {
        Self::new()
    }
}
