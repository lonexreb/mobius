use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single labeled ground-truth datum. Domain-agnostic.
///
/// In paloa_bench this was `GroundTruthShot` with fields like time_sec, jersey, team.
/// In Mobius, domain-specific fields go in `attributes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    pub id: String,
    /// Temporal position (seconds). Optional for non-temporal domains.
    pub timestamp: Option<f64>,
    /// Primary label (e.g., "make", "miss", "positive", "negative").
    pub label: String,
    /// Domain-specific fields (e.g., jersey number, team, shot type).
    pub attributes: HashMap<String, serde_json::Value>,
    /// Edge case tags for breakdown analysis (e.g., "free_throw", "fast_break").
    #[serde(default)]
    pub tags: Vec<String>,
}

/// A single prediction from a pipeline.
///
/// Generalizes paloa_bench's `PredictedShot`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub timestamp: Option<f64>,
    pub label: Option<String>,
    pub confidence: Option<f64>,
    /// Which detector/model produced this prediction.
    pub source: Option<String>,
    pub attributes: HashMap<String, serde_json::Value>,
}

/// A matched pair: one prediction aligned to one ground truth datum.
#[derive(Debug, Clone)]
pub struct MatchedPair {
    pub ground_truth: GroundTruth,
    pub prediction: Prediction,
    /// Temporal distance between prediction and ground truth (if applicable).
    pub time_delta: Option<f64>,
}

/// Result of matching predictions against ground truth.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub matched: Vec<MatchedPair>,
    pub false_positives: Vec<Prediction>,
    pub missed: Vec<GroundTruth>,
}

/// Score for a single evaluation dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    pub name: String,
    pub score: f64,
    pub weight: f64,
    pub weighted_score: f64,
    pub details: HashMap<String, serde_json::Value>,
}

/// Per-prediction quality flag (from paloa_bench's ShotFlag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Flag {
    /// Accurate — all fields correct.
    Green,
    /// Borderline — partial match, needs review.
    Yellow,
    /// Incorrect — wrong result or major mismatch.
    Red,
}

/// Full evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub bench_score: f64,
    pub grade: Grade,
    pub dimensions: Vec<DimensionScore>,
    pub match_count: usize,
    pub false_positive_count: usize,
    pub missed_count: usize,
    pub flags: HashMap<Flag, usize>,
    pub edge_case_breakdown: HashMap<String, EdgeCaseStats>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    Fail,      // < 50
    Pass,      // >= 50
    Good,      // >= 70
    Excellent, // >= 85
}

impl Grade {
    pub fn from_score(score: f64) -> Self {
        if score >= 85.0 {
            Grade::Excellent
        } else if score >= 70.0 {
            Grade::Good
        } else if score >= 50.0 {
            Grade::Pass
        } else {
            Grade::Fail
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeCaseStats {
    pub total: usize,
    pub detected: usize,
    pub recall: f64,
}
