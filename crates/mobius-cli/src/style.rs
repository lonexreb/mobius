//! Color and formatting helpers for CLI output.
//!
//! Uses `owo-colors` for cross-platform terminal coloring with
//! consistent styling across all Mobius CLI commands.

use owo_colors::OwoColorize;

/// Format a metric name-value pair: bold name, green value.
pub fn metric(name: &str, value: f64) -> String {
    format!("{}: {}", name.bold(), format!("{:.4}", value).green())
}

/// Format a delta value: green for positive improvement, red for negative.
pub fn improvement(delta: f64) -> String {
    if delta >= 0.0 {
        format!("+{:.4}", delta).green().to_string()
    } else {
        format!("{:.4}", delta).red().to_string()
    }
}

/// Format a target status indicator: green "MET" or red "MISS".
pub fn target_status(met: bool) -> String {
    if met {
        "MET".green().bold().to_string()
    } else {
        "MISS".red().bold().to_string()
    }
}

/// Format a section header with bold title and "=" separator line.
pub fn section(title: &str) -> String {
    let separator = "=".repeat(60);
    format!("{}\n  {}\n{}", separator, title.bold(), separator)
}

/// Format a warning message in yellow.
pub fn warn(msg: &str) -> String {
    msg.yellow().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_format() {
        let output = metric("f1", 0.85);
        assert!(output.contains("f1"));
        assert!(output.contains("0.8500"));
    }

    #[test]
    fn test_improvement_positive() {
        let output = improvement(0.05);
        assert!(output.contains("+0.0500"));
    }

    #[test]
    fn test_improvement_negative() {
        let output = improvement(-0.03);
        assert!(output.contains("-0.0300"));
    }

    #[test]
    fn test_target_status_met() {
        let output = target_status(true);
        assert!(output.contains("MET"));
    }

    #[test]
    fn test_target_status_miss() {
        let output = target_status(false);
        assert!(output.contains("MISS"));
    }

    #[test]
    fn test_section_contains_title() {
        let output = section("TEST SECTION");
        assert!(output.contains("TEST SECTION"));
        assert!(output.contains("===="));
    }

    #[test]
    fn test_warn_contains_message() {
        let output = warn("something went wrong");
        assert!(output.contains("something went wrong"));
    }
}
