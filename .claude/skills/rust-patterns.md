---
name: rust-patterns
description: Rust coding patterns for the Mobius framework. Use when writing new scorers, strategies, hooks, or CLI commands.
---

# Rust Patterns for Mobius

## Trait Object Pattern

Mobius uses `Box<dyn Trait>` for runtime polymorphism:

```rust
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn suggest(&self, ctx: &StrategyContext) -> anyhow::Result<Suggestion>;
}

// In AgentLoop:
strategy: Box<dyn Strategy>,
compute: Box<dyn ComputeBackend>,
experiment_store: Box<dyn ExperimentStore>,
```

## Error Handling

- Library code: `anyhow::Result<T>` with `?` propagation
- Structured errors: `thiserror` derive macro
- Never `.unwrap()` in library code

## Serde Conventions

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub id: String,
    #[serde(default)]        // absent in older data
    pub per_segment: HashMap<String, HashMap<String, f64>>,
    #[serde(skip)]           // transient, not serialized
    state_path: Option<PathBuf>,
}
```

## File Paths

Always `impl AsRef<Path>`:
```rust
pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self>
```

## HashMap as Generic Container

`HashMap<String, serde_json::Value>` for domain-agnostic data:
- `ExperimentConfig::parameters`
- `GroundTruth::attributes`
- `DimensionScore::details`

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_descriptive_name() {
        let file = NamedTempFile::new().unwrap();
        let mut store = JsonlStore::new(file.path()).unwrap();
        store.append(&make_result("exp-1", 0.75)).unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }
}
```
