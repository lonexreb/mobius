# Code Reviewer Agent

Reviews Mobius code changes for quality, correctness, and project standards.

## Checklist

### Rust Quality
- [ ] `cargo fmt --all --check` — no diffs
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — zero warnings
- [ ] `cargo test --workspace` — all green
- [ ] No `.unwrap()` in library code (only tests/CLI)
- [ ] Public types and trait methods have `///` doc comments
- [ ] Functions <= 80 lines
- [ ] `anyhow::Result` (app) or `thiserror` (library) for errors

### Architecture
- [ ] Crate boundaries respected (core has no internal deps)
- [ ] Trait-based extension pattern followed
- [ ] Workspace deps used (no inline version pins)
- [ ] No circular dependencies

### Testing
- [ ] New functionality has tests
- [ ] `tempfile` used for filesystem tests
- [ ] Edge cases covered (empty, boundary, error paths)
- [ ] Names: `test_<what_it_tests>`

## Commands

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
