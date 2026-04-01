---
name: agent-design
description: Agent loop design patterns for Mobius. Use when working on AgentLoop, strategies, hooks, or swarm coordination.
---

# Agent Design Patterns

## The 7-Step Loop

```
AgentLoop.step():
  1. ORIENT   — load history, find best, check budget
  2. RESEARCH — (Phase 3: MCP tools for external context)
  3. PROPOSE  — Strategy.suggest(ctx) -> Suggestion
  4. EXECUTE  — PreHooks -> ComputeBackend.submit() -> RawOutput
  5. EVALUATE — OutputParser -> PostHooks -> ExperimentResult
  6. LEARN    — extract_learning() -> LearningStore
  7. DECIDE   — Continue / SwitchStrategy / Stop
```

## Strategy Interface

```rust
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn suggest(&self, ctx: &StrategyContext) -> Result<Suggestion>;
}
```

StrategyContext provides: history, production_config, sweep_space, targets, learning_store.

## GradientGuidedTuning Algorithm

1. < 2 history -> random exploration
2. Plateau -> explore untried dims, qualitative changes, expand range
3. Normal -> rank params by |avg_metric_delta|, move top in best_direction
4. Dedup against history

## Decision Matrix

| Condition | Decision |
|-----------|----------|
| All targets met | Stop(TargetsMet) |
| Budget < cost_per_run | Stop(BudgetExhausted) |
| iteration >= max | Stop(MaxIterationsReached) |
| 3 consecutive reverts | SwitchStrategy |
| Otherwise | Continue |

## Hook System

```rust
pub enum HookAction { Proceed, Warn(String), Block(String) }
```

- BudgetCheckHook (pre) — blocks if insufficient
- RegressionDetectionHook (post) — warns if metric drops > threshold
- OverfittingDetectionHook (post) — warns if segment spread > threshold

## Learning System

ParamGradient tracks per-parameter signals:
- num_observations, avg_metric_delta, best_direction, confidence (Low/Medium/High)

## Strategy Phases

ParameterTuning -> AlgorithmTuning -> StructuralChanges -> Custom

## Future: Swarm Coordination

Lead-teammate model, shared LearningStore, config deduplication across agents.

## Inspired By

- Karpathy AutoResearch (gradient loops)
- AIDE/WecoAI (tree search)
- Meta REA (production orchestration)
- NemoClaw (sandboxed execution)
- Claude Code (tool-use + permission system)
