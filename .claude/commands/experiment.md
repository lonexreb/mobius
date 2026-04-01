# /experiment — Run and Manage Experiments

```bash
mobius run --config '{"learning_rate": 0.01}'    # Single experiment
mobius suggest                                    # Gradient-guided suggestion
mobius sweep --spec '{"lr": [0.01, 0.1]}'        # Parameter sweep
mobius agent --budget 20.0                        # Autonomous loop
mobius status                                     # Current best + KPI gaps
mobius history --last 10                          # Recent experiments
```

## Agent Loop (7 steps)

1. ORIENT — load history, best, budget
2. RESEARCH — (Phase 3: MCP tools)
3. PROPOSE — strategy.suggest()
4. EXECUTE — pre-hooks -> compute -> parse
5. EVALUATE — post-hooks (regression, overfitting)
6. LEARN — extract gradient signals
7. DECIDE — continue / switch / stop

## Stop Conditions

TargetsMet | BudgetExhausted | MaxIterationsReached | Plateau | UserInterrupted

## State Files

- `~/.mobius/history.jsonl` — experiment results
- `~/.mobius/learnings.jsonl` — gradient signals
- `~/.mobius/budget.json` — spend tracking
