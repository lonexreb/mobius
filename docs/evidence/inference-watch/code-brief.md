# Inference brief for code workloads

**Recorded pilot choice: target-only.**

Apple M4 Max (64 GB), llama.cpp build 7480 on Metal, Qwen2.5 7B target and 1.5B draft.
The reference scores 1.0×. Each configuration was evaluated twice on the same eight development prompts.
The table reports identical outputs per recording and the smaller paired throughput ratio. A draft must match every token in both recordings to qualify.

| Draft setting | Exact outputs | Paired speedup (minimum) | Outcome | Source runs |
|---|---|---|---|---|
| d2-p50 | 3/8 and 3/8 | 0.830× | fails exactness | run_e2a20ea730, run_9216c17f6c |
| d4-p50 | 0/8 and 0/8 | 0.748× | fails exactness | run_407203c958, run_82fdad5a91 |
| d4-p75 | 1/8 and 1/8 | 0.733× | fails exactness | run_9fee8ebda7, run_c1cad73cf4 |

This brief reuses existing measurements. Fresh hardware, model or engine versions need a new test.
Pilot data SHA-256: 0a46e2bf930b65aee4ca4a0e8bb2014ff6b213c62fe452cf3fe125aeff625d6f
