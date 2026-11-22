# Speculative-decoding evaluation

Fixed, independent evaluator used by Möbius Lab. See [the runbook](../../../docs/MOBIUS_LAB_RUNBOOK.md).

| File | Role |
|------|------|
| `manifest.json` | Pinned search space, generation settings, local engine, flagship model revisions |
| `dataset_registry.json` | Source revisions, checksums, prompt IDs + hashes, split hashes (no prompt text) |
| `fetch_prompts.py` | Downloads the pinned prompts into `data/` (git-ignored) |
| `evaluator.py` | Runs target-only and speculative servers interleaved; token-ID equality; paired speedup + CI |
| `tests/` | Split integrity, allowlist, correctness and scoring tests |

```bash
uv run --python 3.12 --with pyarrow python fetch_prompts.py        # once
python3 evaluator.py --target-only --split dev --out base.json
python3 evaluator.py --config '{"lookahead_tokens":4,"draft_p_min":0.75}' --split dev --reference base.json --out cand.json
```

Splits: `dev` (shortest prompts, visible to the controller), `audit` (longer, counted verification),
`sealed` (longest; needs `--unseal-final`, never passed by the API, **not yet evaluated**).

What runs today is Qwen2.5-7B with a Qwen2.5-1.5B draft on llama.cpp/Metal. The Qwen3-8B + EAGLE-3 + vLLM
flagship is pinned in `manifest.json` but has not been run; there is no `VllmEngine` yet because its flags
and token-ID output could not be verified without an installation.
