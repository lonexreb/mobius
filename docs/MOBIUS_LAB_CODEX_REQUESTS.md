# Möbius Lab: external dependency requests for Codex

[Back to runbook](MOBIUS_LAB_RUNBOOK.md)

Maintained by Claude Code. Codex works this list; the app already degrades gracefully for every item
(`GET /api/v1/connections` reports each one as not connected with its blocker).

**Standing rules**

- Spending authorization is **$0**. Nothing here may be purchased, rented or started without the owner's explicit approval.
- Never paste keys, tokens or connect-instructions into chat, issues, commits or this file. Secrets go in a git-ignored `.env.local` or the provider's own credential store.
- Costs below are **not verified**. Where a number is not on an official page that was actually read, it is written as "unknown". Do not replace "unknown" with a guess.
- "Resolver" says who can close the item: **Codex** (no human needed) or **Owner** (interactive login, billing or approval required).

Last updated: 2026-11-22. Nothing on this list has been provisioned.

## Summary

| # | Dependency | Priority | Resolver | Blocks |
|---|------------|----------|----------|--------|
| 1 | CUDA GPU for Qwen3-8B + EAGLE-3 (Modal, $25 cap approved; workspace paused) | P0 | Owner (un-pause) | The flagship benchmark |
| 2 | vLLM with speculators support on that GPU | P0 | Codex, after #1 | The flagship benchmark |
| 3 | Autolab CLI install + login | P1 | Owner (login) | Autolab mode, Hill scoring |
| 4 | Autolab Hill for the fixed evaluator | P1 | Codex, after #3 | Externally scored runs |
| 5 | One real `autolab metrics --format json` capture | P1 | Codex, after #4 | Raw-payload import mapping |
| 6 | Hugging Face token | P2 | Owner | Only if rate-limited or a repo becomes gated |
| 7 | Anthropic API key for an LLM hypothesis proposer | P2 | Owner (paid) | Optional LLM proposer |
| 8 | Egbe business connection | P3 | Owner | Optional distribution work |
| 9 | Public hosting + auth | P3 | Owner | Optional public demo |

---

## 1. CUDA GPU for the flagship benchmark (P0)

> **Update 2026-11-22:** the owner chose **Modal** and approved a **$25 hard cap**. The Modal CLI is installed and
> logged in on the dev machine, but every command is refused with `Workspace ... is paused`, so nothing has run and
> nothing has been spent. **Owner action:** un-pause the workspace in the Modal dashboard (billing/settings), and
> ideally set a workspace spend limit of $25 there as a second guard. Then:
> `modal run integrations/modal/bench_app.py::download` (CPU), `...::probe` (GPU, a few minutes), `modal deploy integrations/modal/bench_app.py`.
> Pricing read from https://modal.com/pricing on 2026-11-22: A100 80 GB $0.000694/s (about $2.50/h), CPU $0.0000131/core/s,
> memory $0.00000222/GiB/s; Starter plan lists $30/month free credits (not verified for this workspace).

- **Provider:** any Linux + NVIDIA CUDA machine. Options to price: an Autolab compute node, Hugging Face Jobs, or a rented GPU.
- **Purpose:** run `Qwen/Qwen3-8B` target-only and with `RedHatAI/Qwen3-8B-speculator.eagle3` on matched hardware. This is the only way to produce the headline result. The local llama.cpp runs are a different model, draft and engine and say nothing about it.
- **Exact access required:** shell on one GPU box for a few hours; enough VRAM for an 8B target, the speculator and KV cache at batch 1. Required VRAM is **not verified**: measure it, do not assume.
- **Permissions:** ability to start/stop one instance. No org-wide billing access needed.
- **Official documentation:** https://docs.vllm.ai/projects/speculators/en/stable/user_guide/getting_started/ , https://huggingface.co/docs/hub/jobs-training , https://docs.autolab.ai/guides/experiments/
- **Cost estimate:** unknown. Codex: record the provider's published hourly price for the chosen GPU class and the source URL here before asking for approval. A pilot is roughly 1 baseline + 6 to 8 candidate runs of 24 prompts × 96 tokens, plus model download and server restarts; that sizing is an estimate, not a measurement.
- **Free credits:** unknown. Check whether the Sundai event grants Autolab or Hugging Face credits and record the allowance and expiry.
- **Infrastructure requirements:** Linux, recent NVIDIA driver, ~40 GB disk for weights, outbound HTTPS to huggingface.co.
- **Verification command:** `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`
- **Resolver:** Owner must approve the spend and the cap. Codex can then provision.
- **Current blocker:** no CUDA GPU on the dev machine (Apple M4 Max); spend authorization is $0.

## 2. vLLM with speculators support (P0, after #1)

- **Provider:** vLLM (open source).
- **Purpose:** serve target-only and EAGLE-3 speculative configurations.
- **Exact access required:** install on the GPU box; download both model repos at the pinned revisions in `experiments/frontier2026/speculative/manifest.json` (`flagship.target_model.revision`, `flagship.speculator.revision`).
- **Official documentation:** https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/serve_vllm/
- **Cost:** software is free; compute is item 1.
- **What Claude Code needs back** (to write the `VllmEngine` in `evaluator.py` without guessing):
  1. `vllm --version` output.
  2. The exact `vllm serve` command that starts each configuration, confirmed working.
  3. Whether the completions endpoint can return **generated token IDs** and which request field enables it. The lossless gate needs token IDs, not text.
  4. Which speculative-decoding counters `/metrics` exposes (names verbatim), if any.
  5. The supported range for the lookahead / `num_speculative_tokens` knob. `flagship.search_space_unverified` is a placeholder until this is known.
- **Verification command:** `curl -fsS http://127.0.0.1:8000/v1/models` on the GPU box after `vllm serve` is up.
- **Resolver:** Codex.
- **Current blocker:** item 1.

## 3. Autolab CLI and account (P1)

- **Provider:** Autolab.
- **Purpose:** run the fixed evaluator as a frozen Hill and track externally scored runs.
- **Exact access required:** CLI installed on the dev machine; `autolab login` completed by the owner; `autolab install claude`.
- **Permissions:** one private project. No public Hill needed.
- **Official documentation:** https://docs.autolab.ai/install/ , https://docs.autolab.ai/reference/cli/ , https://docs.autolab.ai/guides/coding-agents/
- **Cost estimate:** unknown. Record plan price, per-run agent cost and compute cost separately, with source URLs. The Climb `--max-cost` cap and compute rental may be billed separately; confirm which costs the cap covers.
- **Free credits:** unknown; check the account and the event.
- **Infrastructure requirements:** review `https://app.autolab.ai/install.sh` before running it; do not pipe it to `sh` unread.
- **Verification command:** `autolab --version && autolab whoami` (must not print tokens; do not `cat` the credentials file).
- **Resolver:** Owner for login and any billing step; Codex for the install after the script is reviewed.
- **Current blocker:** CLI not installed (`which autolab` is empty); no account verified.

## 4. Autolab Hill wrapping the fixed evaluator (P1, after #3)

- **Purpose:** freeze `experiments/frontier2026/speculative/evaluator.py`, `manifest.json` and `dataset_registry.json` as a Hill so scoring cannot be edited by a candidate.
- **Exact work:** `autolab hills new mobius-qwen3-inference`; implement `eval(submission, *, final=False, **params)` by calling the evaluator; put `data/sealed/` under the Hill's `private/`; prove `autolab hills check`, a baseline eval and a deliberately broken candidate; `autolab hills commit`.
- **Official documentation:** https://docs.autolab.ai/guides/hills/
- **What Claude Code needs back:** the generated scaffold (`hill.yaml`, `eval.py` signature) and the committed Hill hash.
- **Verification command:** `autolab hills check mobius-qwen3-inference`
- **Resolver:** Codex (no spend for a local check; pushing and running a Climb needs items 1 and 3 and owner approval).
- **Current blocker:** item 3.

## 5. One real Autolab metrics payload (P1, after #4)

- **Purpose:** `POST /api/v1/import/autolab` currently accepts Möbius's own envelope (see `AutolabImport` in `apps/api/src/mobius_lab/schemas.py`): source run id, Hill hash, verdict, cost and the evaluator artifact. Mapping from raw `autolab metrics RUN_ID --format json` is deliberately not written, because no real payload exists to write it against.
- **Exact access required:** one completed run. Save the sanitized JSON to `integrations/autolab/fixtures/` (strip tokens, emails, account ids).
- **Verification command:** `autolab log && autolab metrics <RUN_ID> --format json | python3 -m json.tool | head`
- **Resolver:** Codex.
- **Current blocker:** items 3 and 4.

## 6. Hugging Face token (P2)

- **Purpose:** not needed today. All three prompt datasets and both flagship model repos were readable anonymously on 2026-11-22 (`gated: false`). Needed only for rate limits or if a repo becomes gated.
- **Exact access required:** a **read-only** token, exported as `HF_TOKEN` in the shell that runs `fetch_prompts.py` or vLLM.
- **Official documentation:** https://huggingface.co/docs/hub/security-tokens
- **Cost:** free for read access. Hugging Face Jobs or Inference usage would be billed and is not authorized.
- **Verification command:** `curl -fsS -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 >/dev/null && echo ok`
- **Resolver:** Owner creates the token; Codex stores it.
- **Current blocker:** none.

## 7. LLM hypothesis proposer (P2, optional)

- **Purpose:** replace the rule-based neighbourhood proposer with structured-output proposals from Claude. Not required: the controller, gates and budget work without it, and LLM spend is currently a true $0.
- **Exact access required:** an Anthropic API key as `ANTHROPIC_API_KEY` in `apps/api/.env.local`, with a small monthly limit.
- **Official documentation:** https://docs.claude.com/ ; pricing: https://www.anthropic.com/pricing
- **Cost estimate:** unknown until a model and prompt size are chosen. A proposal is one short structured call per candidate; read current per-token prices from the pricing page and record them here.
- **Verification command:** a one-token Messages API call from the API host returning HTTP 200.
- **Resolver:** Owner (billing). Codex must not create keys.
- **Current blocker:** no key; no spend authorization. Token usage and cost would have to be recorded per proposal in the ledger's `llm_inference` category before this is turned on.

## 8. Egbe connection (P3, optional, outside the scoring path)

- **Purpose:** audience research and, with approval, a landing page. Never part of ML scoring.
- **Exact access required:** owner opens the business dashboard → Connect your agent → copies the private instructions into a private agent session. Those instructions must never be committed or pasted here.
- **Official documentation:** https://docs.egbe.ai/start/quickstart/ , https://docs.egbe.ai/start/connect-agent/ , https://docs.egbe.ai/reference/access-and-costs/
- **Cost / credits:** unknown; verify the allowance inside the connected business.
- **Resolver:** Owner.
- **Current blocker:** not started. No emails, ads, publishing or credit spend without explicit approval.

## 9. Public hosting (P3, optional)

- **Purpose:** a read-only public view of a finished study.
- **Current blocker:** the API has **no authentication** and binds to 127.0.0.1 only. Do not expose it. A public deployment needs auth on every write route, rate limits and owner approval first.
- **Resolver:** Owner decision; then Codex.
