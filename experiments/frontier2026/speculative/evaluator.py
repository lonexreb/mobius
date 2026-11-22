"""Fixed, independent evaluator for speculative-decoding efficiency.

Stdlib only. The candidate is a config dict validated against the pinned allowlist;
it never supplies code, commands or its own score. Every number in the artifact is
measured here: token IDs and decode timings come from the engine's response, latency
from this process's clock.

    python evaluator.py --target-only --split dev --out base.json
    python evaluator.py --config '{"lookahead_tokens":8,"draft_p_min":0.75}' \
        --split dev --reference base.json --out cand.json

Measurement design (pre-registered, forced by what was observed on this machine):
  * Paired and interleaved. A candidate run keeps a target-only server and the
    speculative server loaded together and alternates requests prompt by prompt
    (order flips each prompt), never concurrently. Background load and thermal drift
    moved target-only throughput by ~30% between back-to-back runs, so speedup is
    computed only against the in-run control, never against an earlier run.
  * Streaming. Token IDs are accumulated from stream chunks, which also gives a
    measured time-to-first-token.
  * decode_tps = emitted tokens / server-reported decode time.

Engines implement `Engine`. `LlamaCppEngine` is verified locally. The vLLM/CUDA
flagship engine is intentionally absent until it can be written against an installed
vLLM (see docs/MOBIUS_LAB_CODEX_REQUESTS.md); nothing here guesses its flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
REGISTRY_PATH = HERE / "dataset_registry.json"
HASHED_FILES = ("evaluator.py", "manifest.json", "dataset_registry.json")
ONLINE_SPLITS = ("dev", "audit")


class EvalError(Exception):
    """Evaluation could not produce a trustworthy artifact."""


def evaluator_sha256() -> str:
    h = hashlib.sha256()
    for name in HASHED_FILES:
        h.update(name.encode() + b"\0" + (HERE / name).read_bytes())
    return h.hexdigest()


def validate_config(config: dict) -> dict:
    """Allowlist check: exactly the pinned knobs, exactly the pinned values."""
    space = MANIFEST["search_space"]
    if not isinstance(config, dict) or set(config) != set(space):
        raise EvalError(f"config keys must be exactly {sorted(space)}")
    for k, v in config.items():
        if isinstance(v, bool) or v not in space[k]:
            raise EvalError(f"{k}={v!r} not in pinned search space {space[k]}")
    return {k: config[k] for k in sorted(config)}


def load_prompts(split: str, unseal_final: bool = False) -> tuple[list[dict], str]:
    """Prompts for a split in registry order, verified against registry hashes."""
    if split == "sealed":
        if not unseal_final:
            raise EvalError("sealed split requires --unseal-final (post-freeze only)")
    elif split not in ONLINE_SPLITS:
        raise EvalError(f"unknown split {split!r}")
    registry = json.loads(REGISTRY_PATH.read_text())
    path = HERE / "data" / ("sealed/prompts_sealed.jsonl" if split == "sealed" else f"prompts_{split}.jsonl")
    if not path.exists():
        raise EvalError(f"prompt cache missing: {path}. Run fetch_prompts.py")
    texts = {r["prompt_id"]: r for r in map(json.loads, path.read_text().splitlines())}
    out = []
    for meta in registry["prompts"]:
        if meta["split"] != split:
            continue
        row = texts.get(meta["prompt_id"])
        if row is None or hashlib.sha256(row["text"].encode()).hexdigest() != meta["sha256"]:
            raise EvalError(f"prompt {meta['prompt_id']} missing or does not match registry hash")
        out.append(row)
    return out, registry["splits"][split]["sha256"]


# ---------- scoring (pure functions) ----------

def compare_tokens(reference: list[dict], candidate: list[dict]) -> dict:
    """Exact token-ID equality per paired prompt. Any difference fails the lossless gate.

    Mismatches are also classified (truncated prefix vs true divergence) for diagnosis;
    the classification never softens the gate.
    """
    ref = {r["prompt_id"]: r["token_ids"] for r in reference}
    truncated, diverged = [], []
    for c in candidate:
        want, got = ref.get(c["prompt_id"]), c["token_ids"]
        if want == got:
            continue
        if want is not None and len(got) < len(want) and want[:len(got)] == got:
            truncated.append(c["prompt_id"])
        else:
            first = next((i for i, (a, b) in enumerate(zip(want or [], got)) if a != b), min(len(want or []), len(got)))
            diverged.append({"prompt_id": c["prompt_id"], "first_divergent_index": first})
    missing = sorted(set(ref) - {c["prompt_id"] for c in candidate})
    compared = len(candidate)
    bad = len(truncated) + len(diverged)
    return {
        "compared": compared,
        "matched": compared - bad,
        "output_token_match_rate": (compared - bad) / compared if compared else None,
        "truncated_prefix_prompt_ids": truncated,
        "diverged": diverged,
        "missing_prompt_ids": missing,
        "exact_token_match": compared > 0 and not bad and not missing,
    }


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def aggregate(requests: list[dict]) -> dict:
    by_domain: dict[str, list[float]] = {}
    for r in requests:
        by_domain.setdefault(r["domain"], []).append(r["decode_tps"])
    drafted = sum(r["draft_n"] or 0 for r in requests)
    accepted = sum(r["draft_n_accepted"] or 0 for r in requests)
    e2e = [r["e2e_ms"] for r in requests]
    ttft = [r["ttft_ms"] for r in requests if r.get("ttft_ms") is not None]
    return {
        "throughput_tokens_per_sec": statistics.median(r["decode_tps"] for r in requests),
        "throughput_by_domain": {d: statistics.median(v) for d, v in sorted(by_domain.items())},
        "p50_latency_ms": _pct(e2e, 0.5),
        "p95_latency_ms": _pct(e2e, 0.95),
        "time_to_first_token_ms": statistics.median(ttft) if ttft else None,
        "draft_acceptance_rate": accepted / drafted if drafted else None,
        "total_output_tokens": sum(r["output_tokens"] for r in requests),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in requests),
        "prompts": len(requests),
    }


def _geomean_domain_speedup(base: dict[str, list[float]], cand: dict[str, list[float]]) -> tuple[float, dict]:
    per = {d: statistics.median(cand[d]) / statistics.median(base[d]) for d in sorted(base)}
    return math.exp(sum(map(math.log, per.values())) / len(per)), per


def speedup_stats(reference: list[dict], candidate: list[dict]) -> dict:
    """speedup = geometric mean over domains of median(cand tps) / median(base tps).

    CI: percentile bootstrap, resampling paired prompts within each domain. It captures
    prompt-to-prompt variation inside this pair of runs only; it does NOT capture
    run-to-run drift (thermal state, background load). That is what REPLICATE is for.
    """
    ref = {r["prompt_id"]: r for r in reference}
    pairs: dict[str, list[tuple[float, float]]] = {}
    for c in candidate:
        if c["prompt_id"] in ref:
            pairs.setdefault(c["domain"], []).append((ref[c["prompt_id"]]["decode_tps"], c["decode_tps"]))
    if not pairs:
        raise EvalError("no paired prompts between reference and candidate")

    def stat(sample: dict[str, list[tuple[float, float]]]) -> tuple[float, dict]:
        return _geomean_domain_speedup({d: [b for b, _ in v] for d, v in sample.items()},
                                       {d: [c for _, c in v] for d, v in sample.items()})

    point, per_domain = stat(pairs)
    cfg = MANIFEST["statistics"]
    rng = random.Random(cfg["bootstrap_seed"])
    boots = sorted(stat({d: rng.choices(v, k=len(v)) for d, v in pairs.items()})[0]
                   for _ in range(cfg["bootstrap_resamples"]))
    alpha = (1 - cfg["ci"]) / 2
    return {
        "speedup_vs_target": point,
        "speedup_by_domain": per_domain,
        "speedup_ci_low": boots[int(alpha * len(boots))],
        "speedup_ci_high": boots[int((1 - alpha) * len(boots)) - 1],
        "speedup_ci_method": f"{int(cfg['ci'] * 100)}% percentile bootstrap over paired prompts, "
                             f"{cfg['bootstrap_resamples']} resamples; within-run variation only",
        "paired_prompts": sum(map(len, pairs.values())),
    }


# ---------- engine ----------

class Engine(Protocol):
    def start(self, config: dict | None) -> None: ...
    def complete(self, prompt: str, speculative: bool) -> dict: ...
    def describe(self) -> dict: ...
    def stop(self) -> None: ...


def _resolve_gguf(spec: dict) -> tuple[Path, str]:
    """Model path and sha256. Ollama blobs are content-addressed, so the digest is free."""
    if os.environ.get(spec["env"]):
        path = Path(os.environ[spec["env"]])
        if not path.is_file():
            raise EvalError(f"{spec['env']} does not point to a file")
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return path, h.hexdigest()
    name, tag = spec["ollama"].split(":")
    root = Path.home() / ".ollama" / "models"
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest.exists():
        raise EvalError(f"model {spec['ollama']} not found; `ollama pull {spec['ollama']}` or set {spec['env']}")
    digest = next(layer["digest"] for layer in json.loads(manifest.read_text())["layers"]
                  if layer["mediaType"].endswith("image.model"))
    return root / "blobs" / digest.replace(":", "-"), digest.split(":")[1]


class LlamaCppEngine:
    """llama-server with Metal offload; speculation via --model-draft.

    Always runs a target-only server; candidates add a second, speculative server.
    """

    def __init__(self) -> None:
        self.binary = shutil.which("llama-server")
        if not self.binary:
            raise EvalError("llama-server not found on PATH (brew install llama.cpp)")
        local = MANIFEST["local_engine"]
        self.target, self.target_sha = _resolve_gguf(local["target"])
        self.draft, self.draft_sha = _resolve_gguf(local["draft"])
        self.servers: dict[bool, tuple[subprocess.Popen, int, list[str]]] = {}

    def _launch(self, extra: list[str]) -> tuple[subprocess.Popen, int, list[str]]:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        gen = MANIFEST["generation"]
        argv = [self.binary, "-m", str(self.target), "-ngl", "99", "-c", str(gen["ctx_size"]),
                "-np", str(gen["request_concurrency"]), "--host", "127.0.0.1", "--port", str(port), "--no-webui", *extra]
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise EvalError(f"llama-server exited with code {proc.returncode} during startup")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                return proc, port, argv
            except OSError:
                time.sleep(0.5)
        proc.kill()
        raise EvalError("llama-server did not become ready within 120s")

    def start(self, config: dict | None) -> None:
        self.servers[False] = self._launch([])
        if config is not None:
            self.servers[True] = self._launch(
                ["-md", str(self.draft), "-ngld", "99", "--draft-max", str(config["lookahead_tokens"]),
                 "--draft-min", "0", "--draft-p-min", str(config["draft_p_min"])])

    def complete(self, prompt: str, speculative: bool) -> dict:
        _, port, _ = self.servers[speculative]
        gen = MANIFEST["generation"]
        body = json.dumps({"prompt": prompt, "n_predict": gen["max_output_tokens"], "temperature": gen["temperature"],
                           "seed": gen["seed"], "cache_prompt": False, "return_tokens": True, "stream": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        tokens: list[int] = []
        ttft = final = None
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("tokens"):
                    ttft = ttft if ttft is not None else (time.monotonic() - t0) * 1000
                    tokens += chunk["tokens"]
                if chunk.get("stop"):
                    final = chunk
        if final is None:
            raise EvalError("stream ended without a final chunk")
        t = final["timings"]
        return {
            "token_ids": tokens,
            "decode_tps": len(tokens) / (t["predicted_ms"] / 1000) if t["predicted_ms"] > 0 else float("nan"),
            "output_tokens": len(tokens),
            "prompt_tokens": final["tokens_evaluated"],
            "ttft_ms": ttft,
            "stop_type": final.get("stop_type"),
            # counters exist only when a draft model is loaded; absent means "not applicable", not zero
            "draft_n": t.get("draft_n") if speculative else None,
            "draft_n_accepted": t.get("draft_n_accepted") if speculative else None,
        }

    def rss_gb(self, speculative: bool) -> float | None:
        try:
            pid = self.servers[speculative][0].pid
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1024 / 1024
        except (KeyError, ValueError, OSError, subprocess.SubprocessError):
            return None

    def describe(self) -> dict:
        version = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=20)
        line = next((ln for ln in (version.stderr + version.stdout).splitlines() if ln.startswith("version")), None)
        spec = True in self.servers
        argv = self.servers[spec][2]
        return {
            "name": MANIFEST["local_engine"]["name"],
            "engine_version": line,
            "argv": [a if not a.startswith("/") else Path(a).name for a in argv],
            "target_model": MANIFEST["local_engine"]["target"]["ollama"],
            "target_model_sha256": self.target_sha,
            "draft_model": MANIFEST["local_engine"]["draft"]["ollama"] if spec else None,
            "draft_model_sha256": self.draft_sha if spec else None,
        }

    def stop(self) -> None:
        for proc, _, _ in self.servers.values():
            if proc.poll() is None:
                proc.terminate()
        for proc, _, _ in self.servers.values():
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.servers = {}


def hardware() -> dict:
    def sysctl(key: str) -> str | None:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    mem = sysctl("hw.memsize")
    return {"cpu": sysctl("machdep.cpu.brand_string") or platform.processor(), "gpu_model": None,
            "accelerator": "Apple Metal (integrated)" if platform.system() == "Darwin" else None,
            "memory_gb": int(mem) / 2**30 if mem and mem.isdigit() else None,
            "os": f"{platform.system()} {platform.release()}", "arch": platform.machine()}


# ---------- runner ----------

def run(engine: Engine, config: dict | None, split: str, reference: dict | None,
        unseal_final: bool = False, max_prompts: int | None = None) -> dict:
    """Run one evaluation and return the artifact. Raises EvalError rather than emit partial metrics."""
    if config is not None:
        config = validate_config(config)
        if reference is None:
            raise EvalError("candidate evaluation requires a target-only reference artifact")
    prompts, split_sha = load_prompts(split, unseal_final)
    if max_prompts:
        prompts = prompts[:max_prompts]
    gen = MANIFEST["generation"]
    ev_sha = evaluator_sha256()

    def wrap(text: str) -> str:
        return gen["chat_template"].format(system=gen["system_prompt"], prompt=text)

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    spec = config is not None
    arms = [False, True] if spec else [False]
    rows: dict[bool, list[dict]] = {False: [], True: []}
    try:
        engine.start(config)
        warm0 = time.monotonic()
        for arm in arms:
            for _ in range(gen["warmup_requests"]):
                engine.complete(wrap(gen["warmup_prompt"]), arm)
        warmup_s = time.monotonic() - warm0
        for i, p in enumerate(prompts):
            for arm in (arms if i % 2 == 0 else arms[::-1]):  # flip order each prompt to cancel order effects
                r0 = time.monotonic()
                out = engine.complete(wrap(p["text"]), arm)
                rows[arm].append({"prompt_id": p["prompt_id"], "domain": p["domain"],
                                  "e2e_ms": (time.monotonic() - r0) * 1000, **out})
        rss = engine.rss_gb(spec) if hasattr(engine, "rss_gb") else None
        engine_info = engine.describe()
    finally:
        engine.stop()

    for r in rows[False] + rows[True]:
        if not r["token_ids"] or not math.isfinite(r["decode_tps"]) or r["decode_tps"] <= 0:
            raise EvalError(f"non-finite or empty measurement for {r['prompt_id']}")

    requests = rows[spec]
    artifact = {
        "schema_version": 2,
        "kind": "candidate" if spec else "target_only",
        "config": config,
        "split": split,
        "evaluator_sha256": ev_sha,
        "dataset_split_sha256": split_sha,
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "generation": {k: gen[k] for k in ("max_output_tokens", "temperature", "seed", "request_concurrency", "ctx_size")},
        "design": "interleaved paired control" if spec else "target-only reference",
        "engine": engine_info,
        "hardware": hardware(),
        "started_at": started.isoformat(),
        "wall_seconds": time.monotonic() - t0,
        "warmup_seconds": warmup_s,
        "server_rss_gb": rss,
        "requests": requests,
        "control_requests": rows[False] if spec else None,
        "metrics": aggregate(requests),
        "correctness": None,
        "reference": None,
    }
    if spec:
        mismatch = [k for k in ("evaluator_sha256", "dataset_split_sha256", "prompt_ids", "generation")
                    if reference.get(k) != artifact[k]]
        if reference.get("kind") != "target_only":
            mismatch.append("kind")
        if reference.get("engine", {}).get("target_model_sha256") != engine_info["target_model_sha256"]:
            mismatch.append("target_model_sha256")
        if mismatch:
            raise EvalError(f"reference is not a matched target-only run; differs in: {mismatch}")
        artifact["correctness"] = compare_tokens(reference["requests"], requests)
        # the in-run control must reproduce the stored reference, otherwise the gate itself is unreliable
        artifact["correctness"]["control_reproduces_reference"] = compare_tokens(
            reference["requests"], rows[False])["exact_token_match"]
        artifact["metrics"].update(speedup_stats(rows[False], requests))
        artifact["metrics"]["control_throughput_tokens_per_sec"] = aggregate(rows[False])["throughput_tokens_per_sec"]
        artifact["reference"] = {"started_at": reference["started_at"]}
    return artifact


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target-only", action="store_true")
    mode.add_argument("--config", help="candidate config JSON (allowlisted knobs only)")
    ap.add_argument("--split", default="dev", choices=["dev", "audit", "sealed"])
    ap.add_argument("--reference", help="target-only artifact for the same split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unseal-final", action="store_true", help="post-freeze only; never passed by the API")
    ap.add_argument("--max-prompts", type=int)
    args = ap.parse_args()
    try:
        config = None if args.target_only else json.loads(args.config)
        reference = json.loads(Path(args.reference).read_text()) if args.reference else None
        artifact = run(LlamaCppEngine(), config, args.split, reference, args.unseal_final, args.max_prompts)
    except (EvalError, json.JSONDecodeError, OSError) as exc:
        sys.stderr.write(f"EVAL_ERROR: {exc}\n")
        return 2
    Path(args.out).write_text(json.dumps(artifact))
    sys.stdout.write(json.dumps({"out": args.out, "metrics": artifact["metrics"],
                                 "correctness": artifact["correctness"]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
