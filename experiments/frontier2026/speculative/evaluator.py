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
  * Lossless is judged against the in-run control, and only when the target is
    self-consistent. On the CUDA flagship the in-run target-only control differed from
    the stored baseline on 4/24 prompts, so a stored baseline is not a usable token
    reference. Every run therefore sends each prompt to the target a second time; a
    candidate is lossless only if that repeat equals the control on every prompt (the
    target's own noise floor is zero in this run) and the candidate equals the control.
    Agreement with the stored baseline is still recorded, as a diagnostic only.

Engines implement `Engine`: `LlamaCppEngine` (local) and `VllmEngine` (CUDA flagship).
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


def space_for(engine: str) -> dict:
    """Pinned search space for an engine: 'llama' (local) or 'vllm' (flagship)."""
    return MANIFEST["flagship"]["search_space"] if engine == "vllm" else MANIFEST["search_space"]


def fixed_spec_for(engine: str) -> dict:
    return MANIFEST["flagship"]["fixed_spec"] if engine == "vllm" else MANIFEST["fixed_spec"]


def validate_config(config: dict, engine: str = "llama") -> dict:
    """Allowlist check: exactly the pinned knobs, exactly the pinned values."""
    space = space_for(engine)
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

    name = "llama"

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
        prompt = gen["chat_template"].format(system=gen["system_prompt"], prompt=prompt)
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


class VllmEngine:
    """vLLM on a CUDA GPU: Qwen3-8B target-only, plus the EAGLE-3 speculator for candidates.

    Written against a live install (see `probe`). The target-only server is kept alive across
    runs in one container; `stop` only tears down the speculative server.
    """

    name = "vllm"
    PORTS = {False: 8001, True: 8002}

    def __init__(self) -> None:
        self.binary = shutil.which("vllm")
        if not self.binary:
            raise EvalError("vllm not found on PATH")
        try:
            from huggingface_hub import snapshot_download
            self.flagship = MANIFEST["flagship"]
            self.paths = {key: snapshot_download(self.flagship[key]["repo"], revision=self.flagship[key]["revision"],
                                                 local_files_only=True) for key in ("target_model", "speculator")}
        except Exception as exc:
            raise EvalError(f"pinned weights not cached; run the download step first ({exc})") from exc
        self.servers: dict[bool, tuple[subprocess.Popen, list[str]]] = {}
        self.load_seconds: dict[bool, float] = {}

    def _launch(self, speculative: bool, config: dict | None) -> None:
        gen, port = MANIFEST["generation"], self.PORTS[speculative]
        util = self.flagship["gpu_memory_utilization"]["speculative" if speculative else "target_only"]
        argv = [self.binary, "serve", self.paths["target_model"], "--served-model-name", "target",
                "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1",
                "--max-model-len", str(gen["ctx_size"]), "--gpu-memory-utilization", str(util),
                "--seed", str(gen["seed"])]
        if speculative:
            argv += ["--speculative-config", json.dumps({
                "model": self.paths["speculator"], "num_speculative_tokens": config["num_speculative_tokens"],
                "method": "eagle3"})]
        log_path = Path(f"/tmp/vllm_{port}.log")
        t0 = time.monotonic()
        with log_path.open("w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        while time.monotonic() - t0 < 900:
            if proc.poll() is not None:
                raise EvalError(f"vllm exited with code {proc.returncode} during startup: "
                                f"{log_path.read_text()[-1500:]}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                self.servers[speculative] = (proc, argv)
                self.load_seconds[speculative] = time.monotonic() - t0
                return
            except OSError:
                time.sleep(2)
        proc.kill()
        raise EvalError("vllm did not become ready within 900s")

    def start(self, config: dict | None) -> None:
        if False not in self.servers or self.servers[False][0].poll() is not None:
            self._launch(False, None)
        if config is not None:
            self._launch(True, config)

    def _request(self, speculative: bool, body: dict):
        req = urllib.request.Request(f"http://127.0.0.1:{self.PORTS[speculative]}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=300)

    def _body(self, prompt: str) -> dict:
        gen = MANIFEST["generation"]
        return {"model": "target", "temperature": gen["temperature"], "seed": gen["seed"],
                "max_tokens": gen["max_output_tokens"],
                "messages": [{"role": "system", "content": gen["system_prompt"]}, {"role": "user", "content": prompt}],
                "chat_template_kwargs": self.flagship["chat_template_kwargs"]}

    def _spec_counters(self) -> tuple[float, float]:
        """Cumulative (drafted, accepted) token counters from /metrics (names verified on vLLM 0.34.0)."""
        drafted = accepted = 0.0
        for line in self.metrics_text(True).splitlines():
            if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                drafted += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                accepted += float(line.rsplit(" ", 1)[1])
        return drafted, accepted

    def complete(self, prompt: str, speculative: bool) -> dict:
        """Streamed chat completion with `return_token_ids` (verified on vLLM 0.34.0).

        vLLM reports no decode timing, so throughput is client-side inside the same container:
        tokens after the first token-bearing chunk / time between the first and last such chunk.
        Speculative chunks carry several tokens each; the definition is identical for both arms.
        """
        before = self._spec_counters() if speculative else None
        body = {**self._body(prompt), "stream": True, "return_token_ids": True,
                "stream_options": {"include_usage": True}}
        tokens: list[int] = []
        t_first = t_last = usage = finish = None
        first_n = 0
        t0 = time.monotonic()
        with self._request(speculative, body) as resp:
            for line in resp:
                if not line.startswith(b"data: ") or b"[DONE]" in line:
                    continue
                chunk, now = json.loads(line[6:]), time.monotonic()
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    ids = choice.get("token_ids") or []
                    if ids:
                        if t_first is None:
                            t_first, first_n = now, len(ids)
                        t_last = now
                        tokens += ids
        if not tokens or usage is None:
            raise EvalError("vLLM stream returned no token IDs or no usage record")
        if usage.get("completion_tokens") != len(tokens):
            raise EvalError(f"token-ID stream incomplete: {len(tokens)} ids vs {usage.get('completion_tokens')} generated")
        span = t_last - t_first
        tps = (len(tokens) - first_n) / span if span > 0 and len(tokens) > first_n else len(tokens) / (t_last - t0)
        after = self._spec_counters() if speculative else None
        return {
            "token_ids": tokens,
            "decode_tps": tps,
            "output_tokens": len(tokens),
            "prompt_tokens": usage["prompt_tokens"],
            "ttft_ms": (t_first - t0) * 1000,
            "stop_type": finish,
            "draft_n": int(after[0] - before[0]) if speculative else None,
            "draft_n_accepted": int(after[1] - before[1]) if speculative else None,
        }

    def metrics_text(self, speculative: bool) -> str:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.PORTS[speculative]}/metrics", timeout=10) as r:
            return r.read().decode()

    def probe(self) -> dict:
        """Report what this build returns, so `complete` is written against facts, not memory."""
        out: dict = {"vllm_version": subprocess.run([self.binary, "--version"], capture_output=True, text=True).stdout.strip(),
                     "hardware": hardware()}
        try:
            self.start({"num_speculative_tokens": 3})
            out["load_seconds"] = dict(self.load_seconds)
            out["gpu_memory_used_mib"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip()
            prompt = "Write a Python function that returns the n-th Fibonacci number."
            for label, extra in (("return_token_ids", {"return_token_ids": True}),
                                 ("logprobs_as_ids", {"logprobs": True, "return_tokens_as_token_ids": True})):
                try:
                    with self._request(True, {**self._body(prompt), "max_tokens": 8, **extra}) as r:
                        out[f"nonstream_{label}"] = json.loads(r.read())
                except Exception as exc:
                    out[f"nonstream_{label}"] = f"ERROR {exc}"
            for arm in (False, True):
                chunks, stamps = [], []
                try:
                    body = {**self._body(prompt), "stream": True, "return_token_ids": True,
                            "stream_options": {"include_usage": True}}
                    t0 = time.monotonic()
                    with self._request(arm, body) as r:
                        for line in r:
                            if line.startswith(b"data: ") and b"[DONE]" not in line:
                                chunks.append(json.loads(line[6:]))
                                stamps.append(round(time.monotonic() - t0, 4))
                    out[f"stream_{'spec' if arm else 'target'}"] = {
                        "n_chunks": len(chunks), "first": chunks[:3], "last": chunks[-2:],
                        "stamps_head": stamps[:6], "stamps_tail": stamps[-3:]}
                except Exception as exc:
                    out[f"stream_{'spec' if arm else 'target'}"] = f"ERROR {exc}"
            out["spec_metric_names"] = sorted({ln.split("{")[0].split(" ")[0] for ln in self.metrics_text(True).splitlines()
                                               if ln and not ln.startswith("#") and ("spec" in ln or "token" in ln)})
            out["spec_metric_lines"] = [ln for ln in self.metrics_text(True).splitlines()
                                        if "spec" in ln and not ln.startswith("#")][:20]
        except EvalError as exc:
            out["error"] = str(exc)
        finally:
            self.stop(everything=True)
        return out

    def rss_gb(self, speculative: bool) -> float | None:
        return None  # host RSS is not the interesting number on a GPU; see gpu_memory in describe()

    def describe(self) -> dict:
        version = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        spec = True in self.servers
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        return {
            "name": "vLLM (CUDA)", "engine_version": f"vllm {version}",
            "argv": [a if not a.startswith("/") else Path(a).name for a in self.servers[spec][1]],
            "target_model": self.flagship["target_model"]["repo"],
            "target_model_sha256": self.flagship["target_model"]["revision"],
            "draft_model": self.flagship["speculator"]["repo"] if spec else None,
            "draft_model_sha256": self.flagship["speculator"]["revision"] if spec else None,
            "model_id_kind": "Hugging Face repo revision (git commit), not a file sha256",
            "gpu_memory_used_mib_both_servers": int(smi) if smi.isdigit() else None,
            "server_load_seconds": {("speculative" if k else "target_only"): round(v, 1)
                                    for k, v in self.load_seconds.items()},
        }

    def stop(self, everything: bool = False) -> None:
        for arm in ([True, False] if everything else [True]):
            entry = self.servers.pop(arm, None)
            if entry and entry[0].poll() is None:
                entry[0].terminate()
                try:
                    entry[0].wait(timeout=60)
                except subprocess.TimeoutExpired:
                    entry[0].kill()


def hardware() -> dict:
    def sysctl(key: str) -> str | None:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    mem = sysctl("hw.memsize")
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            gpu = None
    return {"cpu": sysctl("machdep.cpu.brand_string") or platform.processor(), "gpu_model": gpu,
            "accelerator": "Apple Metal (integrated)" if platform.system() == "Darwin" else ("NVIDIA CUDA" if gpu else None),
            "memory_gb": int(mem) / 2**30 if mem and mem.isdigit() else None,
            "os": f"{platform.system()} {platform.release()}", "arch": platform.machine()}


# ---------- runner ----------

def run(engine: Engine, config: dict | None, split: str, reference: dict | None,
        unseal_final: bool = False, max_prompts: int | None = None) -> dict:
    """Run one evaluation and return the artifact. Raises EvalError rather than emit partial metrics."""
    if config is not None:
        config = validate_config(config, getattr(engine, "name", "llama"))
        if reference is None:
            raise EvalError("candidate evaluation requires a target-only reference artifact")
    prompts, split_sha = load_prompts(split, unseal_final)
    if max_prompts:
        prompts = prompts[:max_prompts]
    gen = MANIFEST["generation"]
    ev_sha = evaluator_sha256()

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    spec = config is not None
    arms = [False, True] if spec else [False]
    rows: dict[bool, list[dict]] = {False: [], True: []}
    repeat: list[dict] = []   # second target-only pass: the target's own noise floor in this run
    try:
        engine.start(config)
        warm0 = time.monotonic()
        for arm in arms:
            for _ in range(gen["warmup_requests"]):
                engine.complete(gen["warmup_prompt"], arm)
        warmup_s = time.monotonic() - warm0
        for i, p in enumerate(prompts):
            for arm in (arms if i % 2 == 0 else arms[::-1]):  # flip order each prompt to cancel order effects
                r0 = time.monotonic()
                out = engine.complete(p["text"], arm)
                rows[arm].append({"prompt_id": p["prompt_id"], "domain": p["domain"],
                                  "e2e_ms": (time.monotonic() - r0) * 1000, **out})
            repeat.append({"prompt_id": p["prompt_id"], "domain": p["domain"], **engine.complete(p["text"], False)})
        rss = engine.rss_gb(spec) if hasattr(engine, "rss_gb") else None
        engine_info = engine.describe()
    finally:
        engine.stop()

    for r in rows[False] + rows[True] + repeat:
        if not r["token_ids"] or not math.isfinite(r["decode_tps"]) or r["decode_tps"] <= 0:
            raise EvalError(f"non-finite or empty measurement for {r['prompt_id']}")

    requests = rows[spec]
    artifact = {
        "schema_version": 3,
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
        "target_repeat_requests": repeat,
        "target_self_consistency": compare_tokens(rows[False], repeat),
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
        vs_stored = compare_tokens(reference["requests"], rows[False])
        artifact["correctness"] = {
            **compare_tokens(rows[False], requests), "compared_against": "in-run target-only control",
            # the target must reproduce itself within this run, otherwise an exact-match gate means nothing
            "control_self_consistent": artifact["target_self_consistency"]["exact_token_match"],
            "control_self_matched": artifact["target_self_consistency"]["matched"],
            # diagnostic only: a warm server need not reproduce an earlier cold run
            "control_reproduces_reference": vs_stored["exact_token_match"],
            "control_matched_reference": vs_stored["matched"]}
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
