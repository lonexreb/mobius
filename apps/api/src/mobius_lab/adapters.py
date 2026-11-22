"""Execution adapters and external bridges. All subprocesses use argv lists and timeouts."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .persistence import Ledger

REPO = Path(__file__).resolve().parents[4]
SPEC_DIR = REPO / "experiments" / "frontier2026" / "speculative"


def load_evaluator():
    """Import the fixed evaluator module (stdlib-only) for its hash, allowlist and manifest."""
    spec = importlib.util.spec_from_file_location("mobius_spec_evaluator", SPEC_DIR / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class RunResult:
    status: str                    # ok | failed | timeout | cancelled
    artifact: dict | None = None
    error: str | None = None


class ExecutionAdapter(Protocol):
    evidence_class: str

    def evaluate(self, exp: dict, reference_path: Path | None, out_path: Path, timeout: float) -> RunResult: ...
    def cancel(self) -> None: ...


class LocalAdapter:
    """Runs the fixed evaluator in a separate process on this machine. Real measurements, labelled LOCAL."""

    evidence_class = "local"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._cancelled = threading.Event()

    def evaluate(self, exp: dict, reference_path: Path | None, out_path: Path, timeout: float) -> RunResult:
        argv = [sys.executable, str(SPEC_DIR / "evaluator.py"), "--split", exp["split"], "--out", str(out_path)]
        argv += ["--target-only"] if exp["config"] is None else ["--config", json.dumps(exp["config"])]
        if reference_path is not None:
            argv += ["--reference", str(reference_path)]
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                      start_new_session=True)  # own process group so cancel reaches llama-server
        try:
            _, err = self._proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill()
            return RunResult("timeout", error=f"evaluator exceeded {timeout:.0f}s")
        if self._cancelled.is_set():
            return RunResult("cancelled", error="cancelled by operator")
        if self._proc.returncode != 0:
            return RunResult("failed", error=(err or "").strip()[-500:] or f"exit code {self._proc.returncode}")
        try:
            return RunResult("ok", artifact=json.loads(out_path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            return RunResult("failed", error=f"evaluator produced no readable artifact: {exc}")

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, 15)
            except ProcessLookupError:
                pass
            try:
                self._proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, 9)

    def cancel(self) -> None:
        self._cancelled.set()
        self._kill()


class ReplayAdapter:
    """Replays artifacts recorded by an earlier study in this ledger. Nothing is re-measured.

    The n-th run of a (config, split) pair maps to the n-th recorded run of that pair; if the
    controller asks for something that was never recorded, the run fails rather than inventing data.
    """

    evidence_class = "replay"

    def __init__(self, ledger: Ledger, source_study_id: str, study_id: str) -> None:
        self.ledger, self.source, self.study_id = ledger, source_study_id, study_id

    def evaluate(self, exp: dict, reference_path: Path | None, out_path: Path, timeout: float) -> RunResult:
        def same(e: dict) -> bool:
            return e["config"] == exp["config"] and e["split"] == exp["split"] and e["kind"] == exp["kind"]

        nth = sum(1 for e in self.ledger.experiments(self.study_id) if same(e) and e["seq"] < exp["seq"])
        recorded = [e for e in self.ledger.experiments(self.source) if same(e) and e["artifact_path"]
                    and e["evidence_class"] in ("local", "observed")]
        if nth >= len(recorded):
            return RunResult("failed", error=f"no recorded run #{nth + 1} for config={exp['config']} "
                                             f"split={exp['split']} in {self.source}")
        src = recorded[nth]
        try:
            artifact = json.loads(Path(src["artifact_path"]).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return RunResult("failed", error=f"recorded artifact unreadable: {exc}")
        artifact["replayed_from"] = {"study_id": self.source, "experiment_id": src["id"],
                                     "original_evidence_class": src["evidence_class"]}
        out_path.write_text(json.dumps(artifact))
        return RunResult("ok", artifact=artifact)

    def cancel(self) -> None:
        pass


# ---------- Rust Möbius ask/tell bridge ----------

class BridgeError(Exception):
    pass


class MobiusBridge:
    """Derived optimizer history in the existing Rust CLI, isolated per study + policy arm.

    The CLI keeps state under ~/.mobius; each arm gets its own HOME and working directory
    (verified on macOS: dirs::home_dir honours $HOME), so studies never share history.
    """

    METRIC = "speedup_vs_target"

    def __init__(self, state_root: Path, search_space: dict) -> None:
        self.root, self.space = state_root, search_space

    @staticmethod
    def binary() -> Path | None:
        for build in ("release", "debug"):
            path = REPO / "target" / build / "mobius"
            if path.is_file():
                return path
        return None

    def _run(self, study_id: str, arm: str, *args: str) -> dict:
        binary = self.binary()
        if binary is None:
            raise BridgeError("mobius CLI not built; run `cargo build -p mobius-cli`")
        home = self.root / study_id / arm
        home.mkdir(parents=True, exist_ok=True)
        toml = home / "mobius.toml"
        if not toml.exists():
            lines = ['[project]', 'name = "mobius-lab"', 'version = "0.1.0"', '', '[experiment]', 'budget_usd = 0.0',
                     '', '[experiment.sweep_space]']
            lines += [f"{k} = {json.dumps(v)}" for k, v in sorted(self.space.items())]
            toml.write_text("\n".join(lines) + "\n")
        try:
            out = subprocess.run([str(binary), *args, "--store", "sqlite"], cwd=home, capture_output=True, text=True,
                                 timeout=60, env={**os.environ, "HOME": str(home)})
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BridgeError(f"mobius {args[0]} failed to run: {exc}") from exc
        if out.returncode != 0:
            raise BridgeError(f"mobius {args[0]} exited {out.returncode}: {out.stderr.strip()[-300:]}")
        try:
            return json.loads(out.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise BridgeError(f"mobius {args[0]} printed no JSON") from exc

    def ask(self, study_id: str, arm: str) -> tuple[dict, str]:
        out = self._run(study_id, arm, "ask", "--strategy", "tpe", "--metric", self.METRIC)
        return out["config"], out.get("rationale", "")

    def tell(self, study_id: str, arm: str, config: dict, value: float) -> str:
        """Only ever called for VALID results: `tell` records status Success by design."""
        out = self._run(study_id, arm, "tell", "--config", json.dumps(config),
                        "--metrics", json.dumps({self.METRIC: value}))
        return out["id"]


# ---------- connection readiness (never returns secrets) ----------

def connections() -> dict:
    autolab = shutil.which("autolab")
    llama = shutil.which("llama-server")
    prompts = (SPEC_DIR / "data" / "prompts_dev.jsonl").exists()
    try:
        ev = load_evaluator()
        ev.LlamaCppEngine()
        local_blocker = None if prompts else "prompt cache missing: run fetch_prompts.py"
    except Exception as exc:  # readiness probe: any failure is reported, not raised
        local_blocker = str(exc)
    return {
        "local": {"ready": local_blocker is None, "blocker": local_blocker,
                  "detail": "llama.cpp on this machine; LOCAL evidence, not comparable to the CUDA flagship"},
        "mobius_cli": {"ready": MobiusBridge.binary() is not None,
                       "blocker": None if MobiusBridge.binary() else "run `cargo build -p mobius-cli`"},
        "autolab": {"ready": False, "cli_installed": autolab is not None,
                    "blocker": ("CLI installed but login, Hill and compute are not verified by this app yet"
                                if autolab else "autolab CLI not installed; see docs/MOBIUS_LAB_CODEX_REQUESTS.md")},
        "gpu": {"ready": False, "blocker": "no CUDA GPU or vLLM on this machine; rented compute not authorized ($0)"},
        "egbe": {"ready": False, "blocker": "not connected; outside the scoring path"},
        "llm_proposer": {"ready": False, "blocker": "not wired; proposals are rule-based, so LLM spend is $0"},
        "llama_server": llama is not None,
    }
