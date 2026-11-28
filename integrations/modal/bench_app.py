"""Modal app: Qwen3-8B + EAGLE-3 on vLLM, scored by the fixed evaluator inside the container.

    modal run integrations/modal/bench_app.py::download     # CPU only: cache pinned weights in a Volume
    modal run integrations/modal/bench_app.py::probe        # GPU: print what the installed vLLM supports
    modal deploy integrations/modal/bench_app.py            # then the API's Modal adapter calls Bench.evaluate

Every GPU second is billed. Nothing here runs unless invoked; max_containers=1 and timeouts bound the spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
# On the dev machine this file lives in the repo; inside the container it is /root/bench_app.py
# and the evaluator directory is already mounted at /spec.
SPEC_DIR = (_HERE.parents[2] / "experiments" / "frontier2026" / "speculative") if len(_HERE.parents) > 2 else Path("/spec")
GPU = "A100-80GB"
MODELS = "/models"

app = modal.App("mobius-lab-bench")
weights = modal.Volume.from_name("mobius-lab-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub")
    .env({"HF_HOME": f"{MODELS}/hf", "HF_HUB_OFFLINE": "0", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
    # sealed prompts and tests never leave this machine
    .add_local_dir(str(SPEC_DIR), "/spec", ignore=["data/sealed/**", "tests/**", "**/__pycache__/**", "*.md"])
)


def _manifest() -> dict:
    return json.loads(Path("/spec/manifest.json").read_text())


@app.function(image=image, volumes={MODELS: weights}, timeout=3600, cpu=4)
def download() -> dict:
    """Fetch both repos at the revisions pinned in manifest.json. No GPU."""
    from huggingface_hub import snapshot_download

    flagship, out = _manifest()["flagship"], {}
    for key in ("target_model", "speculator"):
        spec = flagship[key]
        out[key] = snapshot_download(spec["repo"], revision=spec["revision"])
    weights.commit()
    return out


@app.function(image=image, gpu=GPU, volumes={MODELS: weights}, timeout=1500, cpu=4, memory=32768, max_containers=1)
def probe() -> dict:
    """Start both servers once and report what this vLLM build actually returns."""
    import sys
    sys.path.insert(0, "/spec")
    import evaluator as ev

    return ev.VllmEngine().probe()


@app.cls(image=image, gpu=GPU, volumes={MODELS: weights}, timeout=1500, cpu=4, memory=32768, max_containers=1,
         scaledown_window=60)
class Bench:
    @modal.enter()
    def boot(self) -> None:
        import sys
        sys.path.insert(0, "/spec")
        import evaluator as ev

        self.ev = ev
        self.engine = ev.VllmEngine()      # keeps the target-only server warm across calls in this container

    @modal.method()
    def evaluate(self, config: dict | None, split: str, reference: dict | None) -> dict:
        """One scored run. Returns {"artifact": ...} or {"error": ...}; never a partial measurement."""
        if split not in self.ev.ONLINE_SPLITS:
            return {"error": f"split {split!r} is not served remotely"}
        try:
            return {"artifact": self.ev.run(self.engine, config, split, reference)}
        except self.ev.EvalError as exc:
            return {"error": str(exc)}
