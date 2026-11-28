"""Fetch pinned prompt workloads and write the immutable split registry.

Run once (needs network + pyarrow for the ultrachat parquet):

    uv run --with pyarrow python experiments/frontier2026/speculative/fetch_prompts.py

Writes:
  dataset_registry.json   committed: source revisions, checksums, prompt IDs + hashes, split hashes
  data/prompts_dev.jsonl, data/prompts_audit.jsonl, data/sealed/prompts_sealed.jsonl   git-ignored text

Selection rule (pre-registered, do not tune after seeing results):
  pool   = first POOL rows in file order with MIN_CHARS <= len(prompt) <= MAX_CHARS
  pick   = PER_DOMAIN rows with the smallest sha256("{SEED}:{source}:{row_id}")
  splits = sort picked rows by char length; shortest third -> dev, middle -> audit, longest -> sealed
The length shift between dev and audit/sealed is deliberate (handoff section 5).
Only the prompt field is extracted; answers, solutions and test cases never leave this script.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SEED = 2026
PER_DOMAIN = 24  # 8 dev + 8 audit + 8 sealed: an instrumentation pilot, not a statistical sample
MIN_CHARS, MAX_CHARS = 40, 6000
SPLITS = ("dev", "audit", "sealed")

SOURCES = [
    {
        "domain": "math",
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "file": "test.jsonl",
        "file_checksum": {"git_blob_sha1": "2376b9a194b46c0790e197c91b7249e5f88ac09b"},
        "license": "not declared in dataset card (MATH derivative); prompt text kept out of git",
        "original_split": "test",
        "prompt_field": "problem",
        "id_field": "unique_id",
        "pool": 200,
        "contamination": "public 2021-era benchmark; used as an inference workload only, no capability claim",
    },
    {
        "domain": "code",
        "repo": "livecodebench/code_generation_lite",
        "revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "file": "test6.jsonl",
        "file_checksum": {"lfs_sha256": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
                          "note": "from HF metadata; only the first rows are streamed, file not hashed locally"},
        "license": "cc (as declared in dataset card)",
        "original_split": "test (release_v6 file)",
        "prompt_field": "question_content",
        "id_field": "question_id",
        "pool": 60,
        "contamination": "contest problems dated 2025; used as an inference workload only",
    },
    {
        "domain": "chat",
        "repo": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "file": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        "file_checksum": {"lfs_sha256": "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70"},
        "license": "mit",
        "original_split": "test_sft",
        "prompt_field": "prompt",
        "id_field": "prompt_id",
        "pool": 200,
        "contamination": "synthetic dialogues from 2023; first user turn only",
    },
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _open(src: dict):
    url = f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{src['file']}"
    return urllib.request.urlopen(url, timeout=120)


def read_rows(src: dict) -> list[dict]:
    """First `pool` rows in file order, reduced to {row_id, text}."""
    if src["file"].endswith(".parquet"):
        import pyarrow.parquet as pq

        raw = _open(src).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != src["file_checksum"]["lfs_sha256"]:
            raise SystemExit(f"{src['repo']}: checksum mismatch {got}")
        table = pq.read_table(io.BytesIO(raw), columns=[src["id_field"], src["prompt_field"]])
        rows = table.slice(0, src["pool"]).to_pylist()
    else:
        rows = []
        with _open(src) as fh:  # streamed: livecodebench rows carry MBs of test cases we never keep
            for line in fh:
                rows.append(json.loads(line))
                if len(rows) >= src["pool"]:
                    break
    return [{"row_id": str(r[src["id_field"]]), "text": str(r[src["prompt_field"]]).strip()} for r in rows]


def select(src: dict, rows: list[dict]) -> tuple[list[dict], dict]:
    eligible = [r for r in rows if MIN_CHARS <= len(r["text"]) <= MAX_CHARS]
    rejected = {"too_short_or_long": len(rows) - len(eligible)}
    if len(eligible) < PER_DOMAIN:
        raise SystemExit(f"{src['repo']}: only {len(eligible)} eligible rows")
    picked = sorted(eligible, key=lambda r: sha(f"{SEED}:{src['repo']}:{r['row_id']}"))[:PER_DOMAIN]
    picked.sort(key=lambda r: (len(r["text"]), r["row_id"]))
    third = PER_DOMAIN // 3
    out = []
    for i, r in enumerate(picked):
        out.append({
            "prompt_id": f"{src['domain']}:{r['row_id']}",
            "domain": src["domain"],
            "split": SPLITS[i // third],
            "chars": len(r["text"]),
            "sha256": sha(r["text"]),
            "text": r["text"],
        })
    return out, rejected


def split_hash(prompts: list[dict]) -> str:
    return sha("\n".join(sorted(f"{p['prompt_id']}:{p['sha256']}" for p in prompts)))


def main() -> None:
    prompts, sources = [], []
    for src in SOURCES:
        picked, rejected = select(src, read_rows(src))
        prompts += picked
        sources.append({**src, "url": f"https://huggingface.co/datasets/{src['repo']}",
                        "selected": len(picked), "rejected": rejected})
        print(f"{src['domain']}: {len(picked)} prompts from {src['repo']}@{src['revision'][:8]}")

    hashes = [p["sha256"] for p in prompts]
    if len(set(hashes)) != len(hashes):
        raise SystemExit("duplicate prompt text across splits/domains")

    (HERE / "data" / "sealed").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = HERE / "data" / ("sealed/prompts_sealed.jsonl" if split == "sealed" else f"prompts_{split}.jsonl")
        rows = [p for p in prompts if p["split"] == split]
        path.write_text("".join(json.dumps({k: p[k] for k in ("prompt_id", "domain", "text")}) + "\n" for p in rows))

    registry = {
        "schema_version": 1,
        "seed": SEED,
        "selection_rule": __doc__.split("Selection rule")[1].strip(),
        "sources": sources,
        "splits": {s: {"sha256": split_hash([p for p in prompts if p["split"] == s]),
                       "count": sum(p["split"] == s for p in prompts),
                       "role": {"dev": "visible to proposer and policy",
                                "audit": "independent; queried only as a counted verification",
                                "sealed": "never served by the API; evaluated only after policies are frozen"}[s]}
                   for s in SPLITS},
        "prompts": [{k: v for k, v in p.items() if k != "text"} for p in prompts],
    }
    (HERE / "dataset_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print("wrote dataset_registry.json", {s: registry["splits"][s]["sha256"][:12] for s in SPLITS})


if __name__ == "__main__":
    main()
