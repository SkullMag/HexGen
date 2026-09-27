#!/usr/bin/env python3
"""Download pinned, gated benchmark inputs directly onto an NRP model PVC."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download


MODEL_REPO = "meta-llama/Llama-2-70b-hf"
DATASET_REPO = "lmsys/chatbot_arena_conversations"
DATASET_REVISION = "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8"
DATASET_FILE = "data/train-00000-of-00001-cced8514c7ed782a.parquet"
DATASET_SHA256 = "3726a6352e9bfc34e206460646f6e5e99bb837751966a671ddd30c7f64e5b06e"
MODEL_FILES = ["config.json", "generation_config.json", "tokenizer.model",
               "tokenizer_config.json", "special_tokens_map.json",
               "model.safetensors.index.json", "model-*.safetensors"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(value: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".part",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/model/checkpoint"))
    parser.add_argument("--workload-dir", type=Path, default=Path("/model/workload"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a pinned 40-character commit SHA")
    if not os.environ.get("HF_TOKEN"):
        parser.error("HF_TOKEN must come from a private Kubernetes Secret")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    marker = args.model_dir / "snapshot_revision.json"
    if marker.exists() and json.loads(marker.read_text())["revision"] != args.revision:
        parser.error("model directory already contains a different revision")
    print(f"Downloading {MODEL_REPO} at {args.revision} to NRP storage", flush=True)
    snapshot_download(repo_id=MODEL_REPO, revision=args.revision,
                      local_dir=str(args.model_dir), allow_patterns=MODEL_FILES,
                      max_workers=4, token=True)
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    for shard in sorted(shards):
        path = args.model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty checkpoint shard: {shard}")
    if len(shards) != 15:
        raise RuntimeError(f"Expected 15 safetensors shards, found {len(shards)}")
    atomic_json({"repo_id": MODEL_REPO, "revision": args.revision,
                 "shards": len(shards)}, marker)
    print(f"Checkpoint ready: {len(shards)} shards", flush=True)

    args.workload_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_REPO} at {DATASET_REVISION} to NRP storage", flush=True)
    dataset = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
                                   revision=DATASET_REVISION, filename=DATASET_FILE,
                                   local_dir=str(args.workload_dir), token=True))
    observed = digest(dataset)
    if observed != DATASET_SHA256:
        raise RuntimeError("Pinned prompt dataset SHA-256 did not match")
    atomic_json({"repo_id": DATASET_REPO, "revision": DATASET_REVISION,
                 "file": DATASET_FILE, "sha256": observed},
                args.workload_dir / "dataset_revision.json")
    print("Pinned prompt dataset ready", flush=True)


if __name__ == "__main__":
    main()
