#!/usr/bin/env python3
"""Prepare a fixed 128-token LMSYS prompt bank for a pinned Llama 2 tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from transformers import LlamaTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/native_7b"))
from selection import digest, make_bank  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--candidate-count", type=int, default=1000)
    args = parser.parse_args()
    if args.count < 1 or args.candidate_count < args.count:
        parser.error("candidate-count must be at least count")
    source_sha = digest(args.dataset_file)
    if source_sha != args.dataset_sha256:
        raise ValueError("Dataset file SHA-256 does not match the pinned input")
    tokenizer = LlamaTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, legacy=True)
    settings = {
        "requests_per_experiment": args.candidate_count,
        "input_tokens": 128,
        "prompt_seed": 20260919,
        "model": "meta-llama/Llama-2-70b-hf",
        "model_revision": args.model_revision,
        "workload": {
            "dataset_id": "lmsys/chatbot_arena_conversations",
            "dataset_revision": "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8",
            "dataset_file": "data/train-00000-of-00001-cced8514c7ed782a.parquet",
            "conversation_column": "conversation_a",
            "selection_policy": "First user turn; truncate to 128 Llama 2 tokens including BOS; require exact native text round trip.",
        },
    }
    rows = pq.read_table(args.dataset_file,
                         columns=["question_id", "conversation_a", "language"]).to_pylist()
    bank = make_bank(rows, tokenizer, settings, source_sha)
    compatible = [item for item in bank["prompts"]
                  if tokenizer.encode(item["text"]) == item["input_ids"]
                  and len(item["input_ids"]) == 128]
    if len(compatible) < args.count:
        raise ValueError(f"Only {len(compatible)} compatible prompts; increase candidate-count")
    bank["prompts"] = compatible[:args.count]
    bank["requested_prompts"] = args.count
    bank["round_trip_compatible_candidates"] = len(compatible)
    bank["tokenizer_model_sha256"] = digest(args.tokenizer_path / "tokenizer.model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    payload = json.dumps(bank, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(payload)
    print(json.dumps({"path": str(args.output), "prompt_count": args.count,
                      "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "dataset_sha256": source_sha,
                      "tokenizer_model_sha256": bank["tokenizer_model_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
