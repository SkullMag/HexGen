"""Replay one fixed prompt bank and Poisson arrival trace against each NRP arm."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import aiohttp

from s3_results import upload, upload_metadata


RESULTS = Path("/results")


def prompts():
    raw = Path(os.environ["PROMPTS_FILE"]).read_bytes()
    bank = json.loads(raw)
    if not isinstance(bank, dict) or not bank.get("model_revision") or not bank.get("tokenizer_model_sha256"):
        raise ValueError("PROMPTS_FILE needs a pinned model revision and tokenizer hash")
    items = bank["prompts"]
    if not items or not all(isinstance(item.get("text"), str) and item["text"]
                            for item in items):
        raise ValueError("PROMPTS_FILE needs nonempty prompts with text")
    metadata = {"model_revision": bank["model_revision"],
                "tokenizer_model_sha256": bank["tokenizer_model_sha256"]}
    return items, hashlib.sha256(raw).hexdigest(), metadata


def arrival_offsets(count, seed):
    rng = random.Random(seed)
    elapsed = 0.0
    offsets = []
    for _ in range(count):
        elapsed += rng.expovariate(1.0)
        offsets.append(elapsed)
    return offsets


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def request(session, item, index, request_id, scheduled_at, origin, new_tokens, head=None):
    submitted = time.perf_counter()
    payload = {
        "model_name": os.environ["MODEL_NAME"] + "_0",
        "params": {"prompt": item["text"], "max_new_tokens": new_tokens,
                   "temperature": 1.0, "top_p": 0.0, "top_k": 1,
                   "request_id": request_id,
                   "prompt_id": item.get("prompt_id", index)},
    }
    record = {"request_id": request_id, "prompt_id": payload["params"]["prompt_id"],
              "request_index": index, "scheduled_offset_s": scheduled_at - origin,
              "submitted_offset_s": submitted - origin,
              "submission_lag_s": submitted - scheduled_at}
    if head is None:
        head = os.environ["HEAD_NODE"]
    record["replica_index"] = index % len(os.environ.get("HEAD_NODES", head).split())
    try:
        # The auto-routing endpoint currently fails when forwarding to its own peer.
        endpoint = head.rstrip("/") + "/api/v1/request/_inference"
        async with session.post(endpoint, json=payload) as response:
            result = await response.json()
            if response.status >= 400 or "error" in result:
                raise RuntimeError(f"OCF HTTP {response.status}: {result.get('error', result)}")
        output, inference_s = json.loads(result["data"])
        if not isinstance(output, str) or not output or not isinstance(inference_s, (int, float)) or inference_s <= 0:
            raise ValueError("Invalid inference response")
        record.update(success=True, output_text=output, inference_s=inference_s)
    except Exception as exc:
        record.update(success=False, error=repr(exc))
    finished = time.perf_counter()
    record.update(finished_offset_s=finished - origin,
                  client_elapsed_s=finished - submitted,
                  end_to_end_latency_s=finished - scheduled_at)
    return record


async def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    bank, bank_sha256, bank_metadata = prompts()
    run_id = os.environ["RUN_ID"]
    rates = [float(rate) for rate in os.environ.get("RATES", "0.125 0.25 0.5 1 2 4").split()]
    count = int(os.environ.get("REQUESTS", "100"))
    warmups = int(os.environ.get("WARMUPS", "3"))
    new_tokens = int(os.environ.get("NEW_TOKENS", "32"))
    seed = int(os.environ.get("ARRIVAL_SEED", "20260919"))
    heads = os.environ.get("HEAD_NODES", os.environ["HEAD_NODE"]).split()
    if not heads or any(not head.startswith("http://") for head in heads):
        raise ValueError("HEAD_NODES must list HTTP coordinator addresses")
    effective_warmups = max(warmups, len(heads))
    if not rates or any(rate <= 0 for rate in rates) or count < 1 or warmups < 0 or new_tokens < 1:
        raise ValueError("Invalid load sweep configuration")
    unit_offsets = arrival_offsets(count, seed)
    config = {"run_id": run_id, "variant": os.environ["VARIANT"],
              "model_name": os.environ["MODEL_NAME"],
              "image_ref": os.environ.get("IMAGE_REF"),
              **bank_metadata,
              "prompt_bank_sha256": bank_sha256, "prompt_count": len(bank),
              "rates_per_second": rates, "requests_per_rate": count,
              "warmups": effective_warmups, "new_tokens": new_tokens, "arrival_seed": seed,
              "unit_rate_offsets_s": unit_offsets,
              "request_endpoint": "/api/v1/request/_inference",
              "replica_count": len(heads), "routing": "request-index-round-robin",
              "started_utc": datetime.now(timezone.utc).isoformat()}
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    upload_metadata(RESULTS, "client", config)
    print(upload(RESULTS / "config.json", "client"), flush=True)
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("REQUEST_TIMEOUT_S", "1800")))
    summary = []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        # A successful inference proves that the model has registered and loaded.
        for replica_index, head in enumerate(heads):
            ready = False
            for attempt in range(120):
                now = time.perf_counter()
                result = await request(session, bank[0], replica_index,
                                       f"{run_id}-readiness-{replica_index}-{attempt}",
                                       now, now, new_tokens, head)
                if result["success"]:
                    ready = True
                    break
                await asyncio.sleep(10)
            if not ready:
                (RESULTS / "readiness.json").write_text(json.dumps(result, indent=2) + "\n")
                upload(RESULTS / "readiness.json", "client")
                return 1
        for index in range(effective_warmups):
            now = time.perf_counter()
            result = await request(session, bank[index % len(bank)], index,
                                   f"{run_id}-warmup-{index}", now, now, new_tokens,
                                   heads[index % len(heads)])
            if not result["success"]:
                raise RuntimeError(f"Warmup failed: {result['error']}")
        for rate in rates:
            label = format(rate, "g").replace(".", "p")
            path = RESULTS / f"rate-{label}.jsonl"
            origin = time.perf_counter()
            records = []
            lock = asyncio.Lock()

            async def one(index, offset):
                scheduled_at = origin + offset / rate
                await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
                result = await request(session, bank[index % len(bank)], index,
                                       f"{run_id}-rate-{label}-{index:04d}",
                                       scheduled_at, origin, new_tokens,
                                       heads[index % len(heads)])
                async with lock:
                    records.append(result)
                    with path.open("a") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")

            await asyncio.gather(*(one(index, offset) for index, offset in enumerate(unit_offsets)))
            print(upload(path, "client"), flush=True)
            latencies = [record["end_to_end_latency_s"] for record in records if record["success"]]
            summary.append({"rate_per_second": rate, "offered_requests": count,
                            "successful_requests": len(latencies),
                            "p50_end_to_end_s": percentile(latencies, 0.5),
                            "p95_end_to_end_s": percentile(latencies, 0.95),
                            "mean_end_to_end_s": statistics.mean(latencies) if latencies else None,
                            "results_file": path.name})
            print(json.dumps(summary[-1]), flush=True)
    path = RESULTS / "summary.json"
    path.write_text(json.dumps({**config, "finished_utc": datetime.now(timezone.utc).isoformat(),
                                "rates": summary}, indent=2) + "\n")
    print(upload(path, "client"), flush=True)
    return 0


def main():
    return asyncio.run(run())
