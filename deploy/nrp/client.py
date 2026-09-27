"""Small sequential native OCF smoke run; retain every request outcome in S3."""

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from s3_results import upload, upload_metadata

sys.path.insert(0, "/opt/hexgen/benchmark/send_request")
from request import request_head_node  # noqa: E402


PROMPTS = (
    "Explain why a distributed inference pipeline can be limited by network latency.",
    "Give three ways to measure a GPU inference service fairly.",
    "Summarize the difference between tensor and pipeline parallelism.",
    "What does GPU memory pressure mean during language model inference?",
)


async def one(index: int, phase: str) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    request_id = f"{os.environ['RUN_ID']}-{phase}-{index:04d}"
    prompt = PROMPTS[index % len(PROMPTS)]
    payload = {"model_name": os.environ["MODEL_NAME"] + "_0",
               "params": {"prompt": prompt, "max_new_tokens": 32,
                          "temperature": 1.0, "top_p": 0.0, "top_k": 1,
                          "request_id": request_id, "prompt_id": index}}
    result = {"request_id": request_id, "phase": phase, "prompt_id": index,
              "prompt": prompt, "started_utc": started}
    try:
        text, inference_s, native_client_s = await asyncio.wait_for(
            request_head_node(payload, os.environ["HEAD_NODE"], request_id), timeout=180)
        result.update(success=True, output_text=text, inference_s=inference_s,
                      native_client_s=native_client_s)
    except Exception as exc:
        result.update(success=False, error=repr(exc))
    result["end_to_end_latency_s"] = time.perf_counter() - start
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return result


async def run() -> int:
    folder = Path("/results")
    folder.mkdir(parents=True, exist_ok=True)
    upload_metadata(folder, "client", {"warmups": int(os.environ.get("WARMUPS", "3")),
                                       "requests": int(os.environ.get("REQUESTS", "20")),
                                       "new_tokens": 32, "execution": "sequential"})
    # Registration and model loading can take minutes. Preserve readiness failures.
    ready = False
    for _ in range(120):
        result = await one(0, "readiness")
        if result["success"]:
            ready = True
            break
        await asyncio.sleep(10)
    if not ready:
        path = folder / "readiness.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        upload(path, "client")
        return 1
    failed = 0
    for phase, count in (("warmup", int(os.environ.get("WARMUPS", "3"))),
                         ("measured", int(os.environ.get("REQUESTS", "20")))):
        path = folder / f"{phase}.jsonl"
        with path.open("w") as stream:
            for index in range(count):
                result = await one(index, phase)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                failed += not result["success"]
                print(json.dumps({"request_id": result["request_id"],
                                  "success": result["success"],
                                  "latency_s": result["end_to_end_latency_s"]}), flush=True)
        print(upload(path, "client"), flush=True)
    summary = folder / "summary.json"
    summary.write_text(json.dumps({"measured": int(os.environ.get("REQUESTS", "20")),
                                   "failed_including_warmup": failed}, indent=2) + "\n")
    print(upload(summary, "client"), flush=True)
    return int(failed > 0)


def main() -> int:
    return asyncio.run(run())
