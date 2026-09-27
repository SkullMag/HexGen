"""Run OCF, uneven GPU rank groups, or the native request client in NRP pods."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from s3_results import upload, upload_metadata

ROOT = Path("/opt/hexgen")
RESULTS = Path("/results")
STOP = False


def on_signal(_number, _frame):
    global STOP
    STOP = True


def numbers(name: str) -> list[int]:
    return [int(value) for value in os.environ[name].split()]


def worker() -> int:
    global STOP
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    count = int(os.environ["LOCAL_GPU_COUNT"])
    offset = int(os.environ["RANK_OFFSET"])
    world = int(os.environ["WORLD_SIZE"])
    tp = numbers("HETERO_CONFIG")
    layers = numbers("PP_PARTITION")
    if count < 1 or offset < 0 or offset + count > world or sum(tp) != world:
        raise ValueError("Invalid distributed rank layout")
    if len(tp) != len(layers) or sum(layers) != int(os.environ["MODEL_LAYERS"]):
        raise ValueError("Pipeline partition does not cover the model")
    if not Path(os.environ["CHECKPOINT_PATH"]).is_dir():
        raise FileNotFoundError("CHECKPOINT_PATH must be a mounted model directory")
    if not (Path(os.environ["STATE_DICTS_PATH"]) / "inv_freq.pt").is_file():
        raise FileNotFoundError("STATE_DICTS_PATH needs converted weights and inv_freq.pt")
    RESULTS.mkdir(parents=True, exist_ok=True)
    upload_metadata(RESULTS, "worker", {"ranks": list(range(offset, offset + count)),
                                        "hetero_config": tp, "pp_partition": layers})
    base = [sys.executable, "_llama_worker.py", "--model_size", os.environ["MODEL_SIZE"],
            "--mixed_precision", "fp16", "--fp16", "--num-layers", str(sum(layers)),
            "--num_hidden_layers", str(sum(layers)),
            "--max-position-embeddings", os.environ.get("SEQ_LENGTH", "2048"),
            "--seq-length", os.environ.get("SEQ_LENGTH", "2048"),
            "--micro-batch-size", "1", "--tensor-model-parallel-size", "1",
            "--pipeline-model-parallel-size", str(world), "--hetero_config",
            *map(str, tp), "--pp_partition", *map(str, layers),
            "--checkpoint-path", os.environ["CHECKPOINT_PATH"],
            "--state-dicts-path", os.environ["STATE_DICTS_PATH"],
            "--model_name", os.environ["MODEL_NAME"],
            "--head_node", os.environ["HEAD_NODE"], "--group_id", "0"]
    processes = []
    logs = []
    for local_rank in range(count):
        rank = offset + local_rank
        env = os.environ.copy()
        env.update(RANK=str(rank), LOCAL_RANK=str(local_rank), WORLD_SIZE=str(world),
                   CUDA_VISIBLE_DEVICES=str(local_rank), NCCL_IB_DISABLE="1")
        # One visible GPU makes Megatron's rank % device_count check resolve to zero.
        # The physical GPU selection is retained by CUDA_VISIBLE_DEVICES.
        env["LOCAL_RANK"] = "0"
        log = (RESULTS / f"rank-{rank}.log").open("w")
        logs.append(log)
        process = subprocess.Popen(base + ["--local-rank", "0", "--request-log",
                                           str(RESULTS / f"rank-{rank}.jsonl")],
                                   cwd=ROOT / "benchmark/hexgen_documents", env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
    last_upload = 0.0
    failed_upload = False
    try:
        while not STOP and all(p.poll() is None for p in processes):
            if time.monotonic() - last_upload >= 30:
                for path in RESULTS.glob("rank-*.jsonl"):
                    try:
                        upload(path, "worker")
                    except Exception as exc:
                        print(f"S3 upload pending for {path.name}: {exc}", file=sys.stderr, flush=True)
                last_upload = time.monotonic()
            time.sleep(2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in logs:
            log.close()
        for path in RESULTS.glob("rank-*"):
            try:
                print(upload(path, "worker"), flush=True)
            except Exception as exc:
                failed_upload = True
                print(f"S3 upload failed for {path.name}: {exc}", file=sys.stderr, flush=True)
    if failed_upload:
        return 2
    return 0 if STOP else 1


def main() -> int:
    mode = sys.argv[1]
    if mode == "ocf":
        os.execvp("ocf-core", ["ocf-core", "--config", "/etc/hexgen/ocf.yaml", "start"])
    if mode == "worker":
        return worker()
    if mode == "client":
        from client import main as client_main
        return client_main()
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
