# Homogeneous 13B A10 validation, October 4, 2026

## Placement and inputs

- One NRP East pod on `gpu-18.nrp.mghpcc.org` acquired two NVIDIA A10 GPUs
  (24 GiB each) for the smoke run. A separate full-sweep server pod acquired
  two A10s on `gpu-16.nrp.mghpcc.org`.
- The workers used one Llama-2-13B FP16 replica, TP 2, PP 1, 40 layers, and
  image `ghcr.io/skullmag/hexgen@sha256:c2a8a83d7ca6c5bce16514942a206e4e871dfafc0aa373f9942bd56b22b21391`.
- Checkpoint revision `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1`,
  converted weights, and the 500-prompt bank remained on the NRP West PVC
  `hexgen-model-west-13b`. The East pod mounted that PVC read-only. No model
  bytes were downloaded onto the submitting computer.
- The published image contains an older client that uses the OCF proxy route,
  which returned HTTP 500. The direct `_inference` route returned HTTP 200.
  `run.py` therefore mounts the current client source from a run-specific
  ConfigMap; the worker image digest remains unchanged.

## Smoke run

Run ID `13b-a10-smoke-001` used one offered rate of 1 request/second, two
measured requests, one warmup, and eight generated tokens. Both measured
requests succeeded (2/2); client end-to-end latency was 0.494 seconds p50
and 0.530 seconds p95. This is a functional smoke test, not a performance
estimate. The worker imported FlashAttention 2.0.8, was launched with
`--use-flash-attn`, loaded the model on both A10s, and completed inference.
The model's configuration reported `use_flash_attn: true`.

The S3 prefix `s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/13b-a10-smoke-001/centralized-13b-a10-east/`
was listed after the server exited. It contained the client config, metadata,
raw `rate-1.jsonl`, summary, both rank JSONL request logs, both rank text logs,
and worker metadata. The server Job was deleted after upload; the West model
PVC and S3 objects were retained.

## Full sweep

Run ID `13b-a10-full-001` was submitted with the frozen default rates
0.125, 0.25, 0.5, 1, 2, and 4 requests/second, 100 requests per rate, three
warmups, 32 generated tokens, and arrival seed 20260919. The run is not
complete until all six per-rate JSONL files, the summary, both rank logs, and
S3 upload are verified. Its S3 prefix is
`s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/13b-a10-full-001/centralized-13b-a10-east/`.

This 13B single-replica homogeneous trial is separate from the planned 70B
homogeneous/heterogeneous comparison.
