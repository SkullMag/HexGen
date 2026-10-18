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

Run ID `13b-a10-full-001` completed with the frozen default rates 0.125,
0.25, 0.5, 1, 2, and 4 requests/second, 100 requests per rate, three warmups,
32 generated tokens, and arrival seed 20260919. All 600 measured requests
succeeded. The S3 summary, all six raw per-rate JSONL objects, and both ranks'
JSONL and text logs were listed and read back after the run. The raw request
files each contained exactly 100 successful records. The prompt-bank SHA-256
in the S3 summary matched the staged bank:
`ba0a6ab5367ff809b494be8b910edafc96d22d111a91227a58c1827adeb6b780`.

| Offered requests/s | Successes | p50 end-to-end (s) | p95 end-to-end (s) |
| ---: | ---: | ---: | ---: |
| 0.125 | 100/100 | 1.449 | 3.767 |
| 0.25 | 100/100 | 1.458 | 4.272 |
| 0.5 | 100/100 | 2.747 | 5.834 |
| 1 | 100/100 | 45.335 | 67.214 |
| 2 | 100/100 | 95.999 | 116.213 |
| 4 | 100/100 | 128.596 | 138.657 |

Latency rose sharply at 1 request/second and above as requests queued. This
single run establishes one homogeneous 13B reference; it does not by itself
support a heterogeneous comparison or a confidence interval.

Results are under
`s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/13b-a10-full-001/centralized-13b-a10-east/`.
The GPU server Job was deleted after final rank-log upload; the West model PVC
and S3 objects remain.

This 13B single-replica homogeneous trial is separate from the planned 70B
homogeneous/heterogeneous comparison.
