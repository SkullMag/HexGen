# Two-site heterogeneous 13B trial, October 4, 2026

## Placement gate

The NRP `nautilus` scheduler admitted one A10 at `gpu-09.nrp.mghpcc.org`
in US East and one RTX 5000 Ada at `k8s-gpu-6.ucsc.edu` in US West. Both
mounted the Bound `hexgen-model-west-13b` CephFS claim and ran the pinned
image `ghcr.io/skullmag/hexgen@sha256:c2a8a83d7ca6c5bce16514942a206e4e871dfafc0aa373f9942bd56b22b21391`.
Small packed-QKV and packed-KV FlashAttention 2.0.8 CUDA kernel probes passed
on each card. The RTX 5000 reported compute capability 8.9 and 32,760 MiB
VRAM. All scheduler probes were deleted promptly.

A West RTX 3090 (24 GiB) also passed both kernel probes, but its capacity
disappeared after the probe. Subsequent West, Central, East, Mountain, and
Pacific RTX 3090 placements did not pass the scheduler gate. The Central
candidates had untolerated remediation/issue taints, the Pacific node was
reserved, and the remaining sites lacked free GPU, CPU, or memory. West
A4000, L4, and RTX 4090 alternatives were also unschedulable. No Pending
GPU Job was retained while selecting the actual placement.

## Model and protocol

The run serves the same Llama-2-13B FP16 checkpoint revision
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` as the completed
homogeneous A10 run, with the same 500-prompt bank SHA-256
`ba0a6ab5367ff809b494be8b910edafc96d22d111a91227a58c1827adeb6b780`.
The two workers use TP groups `[1,1]` and PP layers `[16,24]`; rank 0 is the
East A10 and rank 1 is the West RTX 5000. The load client is pinned to
`usra-sti-01.uah.edu`, the same client node as the homogeneous run. No model
data was downloaded to the submitting computer.

The heterogeneous pair has 24 + 32 = 56 GiB nominal VRAM, compared with
2 × 24 = 48 GiB for the homogeneous pair. GPU type, topology, parallel
partition, and available memory all differ. Therefore any latency difference
is an exploratory system-level comparison, not an isolated GPU-type effect,
cost-matched pair, or reproduction of the paper's independent replicas.

## Smoke and full run

The two-request normal-mode smoke on run ID `13b-hetero-ada-smoke-001`
completed 2/2 measured requests at 1 offered request/second, with one warmup
and eight generated tokens. Client p50 was 1.474 seconds and p95 was 1.492
seconds. The client uploaded raw `rate-1.jsonl` and `summary.json` to S3.

The same server pair completed the frozen full sweep: six rates (0.125,
0.25, 0.5, 1, 2, 4 requests/second), 100 requests per rate, three warmups,
32 generated tokens, and arrival seed 20260919. All 600 measured requests
succeeded. Its client was a separate Pod
`hexgen-nrp-13b-hetero-client-13b-hetero-ada-smoke-001-5dw5f` under the same
run ID; the S3 pod subdirectories distinguish smoke and full records.

| Offered requests/s | Successes | p50 end-to-end (s) | p95 end-to-end (s) |
| ---: | ---: | ---: | ---: |
| 0.125 | 100/100 | 5.410 | 18.151 |
| 0.25 | 100/100 | 92.999 | 146.750 |
| 0.5 | 100/100 | 250.835 | 306.148 |
| 1 | 100/100 | 365.905 | 406.767 |
| 2 | 100/100 | 258.954 | 434.709 |
| 4 | 100/100 | 449.867 | 459.996 |

The S3 summary and all six raw JSONL files were read back after completion.
Each raw file contained 100 successful records, and the summary prompt-bank
SHA-256 matched the homogeneous trial. Both ranks' JSONL and text logs were
uploaded and listed with nonzero sizes. The client and GPU Jobs were deleted
after verification; the West model PVC and S3 results were retained.

The cross-region pipeline was much slower than the single-node two-A10
reference, including at the lowest offered rate. At higher rates, substantial
queueing makes p50 non-monotonic across this one sweep. This is one
exploratory run with different GPU capacity and parallelism; it does not
establish a cost-matched or statistically replicated comparison. The paper's
independently placed replicas were not reproduced.

S3 prefix:
`s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/13b-hetero-ada-smoke-001/decentralized-13b-a10-rtx5000/`.
