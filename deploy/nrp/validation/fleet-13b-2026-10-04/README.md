# Two-replica 13B comparison: prepared, awaiting placement

The previous 13B comparison put one pipeline stage in US East and the next
stage in US West. This design instead keeps every complete FP16 replica on one
node. The homogeneous fleet uses two 2×A10 tensor-parallel replicas at MGH
(one campus); the heterogeneous fleet uses one 2×A10 replica at MGH and one
2×RTX 3090 replica at a West site. The load client sends request indices
alternately to two independent OCF coordinators in **both** arms. No inference
tensor crosses regions. The heterogeneous arm still adds client-to-West
request/response latency, which should be measured and reported.

Both arms have two replicas, four 24 GiB GPUs, and 96 GiB nominal total VRAM.
Each replica has 48 GiB for approximately 26 GiB FP16 weights, or about 54%
weight pressure before KV cache and runtime buffers. Under the existing AWS
one-GPU proxy table, each four-card fleet is $4.024/hour. RTX 3090 is mapped
to an A10G-priced AWS instance for budgeting; that is **not** a claim of equal
GPU speed or actual NRP cost.

`fleet_13b.py` renders two isolated Services and GPU Jobs plus one client Job.
The client uses the same frozen prompt bank, arrival offsets, 32 generated
tokens, and per-request S3 format as the prior trial. It records which replica
handled each request. Before the full sweep, run a small normal-mode smoke on
both replicas and verify full-model loading and FlashAttention 2 on RTX 3090;
only packed-kernel probes, not 13B inference, have run on that card. Freeze
SLO deadlines from the new homogeneous fleet before inspecting the new
heterogeneous result. Use the same run ID, trace seed, client node, model
revision, image digest, and prompt-bank hash for both arms, then repeat with
independent seeds if placement persists. Keep the old single-replica 13B runs
as exploratory evidence, not the baseline for this fleet comparison.

After a simultaneous scheduler gate passes, select the actual hostnames and
run each arm **sequentially** with the same `--run-id`. For each arm, use
`python3 deploy/nrp/fleet_13b.py apply --component servers` with `--arm`,
`--run-id`, `--image`, `--hosts` (the selected East A10 host and second local
replica host), and `--client-node`. Check that both GPU Jobs are Running, both
replicas have loaded the model, and their rank logs show FlashAttention active.
Then submit `apply --component client` with the same arguments. First use
`--requests 2 --rates '0.125' --warmups 2 --new-tokens 8` for a smoke run;
use a new run ID and the default 100-request, six-rate workload for the full
pair. Verify all raw S3 records and rank logs before `delete --component all`.
The script retains the shared OCF ConfigMap, model PVC, and S3 results. Results
appear under `hexgen-nrp/<run-id>/<variant>/`, where variants are
`centralized-13b-fleet-a10` and `decentralized-13b-fleet-a10-3090`.

## Live gate on October 4

No two-replica comparison was run. All short scheduler Jobs were deleted:

| Probe | Outcome |
| --- | --- |
| Four A10s on one East node, 6 CPU, 84 GiB | Pending: insufficient GPUs or memory |
| Two RTX 3090s on one West node, 3 CPU, 42 GiB | Pending: insufficient GPUs, CPU, or memory |
| Two L40S on one Central node | Pending: insufficient GPUs or CPU |
| One L40 in West | Pending: insufficient GPUs |
| Two RTX 5000 Ada in West | Pending: insufficient GPUs or memory |
| One L40S in Central | Pending: insufficient GPUs |
| Two independent 2×A10 East Jobs and one 2×L4 West Job | All Pending: insufficient GPUs or memory |

The two-A10-plus-two-3090 fleet is the selected comparable design. Its
[`fleet-13b-probes.yaml`](../../fleet-13b-probes.yaml) gate requests two
2×A10 pods and one 2×RTX-3090 pod simultaneously; the A10 and West 3090
requests were individually unschedulable in the checks above. When the gate
passes, read actual assigned hosts, delete the probes, inspect node health and
storage locality, then render the fleet with those exact hostnames. Do not
leave Pending GPU Jobs unattended. Scheduler admission is not a successful
model load or completed experiment.

## October 5 placement and homogeneous smoke

A simultaneous gate briefly admitted two 2×A10 pods at MGH East on
`gpu-11.nrp.mghpcc.org` and `gpu-12.nrp.mghpcc.org`, plus a 2×RTX 3090 pod at
SDSC West on `ry-gpu-03.sdsc.optiputer.net`. All three probe Jobs were deleted.
The homogeneous smoke `13b-fleet-smoke-1005` then started two independent local
2×A10 replicas on the East nodes with the pinned image and staged 13B FP16
checkpoint. Both loaded the model, both rank-0 logs reported
`use_flash_attn: true`, and both replicas handled requests. With two warmups,
two measured requests at 0.125 requests/s, and eight generated tokens, the
client reported 2/2 successful measured requests. Its S3 upload log named
`hexgen-nrp/13b-fleet-smoke-1005/centralized-13b-fleet-a10/` and included
`config.json`, `rate-0p125.jsonl`, and `summary.json`. Those objects were not
independently read back in this smoke. The GPU and client Jobs were deleted.

Immediately afterward, the heterogeneous smoke's East A10 Job scheduled but
the West RTX 3090 Job stayed Pending: the selected node reported insufficient
GPU. Both Jobs were deleted without submitting a client. A new simultaneous
gate placed two East A10 pairs on `gpu-16.nrp.mghpcc.org` and
`gpu-11.nrp.mghpcc.org`; the unpinned West RTX 3090 pair stayed Pending. All
gate Jobs and Pods were deleted. This is an incomplete smoke, not a completed
paired fleet comparison; no full six-rate fleet sweep ran.
