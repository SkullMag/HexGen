# September 28 homogeneous 70B A6000 attempt

The full six-rate homogeneous benchmark **did not complete**. One diagnostic
70B request succeeded on four RTX A6000 GPUs, but normal startup failed twice
with a CUDA launch error, and a later four-GPU allocation was rejected because
NRP's device plugin reported a lost GPU. No GPU pods remain in `nyu-networks`.

## Fixed inputs

- Node: `hcc-prp-c5038.unl.edu`; four RTX A6000 48 GiB GPUs in one pod.
- Model: `meta-llama/Llama-2-70b-hf`, revision
  `3aba440b59558f995867ba6e1f58f21d0336b5bb`, FP16, TP 4 / PP 1.
- Image: `ghcr.io/skullmag/hexgen@sha256:e33cd00ca31dd060e6cf183b564a8e8c84bf3a9dc6ad15283e481e2f2e8ae653`.
- Prompt bank SHA-256:
  `e7a7f50d8dfcb736a9c7b444ef0a92aded36eab8a6c5d04c8c257d395fa41a51`.
- Model files and prompt bank stayed on the NRP `hexgen-model-unl` PVC.

## Attempts and evidence

1. `central70b-a6000-smoke-0928`: the first pod was Pending with
   `Insufficient cpu` at the pinned node. The worker CPU request was reduced
   from 12 to 6 (the coordinator requests 1) and the pod scheduled. During
   checkpoint loading, rank 3 exited with `CUDA error: unspecified launch
   failure` at `rotary_emb.inv_freq.copy_`. No client request was sent.
2. `hexgen-a6000-diag-0928`: a four-GPU PyTorch check allocated and wrote
   38.07 GiB on each card and completed FP16 matrix multiplication on all
   four cards. This checked memory and basic compute, not the full model load.
3. `central70b-a6000-debug-0928`: with `CUDA_LAUNCH_BLOCKING=1` for startup
   and inference, all four ranks loaded the 70B model and returned the same
   eight generated token IDs for one request. The client recorded one success
   at 0.125 requests/s with 1.417 s end-to-end latency; each rank recorded
   35,320,196,096 peak allocated bytes. The debug setting makes this a path
   validation, **not** a comparable performance result. The client config,
   raw request JSONL, summary, and four rank logs/JSONLs are in
   `s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/central70b-a6000-debug-0928/centralized/`.
4. `central70b-a6000-full-0928`: normal startup without CUDA debugging
   scheduled and admitted four GPUs, then rank 1 failed at the same checkpoint
   copy step with `CUDA error: unspecified launch failure`. The client was
   stopped before any benchmark rate finished. Logs are under the matching
   `hexgen-nrp/central70b-a6000-full-0928/centralized/` S3 prefix. Do not
   report this as a completed sweep.
5. `central70b-a6000-sync-smoke-0928`: a proposed checkpoint-copy
   synchronization test was rejected before container startup:
   `UnexpectedAdmissionError`, device-plugin NVLink query for devices `(3, 0)`:
   `GPU is lost`. Its unvalidated code override was removed.

After these attempts, Kubernetes still marked the node Ready, but its
`nvidia.com/rtxa6000` allocatable count fell from 7 to 5. The public resource
table did not show another unreserved, healthy four-card 48 GiB node at the
time of the check. The issue needs NRP node repair or another authorized
four-card placement before a full FP16 70B benchmark can be trusted.

The Kubernetes Jobs, Services, and test ConfigMaps were deleted. The staged
model PVC and successful diagnostic results in S3 were retained.
