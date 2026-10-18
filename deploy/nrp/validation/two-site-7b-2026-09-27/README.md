# NRP two-site OpenLLaMA 7B validation — 2026-09-27 UTC

This is a small **correctness run**, not the planned budget-matched 70B
benchmark. The exact applied [manifest](manifest.yaml), readable
[worker script](worker.sh), [rank 0 metrics](rank0.jsonl),
[rank 1 metrics](rank1.jsonl), and [machine-readable summary](summary.json)
are saved here. No S3 credentials or model weights are in this directory.

## Placement and software

| Rank | Site / node | GPU | Layers |
| --- | --- | --- | --- |
| 0 | UNL, `hcc-prp-c5038.unl.edu` (`us-central`) | RTX A6000 48 GiB | 0–15 |
| 1 | MGH, `gpu-06.nrp.mghpcc.org` (`us-east`) | A10 24 GiB | 16–31 |

Both stages used tensor parallelism 1, pipeline parallelism 2, FP16, and
FlashAttention 2.0.8. The model was
`openlm-research/open_llama_7b_v2` at revision
`e5961def23172a2384543940e773ab676033c963`. The tested image was
`ghcr.io/skullmag/hexgen@sha256:c2a8a83d7ca6c5bce16514942a206e4e871dfafc0aa373f9942bd56b22b21391`,
built from commit `bb50d0e`. Each pod downloaded and converted the full
checkpoint in its ephemeral volume before serving.

## Observations

| Request | New tokens | Rank 0 inference | Rank 1 inference | Peak allocated VRAM, rank 0 / 1 |
| --- | ---: | ---: | ---: | ---: |
| `nrp-7b-two-site-fixed-1` | 8 | 2.781 s | 2.778 s | 6.29 / 6.29 GiB |
| `nrp-7b-two-site-fixed-2` | 12 | 0.771 s | 0.762 s | 6.29 / 6.29 GiB |

Both requests returned nonempty text, and the two ranks recorded identical
input and generated token IDs for each request. The first request includes
cold-path work; these two samples are insufficient for throughput or latency
comparison with a centralized cluster. An earlier run exposed a final-token
synchronization bug: rank 0 had appended its unsynchronized final token.
Commit `bb50d0e` broadcasts every sampled token before appending it, and the
recorded rerun verifies the fix.

## Reproduce

The manifest pins both nodes and the tested image, so check current GPU
availability, node labels, and namespace quota first. From the repository root:

```sh
kubectl --context=nautilus apply -f deploy/nrp/validation/two-site-7b-2026-09-27/manifest.yaml
kubectl --context=nautilus -n nyu-networks get pods -l 'app in (hexgen-7b-two-site-head,hexgen-7b-two-site-east)' -w
```

Wait for both workers to load and register. The head Service must report both
peers at `/api/v1/status/peers`. Send requests from the head pod to OCF's direct
request route, with `model_name` set to `NRP-OpenLLaMA-7B-TwoSite_0` and
`params` containing `prompt`, `max_new_tokens`, `temperature: 1.0`,
`top_p: 0.0`, `top_k: 1`, `request_id`, and `prompt_id`. For example:

```sh
HEAD_POD=$(kubectl --context=nautilus -n nyu-networks get pod -l app=hexgen-7b-two-site-head -o jsonpath='{.items[0].metadata.name}')
kubectl --context=nautilus -n nyu-networks exec "$HEAD_POD" -- \
  curl -fsS -H 'Content-Type: application/json' \
  -d '{"model_name":"NRP-OpenLLaMA-7B-TwoSite_0","params":{"prompt":"The capital of France is","max_new_tokens":8,"temperature":1.0,"top_p":0.0,"top_k":1,"request_id":"manual-two-site-1","prompt_id":1}}' \
  http://127.0.0.1:8092/api/v1/request/_inference
```

The tested public `/inference` route failed for a same-peer request with
`dial to self attempted`; the direct OCF route above completed. The manifest
has a one-hour active deadline for each Job. When finished, save the metrics
from `/scratch/rank0.jsonl` and `/scratch/rank1.jsonl`, then release both GPUs:

```sh
kubectl --context=nautilus delete -f deploy/nrp/validation/two-site-7b-2026-09-27/manifest.yaml
```

The Jobs, Service, and ConfigMap from the recorded run were deleted, and no
matching pods remained. This run saved raw metrics in the repository; it did
not test S3 upload. The larger centralized/decentralized manifests and S3
workflow described in the parent guide remain to be validated.
