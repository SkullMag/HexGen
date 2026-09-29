# NRP HexGen starter experiment

This is a small, capacity-matched comparison built on the native HexGen/OCF
path. It contains one image, a centralized four-A6000 Job, and a geographically
separated three-A6000 plus two-A10 layout. Both arms use one FP16 Llama 2 70B
replica, the same frozen prompts and Poisson arrival trace, and the same S3
record format. The [70B staging run](validation/homogeneous-70b-2026-09-27/README.md)
completed on NRP. A [September 28 homogeneous 70B attempt](validation/homogeneous-70b-2026-09-28/README.md)
validated one debug-mode request on four A6000s, but the six-rate benchmark
was blocked by intermittent CUDA and GPU device-plugin failures.

## Layout

| Arm | Placement | HexGen grouping | Nominal GPU VRAM |
| --- | --- | --- | --- |
| Centralized | one pod, 4 RTX A6000 48 GiB on `hcc-prp-c5038.unl.edu` | TP 4, PP 1; 80 layers | 192 GiB |
| Decentralized | 3 RTX A6000 at UNL plus 2 A10 24 GiB at MGH | TP `[2,1,2]`, PP layers `[40,20,20]` | 192 GiB |

The two placements were selected from a public availability snapshot; they are
not reservations. The exact hostname and region labels in the manifests must be
checked again before submitting. Required node selectors enforce both hostname
and region. The GPU toleration only permits scheduling on GPU-tainted nodes; it
does not enforce geography. This setup targets a similar nominal 70B FP16
weight-to-VRAM ratio in both arms. Real allocated memory and KV-cache headroom
are measured by the per-rank JSONL logs and may differ.

Each arm has 192 GB of nominal GPU memory; FP16 weights alone are roughly
140 GB, or 73% of that capacity. This matches **memory capacity**, not a verified
hourly dollar budget: the A10 is absent from the single-provider rate schedule
used in [the budget table](BUDGET.md). That table also gives several candidate
70B comparison pairs with cloud-equivalent costs. NRP capacity is shared, so
advertised GPU count is not free capacity.
The Jobs use `opportunistic` priority and can be preempted. [NRP's GPU guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/)
permits up to eight GPUs per node for Jobs; the four-GPU worker therefore runs
as a Job rather than a Deployment.

## Comparison protocol

Use the same pinned checkpoint and tokenizer, conversion procedure, image,
FP16 precision, greedy decoding, 128-token input prompt bank, 32 generated
tokens, warmup count, and offered request trace in both arms. Stage one
`prompt_bank.json` containing a `prompts` array of objects with `text` and
`prompt_id` at `/model/workload/prompt_bank.json` on the UNL model PVC. Build
it from a permitted, revision-pinned LMSYS first-user-prompt subset using the
Llama 2 tokenizer; verify input token lengths and record the SHA-256. The client
records this hash and uses the same seeded Poisson offsets at rates 0.125,
0.25, 0.5, 1, 2, and 4 requests/second in both arms. Its default is 100
requests per rate. After smoke checks, use at least three independent paired
runs, with a new run ID and trace seed per pair.

Report successes and failures, end-to-end p50/p95 latency, and SLO attainment
at thresholds fixed from the centralized idle reference **before** inspecting
heterogeneous results. Retain raw per-request arrival, submission, completion,
inference time, and output records, plus per-rank token IDs and peak allocated
VRAM. Compare only paired runs with equal prompt-bank hash, image, checkpoint,
request count, output length, and trace seed. One replica per arm tests placement
and asymmetric parallelism; it cannot reproduce the paper's benefit from
packing more independent replicas into the same dollar budget. This layout
also forces a cross-region pipeline edge, while the paper's optimized scheduler
usually avoided such edges within one replica. A slower heterogeneous arm
would be a valid finding.

## FlashAttention compatibility

The NRP workers enable [FlashAttention 2.0.8](https://github.com/Dao-AILab/flash-attention/blob/v2.0.8/README.md) for FP16 inference. The image pins
PyTorch 2.0.1, CUDA 11.7, and the matching upstream FlashAttention wheel and
builds its rotary and fused-dense helpers from the same release. The selected
RTX A6000 and A10 GPUs both have [CUDA compute capability 8.6](https://developer.nvidia.com/cuda/gpus), within the
release's Ampere support. Do not schedule this image's FlashAttention path on
Turing GPUs such as the Quadro RTX 6000.

On one NRP RTX A6000, the pinned OpenLLaMA 7B v2 checkpoint completed a
FlashAttention-enabled request after both packed-QKV and packed-KV CUDA kernel
probes passed. A subsequent [two-site 7B validation](validation/two-site-7b-2026-09-27/README.md)
completed two requests across an RTX A6000 at UNL and an A10 at MGH with
identical generated token IDs on both ranks. These validate the small model
paths; the 70B layouts above still need complete benchmark runs.

## Build

The same image contains the original HexGen worker/client, the OCF coordinator,
and the small NRP launcher. Build on a Linux amd64 host with Docker and enough
free space for PyTorch and FlashAttention:

```sh
docker buildx build --platform linux/amd64 -f deploy/nrp/Dockerfile \
  -t ghcr.io/skullmag/hexgen:nrp-$(git rev-parse HEAD) --push .
```

The branch also has a GitHub Actions image build that publishes the same SHA tag
to GHCR. Inspect its result before using the image. If the package is private,
create an NRP registry pull secret and add `imagePullSecrets` to each pod template.
The Docker image does not contain model weights, prompts, or S3 credentials.

## Prepare before deployment

1. Confirm a namespace with permission to create Services, Jobs, ConfigMaps,
   PVCs, and Secrets and with current capacity for each requested GPU type.
2. Obtain access to both gated Hugging Face repositories:
   `meta-llama/Llama-2-70b-hf` and `lmsys/chatbot_arena_conversations`.
   With approval to use the locally saved HF token, run
   `python3 deploy/nrp/setup_hf_secret.py --use-saved-token`; otherwise run it
   without that flag and enter a separate read token at its hidden prompt.
   It checks both access grants with HEAD requests, then installs only the token
   in the private `hexgen-hf-token` Kubernetes Secret. No model files are
   downloaded onto the submitting machine.
3. Create the homogeneous model volume on the UNL site:

   ```sh
   kubectl --context=nautilus -n nyu-networks apply -f deploy/nrp/model-pvc-central.yaml
   ```

   This 350 GiB `linstor-unl` ReadWriteOnce volume is mounted by both
   homogeneous Jobs on the same pinned UNL node. A two-pod mount test passed.
   NRP Linstor allocates requested capacity, so expand only if the checkpoint
   plus converted layer files approach the limit. The geographically separated
   arm will need another volume at MGH.
4. Stage inputs **on NRP**, using a CPU-only Job pinned to the UNL GPU node.
   It uses the pinned Llama 2 revision
   `3aba440b59558f995867ba6e1f58f21d0336b5bb`, downloads only the
   safetensors checkpoint and tokenizer to `/model/checkpoint`, downloads the
   pinned LMSYS Parquet to `/model/workload`, builds
   `/model/workload/prompt_bank.json`, and converts one layer at a time into
   `/model/converted`. The converter was checked against HexGen's original
   remapping on a synthetic GQA model. The actual 70B checkpoint and all 80
   converted layers were validated on NRP. A four-A6000 70B load and one
   request later passed with CUDA debugging enabled, while normal startup
   remained unstable; see the September 28 validation report above.

   ```sh
   IMAGE="ghcr.io/skullmag/hexgen:nrp-$(git rev-parse HEAD)"
   python3 deploy/nrp/stage.py apply --image "$IMAGE"
   kubectl --context=nautilus -n nyu-networks wait \
     --for=condition=complete job/hexgen-nrp-stage-central-llama70b --timeout=12h
   ```

   Check the staging Job logs and the checkpoint, prompt-bank, and conversion
   markers before requesting GPUs. The dataset revision is
   `1b6335d42a1d2c7e34870c905d03ab964f7f2bd8`; record the prompt-bank
   SHA-256 for both comparison arms.
5. Get an [NRP S3 token](https://nrp.ai/s3token/) for the West pool, then run
   the setup helper from a private terminal with `boto3` installed. It prompts
   without echoing the keys, creates a bucket, verifies an upload and readback,
   and installs the `hexgen-nrp-s3` Secret in `nyu-networks`:

   ```sh
   python3 deploy/nrp/setup_s3.py
   ```

   Use `--endpoint` for another NRP pool and `--bucket` to reuse a bucket.
   The Secret contains `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `S3_BUCKET`, and `S3_ENDPOINT_URL`. Never commit the token or Secret YAML.

## Render and submit

First run a one-request homogeneous smoke test from the repository root, using
the SHA tag created by the image build. Submit the server Jobs first, because
scarce GPU capacity can leave them Pending for an extended period. Submit the
client only after every server pod is Ready; `run.py` checks that condition.
After deleting the smoke Jobs, verify a successful client result and four rank
logs in S3 before starting the sweep:

If the server wait times out because GPUs are unavailable, delete that run's
server Jobs before leaving it unattended; otherwise they can acquire GPUs later
without a client.

```sh
IMAGE="ghcr.io/skullmag/hexgen:nrp-$(git rev-parse HEAD)"
python3 deploy/nrp/run.py apply --component servers --variant centralized --run-id central-smoke-001 \
  --image "$IMAGE" --requests 1 --rates '0.125' --warmups 0 --new-tokens 8
kubectl --context=nautilus -n nyu-networks wait \
  --for=condition=Ready pod -l job-name=hexgen-nrp-central-head-central-smoke-001 --timeout=60m
python3 deploy/nrp/run.py apply --component client --variant centralized --run-id central-smoke-001 \
  --image "$IMAGE" --requests 1 --rates '0.125' --warmups 0 --new-tokens 8
kubectl --context=nautilus -n nyu-networks wait \
  --for=condition=complete job/hexgen-nrp-central-client-central-smoke-001 --timeout=60m
python3 deploy/nrp/run.py delete --variant centralized --run-id central-smoke-001 \
  --image "$IMAGE"
```

Then run the full six-rate, 100-request-per-rate sweep:

```sh
IMAGE="ghcr.io/skullmag/hexgen:nrp-$(git rev-parse HEAD)"
python3 deploy/nrp/run.py render --variant centralized --run-id demo-001 --image "$IMAGE" > /tmp/hexgen-central.yaml
python3 deploy/nrp/run.py apply --component servers --variant centralized --run-id demo-001 --image "$IMAGE"
kubectl --context=nautilus -n nyu-networks wait \
  --for=condition=Ready pod -l job-name=hexgen-nrp-central-head-demo-001 --timeout=60m
python3 deploy/nrp/run.py apply --component client --variant centralized --run-id demo-001 --image "$IMAGE"
kubectl --context=nautilus -n nyu-networks wait --for=condition=complete job/hexgen-nrp-central-client-demo-001 --timeout=60m
python3 deploy/nrp/run.py delete --variant centralized --run-id demo-001 --image "$IMAGE"

python3 deploy/nrp/run.py apply --component servers --variant decentralized --run-id demo-001 --image "$IMAGE"
kubectl --context=nautilus -n nyu-networks wait \
  --for=condition=Ready pod -l job-name=hexgen-nrp-hetero-head-demo-001 --timeout=60m
kubectl --context=nautilus -n nyu-networks wait \
  --for=condition=Ready pod -l job-name=hexgen-nrp-hetero-east-demo-001 --timeout=60m
python3 deploy/nrp/run.py apply --component client --variant decentralized --run-id demo-001 --image "$IMAGE"
kubectl --context=nautilus -n nyu-networks wait --for=condition=complete job/hexgen-nrp-hetero-client-demo-001 --timeout=60m
python3 deploy/nrp/run.py delete --variant decentralized --run-id demo-001 --image "$IMAGE"
```

`run.py` requires PyYAML locally and checks the Secret and PVC names before
applying. It uses the Kubernetes context and namespace shown by default; override
with `--context` and `--namespace`. For later pairs, pass a new `--run-id` and
the same new `--arrival-seed` to both variants. The raw request records, rank logs, and
placement metadata are uploaded to
`s3://$S3_BUCKET/hexgen-nrp/<run-id>/<variant>/<pod-name>/`. Each request record
includes success, scheduled and actual arrival, inference time, and end-to-end
latency. The workers upload rank JSONL periodically and again on termination.
Keep the same `run-id` for a paired run. The one-provider client calls OCF's
direct `/_inference` route because the public route currently fails on same-peer
forwarding; routing across independent replicas remains a separate validation
item. Treat image-build, deployment, model-load, or S3 failures as preflight
failures. Do not report performance from a partial run.

After downloading both clients' `config.json` and `rate-*.jsonl` files into
separate local directories, choose and freeze a numeric SLO threshold from the
centralized low-load reference. The comparison tool checks model revision,
image, prompt-bank hash, trace seed, and every paired request before calculating
SLO attainment (failed requests count as misses):

```sh
python3 deploy/nrp/compare.py --central /path/to/central-client-results \
  --hetero /path/to/hetero-client-results --slo-seconds 10 \
  --output /path/to/new-paired-comparison.json
```

Replace `10` with the predeclared threshold for that experiment. The tool also
reports output agreement between successful paired requests. Its result is an
analysis artifact; preserve the raw S3 files and the actual GPU placement as
the measurement evidence.
