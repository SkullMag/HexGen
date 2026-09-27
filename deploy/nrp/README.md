# NRP HexGen starter experiment

This is the first runnable packaging pass for the native HexGen/OCF path. It
contains one image, a centralized four-A6000 pod, and a geographically separated
three-A6000 plus two-A10 layout. Both arms use FP16 Llama 2 70B, one provider
replica, 80 layers, the same 32-token sequential request set, and the same S3
record format. They are **not** measured results. GPU allocation, model loading,
end-to-end execution and the cross-region network path still need validation on
NRP before treating either arm as a benchmark.

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

1. Confirm a namespace with quota for the requested GPU resource names and
   permission to create Deployments, Services, Jobs, ConfigMaps, and Secrets.
2. Obtain legal access to the Llama 2 70B Hugging Face checkpoint. Stage the
   same pinned revision at `/model/checkpoint` on the model volumes, including
   config and tokenizer. Prepare HexGen's converted layer files at
   `/model/converted/separate_state_dicts/` and
   `/model/converted/inv_freq.pt`. Use the repository's
   `hexgen/llama/load_model_parameters_utils/create_separate_state_dicts_llama_7b.py`
   converter with `--checkpoint-path` and
   `--save-dir /model/converted/separate_state_dicts`; despite its filename it
   takes the layer count from the model config. Conversion of 70B has not yet
   been validated and needs substantial CPU RAM and disk. Check the output before
   scheduling GPUs.
3. Provide namespace PVCs `hexgen-model-unl` and `hexgen-model-mgh`, with the
   checkpoint and converted files at the paths above. Verify each PVC can mount
   at its selected site. The centralized arm only needs the UNL PVC.
4. Create a bucket using an [NRP S3 token](https://nrp.ai/s3token/). Create the
   Kubernetes Secret `hexgen-nrp-s3` with keys `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, and `S3_ENDPOINT_URL`. Example, using
   local environment variables without writing credentials to this repository:

   ```sh
   kubectl --context=nautilus -n nyu-networks create secret generic hexgen-nrp-s3 \
     --from-literal=AWS_ACCESS_KEY_ID="$NRP_S3_ACCESS_KEY_ID" \
     --from-literal=AWS_SECRET_ACCESS_KEY="$NRP_S3_SECRET_ACCESS_KEY" \
     --from-literal=S3_BUCKET="$NRP_S3_BUCKET" \
     --from-literal=S3_ENDPOINT_URL="https://s3-west.nrp-nautilus.io"
   ```

   Protect your shell history and environment. Use the matching endpoint for
   the bucket's NRP pool. Never commit a token or the resulting Secret YAML.

## Render and submit

Run these from the repository root, using the SHA tag created by the image build:

```sh
IMAGE="ghcr.io/skullmag/hexgen:nrp-$(git rev-parse HEAD)"
python3 deploy/nrp/run.py render --variant centralized --run-id demo-001 --image "$IMAGE" > /tmp/hexgen-central.yaml
python3 deploy/nrp/run.py apply --variant centralized --run-id demo-001 --image "$IMAGE"
kubectl --context=nautilus -n nyu-networks wait --for=condition=complete job/hexgen-nrp-central-client-demo-001 --timeout=60m
python3 deploy/nrp/run.py delete --variant centralized --run-id demo-001 --image "$IMAGE"

python3 deploy/nrp/run.py apply --variant decentralized --run-id demo-001 --image "$IMAGE"
kubectl --context=nautilus -n nyu-networks wait --for=condition=complete job/hexgen-nrp-hetero-client-demo-001 --timeout=60m
python3 deploy/nrp/run.py delete --variant decentralized --run-id demo-001 --image "$IMAGE"
```

`run.py` requires PyYAML locally and checks the Secret and PVC names before
applying. It uses the Kubernetes context and namespace shown by default; override
with `--context` and `--namespace`. The raw request records, rank logs, and
placement metadata are uploaded to
`s3://$S3_BUCKET/hexgen-nrp/<run-id>/<variant>/<pod-name>/`. Each request record
includes success, inference time, and end-to-end latency. The workers upload
rank JSONL periodically and again on termination. Keep the same `run-id` for a
paired run and compare only matching successful request sets.

This first pass sends requests sequentially to establish correctness and memory
headroom. A Poisson load sweep, independent output validation, and actual cloud
price comparison are the next benchmark step after both layouts complete a smoke
run. Treat image-build or deployment failures as part of the preflight; do not
report performance from a partial run.
