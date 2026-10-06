# Homogeneous 70B NRP staging and scheduling attempt

## Verified staging

- NRP context and namespace: `nautilus/nyu-networks`.
- Job: `hexgen-nrp-stage-central-llama70b`, completed successfully in 83 minutes with zero pod restarts on `hcc-prp-c5038.unl.edu`.
- Image: `ghcr.io/skullmag/hexgen:nrp-64b8b4eef56fadc4524fb4d53bb833f48123cc79` (observed pod digest `sha256:e33cd00ca31dd060e6cf183b564a8e8c84bf3a9dc6ad15283e481e2f2e8ae653`).
- Model: `meta-llama/Llama-2-70b-hf` revision `3aba440b59558f995867ba6e1f58f21d0336b5bb`; all 15 safetensors shards downloaded onto the 350 GiB `hexgen-model-unl` NRP PVC.
- Dataset: `lmsys/chatbot_arena_conversations` revision `1b6335d42a1d2c7e34870c905d03ab964f7f2bd8`; downloaded Parquet SHA-256 `3726a6352e9bfc34e206460646f6e5e99bb837751966a671ddd30c7f64e5b06e` matched the pinned value.
- Prompt bank: 500 prompts at `/model/workload/prompt_bank.json`, SHA-256 `e7a7f50d8dfcb736a9c7b444ef0a92aded36eab8a6c5d04c8c257d395fa41a51`.
- HexGen conversion: all 80 layers completed under `/model/converted`; the Job checked `conversion_complete.json` before exiting.
- The model and dataset were downloaded and converted on NRP. No model files were transferred to the submitting Mac. The temporary Hugging Face token Secret was deleted after staging completed.

## GPU smoke attempt

The one-request centralized smoke attempt used run ID `central70b-smoke-0928`, four RTX A6000 GPUs, and eight generated tokens. The head Job remained Pending on the pinned UNL node. Kubernetes reported insufficient `nvidia.com/rtxa6000` and CPU among candidate nodes. A separate short-lived Job requesting four RTX A6000 GPUs with only one CPU and 1 GiB memory also remained Pending with `Insufficient nvidia.com/rtxa6000`, confirming that GPU scheduling capacity was the immediate blocker. The public NRP dashboard reported six available A6000s on the node at one snapshot, but the Kubernetes scheduler did not admit a four-GPU pod; dashboard figures are not a scheduling guarantee.

The waiting client logged a configuration upload under `s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/central70b-smoke-0928/centralized/`. It sent no successful inference request, so there are no raw request latencies or four-rank inference logs from this attempt. The diagnostic Job, waiting client, pending head, Service, and ConfigMap were deleted. The completed model PVC and S3 Secret remain for a capacity retry; no GPU Job is active.

## Many-GPU scheduling follow-up (September 27)

[NRP's many-GPU guidance](https://nrp.ai/documentation/userdocs/running/gpu-pods/#requesting-many-gpus) says a controller automatically adds a toleration for nodes reserved for four- or eight-GPU Jobs. The head is already a Job requesting four `nvidia.com/rtxa6000` devices, and its observed pod had the `nautilus.io/hardware=large-gpu:NoSchedule` toleration. The automatic toleration does not reserve four available devices or bypass CPU, memory, storage locality, or the explicit hostname selector.

The September 27 public node-inventory snapshot showed only two RTX A6000 nodes in US Central / UNL. `hcc-prp-c5038.unl.edu` reported five available GPUs, 11 available CPUs, and about 88 GiB available RAM. The head requests 13 CPUs and 146 GiB RAM across its coordinator and worker, so it would still not fit even if the dashboard GPU count matched scheduler allocatable capacity. `hcc-prp-c5036.unl.edu` reported one available GPU and had unreachable/issue taints. Other A6000 nodes were in US West; the staged 350 GiB model PVC uses the [UNL-specific `linstor-unl` storage class](https://nrp.ai/documentation/userdocs/storage/linstor/). Its PVC was verified Bound. The namespace cannot read the underlying PersistentVolume's node affinity, so cross-site mounting has not been verified.

## Retry gate

Wait for a UNL A6000 node to have at least four schedulable GPUs, 13 CPUs, and 146 GiB RAM, or restage the model on storage local to another feasible four-A6000 node. Recheck scheduler events after submitting the server Job; public dashboard counts are only a prefilter. Submit the client only after the server pod is Ready. Keep the same image, model revision, prompt-bank hash, arrival seed, and request parameters for paired homogeneous and heterogeneous runs. A scheduled and completed GPU inference request, with client and all four rank logs in S3, is still required before starting the full rate sweep.
