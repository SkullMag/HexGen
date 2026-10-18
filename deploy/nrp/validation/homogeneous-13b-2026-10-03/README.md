# Homogeneous Llama-2-13B trial, 2026-10-03

Status: **inputs staged; GPU experiment not run**. No inference latency or S3
request records exist for this trial.

The CPU-only NRP Job `hexgen-nrp-stage-llama13b` completed on
`rci-tide-cpu-04.sdsu.edu`. It downloaded the gated
`meta-llama/Llama-2-13b-hf` FP16 checkpoint at revision
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` directly to the Bound
100 GiB `rook-cephfs` PVC `hexgen-model-west-13b`. It also created the
500-prompt, 128-token bank and converted all 40 layers. The prompt-bank SHA-256
is `ba0a6ab5367ff809b494be8b910edafc96d22d111a91227a58c1827adeb6b780`.
The Job log ends with `Conversion complete: 40 layers` and
`Remote Llama-2-13B inputs ready`.

The pinned image lacks the prompt preparation script and `pyarrow`. The Job
mounts checked-in scripts from `hexgen-nrp-13b-scripts` and installs pinned
`pyarrow==18.1.0` in the CPU-only preparation container. This did not change
the inference image or download model data to the submitting computer.

Short, bounded GPU scheduler Jobs requested the indicated homogeneous shape.
Every probe stayed Pending and was deleted promptly. The tested shapes were:

| Region | GPU shape | Requested CPU / RAM | Scheduler result |
| --- | --- | --- | --- |
| West | 2 RTX 3090 | 4 CPU / 62 GiB, then 3 CPU / 42 GiB | Insufficient GPU, CPU, or RAM |
| Central and East | 2 A10 | 4 CPU / 62 GiB, then 3 CPU / 42 GiB | Insufficient GPU, CPU, or RAM |
| West | 1 L40 and 1 L40S | 3 CPU / 42 GiB, then 1.5 CPU / 42 GiB for L40 | Insufficient GPU or CPU |
| Central | 1 L40S | 1.5 CPU / 42 GiB | Insufficient GPU |
| Central, East, Mountain, Pacific | 2 RTX 3090 | 3 CPU / 42 GiB | Insufficient GPU or inaccessible nodes |
| West | 2 RTX A5000 | 3 CPU / 42 GiB | Inaccessible nodes |
| West | 2 RTX A4000 | 3 CPU / 42 GiB, then 1.5 CPU / 42 GiB | Insufficient GPU or CPU |
| West | 2 RTX 4090 | 3 CPU / 42 GiB, then 1.5 CPU / 42 GiB | Insufficient GPU or CPU |

These are scheduler results at the time of the check, not a claim that NRP has
no such hardware. The existing 70B monitor may find a placement later. No GPU
server or client Job was left running or Pending. The completed staging Job and
model PVC remain for another 13B attempt. FlashAttention on the proposed 13B
GPU shape and S3 result upload remain unverified until actual inference runs.
