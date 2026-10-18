# Deliverable 2: AWS-equivalent GPU budgets for NRP

Checked 2026-09-29. This is a planning estimate, not an NRP bill, an AWS
deployment quote, or an NRP reservation. NRP counts come from live Ready
Kubernetes nodes with GPU-specific resource capacity. Reservation, issue, and
testing taints are excluded. Node inventory is not free GPU or CPU capacity.

## One-provider pricing method

Use **AWS EC2 us-east-1 on-demand** published prices for every row and
comparison total. An official AWS example gives g5.xlarge (1 A10G, 24 GB) at
**$1.006/instance-hour**, g6.xlarge (1 L4, 24 GB) at **$0.805/hour**, and
g6e.xlarge (1 L40S, 48 GB) at **$1.861/hour** (May 2026 prices). These are
whole-instance prices, including bundled CPU and RAM. AWS does not offer most
of the exact NRP cards, so the AWS GPU is a **VRAM/architecture proxy**, not a
performance equivalent. In particular, a 3090 is not an A10G, and a 16 GB
A4000 is not a 24 GB L4. Multiplying one-GPU instance rates estimates an
AWS-equivalent pool of one-GPU instances, not the actual price of a multi-GPU
AWS node. Storage and network are excluded. Recheck rates before a later run.

| NRP GPU | Nominal VRAM/GPU | GPUs per unreserved NRP node | Unreserved NRP nodes | AWS EC2 proxy (one GPU) | AWS on-demand $/instance-hour |
| --- | ---: | ---: | ---: | --- | ---: |
| RTX A4000 | 16 GB | 16 | 2 | g6.xlarge (L4 24 GB) | $0.805 |
| RTX A5000 | 24 GB | 1 (4 on a reserved node) | 2 | g5.xlarge (A10G 24 GB) | $1.006 |
| A10 | 24 GB | 7–8 | 33 | g5.xlarge (A10G 24 GB) | $1.006 |
| RTX 3090 | 24 GB | 1–8 | 33 | g5.xlarge (A10G 24 GB) | $1.006 |
| RTX 4090 | 24 GB | 4 or 8 | 3 | g6.xlarge (L4 24 GB) | $0.805 |
| L4 | 24 GB | 16 | 2 | g6.xlarge (L4 24 GB) | $0.805 |
| A40 | 48 GB | 2 (8 on an issue-tainted node) | 2 | g6e.xlarge (L40S 48 GB) | $1.861 |
| RTX A6000 | 48 GB | 1, 4, 6, or 8 | 5 | g6e.xlarge (L40S 48 GB) | $1.861 |
| L40 | 48 GB | 4 | 6 | g6e.xlarge (L40S 48 GB) | $1.861 |
| L40S | 48 GB | 4 | 2 | g6e.xlarge (L40S 48 GB) | $1.861 |

All listed NRP cards are Ampere or Ada and meet FlashAttention-2's hardware
architecture requirement. That does not validate end-to-end HexGen execution.
Our FlashAttention 2.0.8 image completed 7B inference on A6000/A10 and one
debug-mode 70B request on four A6000s. Other 70B GPU/stage combinations need
model-load, attention, and single-request probes.

## Comparable 70B FP16 configurations

FP16 Llama-2-70B weights occupy about 140 GB before KV cache and runtime
buffers. Weight pressure below is weight bytes divided by nominal aggregate
VRAM **per replica**. Every stage also needs its own headroom check. Both arms
of a pair should use the same checkpoint, prompts, decoding, traffic trace,
and measurement code. Costs multiply the one-GPU AWS instance proxy rates
above by NRP GPU counts.

| Pair | Centralized homogeneous on one NRP node | Decentralized heterogeneous across sites | AWS-equivalent cost | Weight pressure | Feasibility |
| --- | --- | --- | --- | --- | --- |
| **A. Small placement test** | 4 A6000, 1 replica | 2 A6000 + 2 A40, 1 replica | $7.444/h vs $7.444/h; **0% difference** | 73% vs 73% | Unreserved node types exist; free GPU/CPU capacity and A40 70B runtime unverified. |
| **B. Paper case-study analogue** | 6 A6000, 1 replica | 4 A6000 + 2 A5000 + 2 A4000, 1 replica | $11.166/h vs $11.066/h; **−0.9%** | 49% vs 51% | Paper's card mix. A two-A5000 local tensor-parallel stage requires reserved CSUSB access; unreserved A5000 nodes have one GPU each. |
| **C. Matched two-replica placement test** | 8 A6000, two 4-A6000 replicas | 2 replicas, each 2 A6000 + 2 A40 at separate sites | $14.888/h vs $14.888/h; **0% difference** | 73% per replica in both arms | Requires an unreserved 8-A6000 node, all four unreserved A40s, two-replica routing, and remote model state. Free capacity unverified. |
| **D. Asymmetric two-replica placement test** | 8 A6000, two 4-A6000 replicas | 2 replicas, each 3 A6000 + 2 A4000 at two sites | $14.888/h vs $14.386/h; **−3.4%** | 73% central; 80% decentralized per replica | NRP advertises required node sizes without reservation; stage fit, free capacity, and routing unverified. |

Pairs A and C control cost, aggregate VRAM, and replica count most closely.
Pair D additionally exercises asymmetric stage sizing. A first model-load
probe for D could try TP [2,1,2] and 46/22/12 transformer layers on the
A6000/A6000/A4000 groups; this is **not** a measured safe split. The current
one-replica manifests do not implement two-replica routing or remote model
state.

The earlier RunPod-priced 8-RTX-3090 baseline versus two replicas of
3-A6000 + 2-A4000 is **not** budget-matched under these AWS proxies:
$8.048/h versus $14.386/h (+78.8%). Do not use it as an equal-budget pair.
These AWS proxy prices also do not support a fair claim of more replicas for
the heterogeneous arm at the same budget; the matched pairs test placement
and asymmetric parallelism.

## Sources

- Live NRP Kubernetes node inventory, checked 2026-09-29;
  [public NRP resource view](https://nrp.ai/viz/resources/).
- [AWS official on-demand example rates, us-east-1, May 2026](https://aws.amazon.com/cn/blogs/china/inference-ai-agent-nvidia-nemoclaw-llm-router-amazon-ec2/).
- [AWS accelerated instance GPU specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).
- [HexGen paper](https://arxiv.org/html/2311.11514);
  [FlashAttention 2.0.8 upstream](https://github.com/Dao-AILab/flash-attention/tree/v2.0.8).
