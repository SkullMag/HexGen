# Deliverable 2: comparable GPU budgets for NRP

Checked 2026-09-29. This is a **planning estimate**, not an NRP bill or a
reservation. NRP node counts below come from the live `nautilus` Kubernetes node
inventory (`Ready` nodes, GPU-specific `nvidia.com/*` capacity). A listed node
may have all of its GPUs or CPU occupied. Nodes with reservation, issue, or
testing `NoSchedule` taints are excluded from the unreserved counts. A
`nautilus.io/hardware` taint on an otherwise accessible large-GPU node can be
tolerated by an appropriately sized Job.

## Pricing method

Use **RunPod Secure Cloud on-demand USD per GPU-hour** for *every* arm total.
It provides prices for the actual NRP A6000, A5000, A4000, A40, and 3090 card
types. The rates are not NRP
charges and do not include storage, network traffic, CPU/RAM differences,
replica routing, or any multi-GPU-node premium. They establish a consistent
cloud-equivalent GPU budget, not a prediction of a bill on another provider.
RunPod's tabulated rates were verified 2026-08-10; its A4000 product page was
updated 2026-08-27. Recheck before reporting a later experiment. GPU counts
are the number on one unreserved NRP physical node, followed by the number of
such nodes seen in the inventory. They do **not** indicate free GPUs.

| NRP GPU | Nominal VRAM per GPU | GPUs per unreserved NRP node | Unreserved NRP nodes | RunPod Secure Cloud $/GPU-hour |
| --- | ---: | ---: | ---: | ---: |
| RTX A4000 | 16 GB | 16 | 2 | $0.25 |
| RTX A5000 | 24 GB | 1 (also 4 on a reserved node) | 2 | $0.27 |
| A10 | 24 GB | 7–8 | 33 | n/a (not listed by RunPod) |
| RTX 3090 | 24 GB | 1–8 | 33 | $0.50 |
| RTX 4090 | 24 GB | 4 or 8 | 3 | $0.74 |
| L4 | 24 GB | 16 | 2 | $0.49 |
| A40 | 48 GB | 2 (also 8 on an issue-tainted node) | 2 | $0.49 |
| RTX A6000 | 48 GB | 1, 4, 6, or 8 | 5 | $0.53 |
| L40 | 48 GB | 4 | 6 | $0.82 |
| L40S | 48 GB | 4 | 2 | $1.09 |

All listed GPU architectures are Ampere or Ada, so they satisfy the upstream
FlashAttention-2 architecture requirement. That is **hardware compatibility**,
not an end-to-end HexGen validation: our pinned FlashAttention 2.0.8 image has
completed 7B work on A6000/A10 and one debug-mode 70B request on four A6000s.
Each other card type needs a one-request model-load and attention check before a
benchmark. Do not use Turing cards on this image's FlashAttention path.

## Matched 70B FP16 comparisons

Llama-2-70b-hf FP16 weights occupy about 140 GB before KV cache and runtime
buffers. Ratios below are *weight bytes / nominal aggregate VRAM*; all stages
must be checked individually for headroom. The two arms in a pair run
sequentially with the same prompts, decoding, traffic trace, and measurement
code. `delta` is `(heterogeneous cost / homogeneous cost) - 1`.

| Pair and purpose | Centralized homogeneous: one NRP node | Decentralized heterogeneous: sites/nodes | Cloud-equivalent GPU cost | VRAM and weight pressure | Status |
| --- | --- | --- | --- | --- | --- |
| **A. Small placement test** | 4× A6000 (1 replica) | 2× A6000 at one site + 2× A40 at another (1 replica) | $2.12/h vs $2.04/h; **−3.8%** | 192 GB vs 192 GB; 73% vs 73% | A6000 and A40 node sizes exist without reservation; GPU/CPU free capacity and A40 runtime still unverified. |
| **B. Paper case-study analogue** | 6× A6000 (1 replica, e.g. local pipeline stages) | 4× A6000 + 2× A5000 + 2× A4000 across sites (1 replica) | $3.18/h vs $3.16/h; **−0.6%** | 288 GB vs 272 GB; 49% vs 51% | Paper's exact card mix, but the only unreserved A5000 nodes have one GPU each. A two-A5000 local tensor-parallel stage needs reserved CSUSB access; otherwise use separate one-GPU pipeline stages and revalidate. |
| **C. Main replica-density test** | 8× RTX 3090 on one node (1 replica) | **2 replicas**, each 3× A6000 on one node/site + 2× A4000 on another | $4.00/h vs $4.18/h; **+4.5%** | 192 GB for central replica (73%); 176 GB **per** heterogeneous replica (80%) | NRP advertises 8×3090 nodes, a 6×A6000 node, and 16×A4000 nodes without reservation. Placement, stage layer splits, CPU/memory, PVCs, two-replica routing, and 70B runtime remain unverified. |
| **D. Closest published-rate match for two replicas** | 8× RTX 3090 on one node (1 replica) | **2 replicas**, each 2× A40 + 2× A5000 + 2× A4000 | $4.00/h vs $4.04/h; **+1.0%** | 192 GB central (73%); 176 GB **per** heterogeneous replica (80%) | Requires four A5000s; only two are on unreserved nodes. Currently contingent on reservation access or a future inventory change. |

Pair C is the best *unreserved-inventory candidate* for testing the paper's
central finding: equal-ish dollars can support more independently routable
70B replicas with an asymmetric, geographically split pool. The original paper
used 16× A100-40G at $65.54/h for four replicas, compared with a $65.04/h
heterogeneous pool that could hold up to 12 replicas. Its 8×3090Ti `[4,4]`
pipeline is a useful starting point for the central side; NRP's **3090 is not
3090Ti**, so speed is not directly comparable. For Pair C, a first model-load
probe could try stages with TP `[2,1,2]` and 46/22/12 transformer layers on
the A6000/A6000/A4000 groups, respectively. This gives the 16 GB A4000s fewer
layers; it is **not** a measured safe split. Aggregate VRAM alone does not
prove the model fits. Two replicas also require a request router and separate model
state at both sites, which the current one-replica manifests do not provide.
Do not publish a performance result for Pair C until those pieces work.

The existing manifest pair, 4×A6000 versus 3×A6000 + 2×A10, matches 192 GB
nominal VRAM but **has no same-provider RunPod price for A10**. It remains a
memory-matched pilot, not a dollar-matched pair under this method.

## Sources

- Live NRP Kubernetes `nautilus` node inventory, checked 2026-09-29 18:45 UTC;
  [NRP resource view](https://nrp.ai/viz/resources/) for a public live check.
- [RunPod Secure Cloud GPU rate table](https://www.runpod.io/articles/guides/ai-server-cost)
  and [A4000 Secure Cloud rate](https://www.runpod.io/gpu-models/rtx-a4000).
- [HexGen paper, Section 5 and Appendix F](https://arxiv.org/html/2311.11514);
  [FlashAttention 2.0.8 upstream](https://github.com/Dao-AILab/flash-attention/tree/v2.0.8).
