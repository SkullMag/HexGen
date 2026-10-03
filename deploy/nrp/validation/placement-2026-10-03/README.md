# 70B NRP placement check, 2026-10-03

This is a live scheduler check, not a 70B benchmark. It used the `nautilus`
context and `nyu-networks` namespace. All temporary GPU Jobs were deleted by
19:23 UTC; no GPU reservation or waiting Job remains.

## Inputs and constraints

- Llama-2-70B FP16 needs roughly 140 GB for weights, plus runtime and KV-cache
  headroom. A 192 GB GPU pool gives about 73% nominal weight pressure.
- The existing 350 GiB `hexgen-model-unl` claim was still `Bound`. The S3
  results Secret was present. The saved Hugging Face token passed access checks
  for the pinned model and prompt dataset and was installed as an NRP Secret;
  no token value was printed or committed.
- The old manifests pin `hcc-prp-c5038.unl.edu`, which was absent from the live
  Kubernetes node list. There is no prepared model claim for a second site.
- The live node list had 533 entries and included candidate A6000, A10,
  RTX 3090, A40, L40, L40S, and A4000 nodes. Allocatable node capacity is not
  the same as currently unallocated capacity. This account cannot list pods
  across all namespaces, so the scheduler was used as the placement check.

## Short GPU scheduling probes

Each probe was a `Job` with `opportunistic` priority, a 300-second active
deadline, the pinned HexGen image, and a command to run `nvidia-smi` and sleep
briefly if scheduled. GPU requests equaled limits. None acquired a node.

| Placement tested | Requests | Scheduler result |
| --- | --- | --- |
| Delaware `gpu00.nrp.hpc.udel.edu` | 3 RTX A6000, 6 CPU, 144 GiB | Insufficient A6000, CPU, and memory |
| Massachusetts `gpu-06.nrp.mghpcc.org` | 2 A10, 6 CPU, 84 GiB | Insufficient GPU and memory |
| Any eligible RTX A6000 node | 4 A6000, 6 CPU, 144 GiB | No schedulable node; insufficient A6000, CPU, or memory |
| Any eligible A10 node | 8 A10, 6 CPU, 144 GiB | No schedulable node; insufficient GPU, CPU, or memory |
| Any eligible RTX 3090 node | 8 RTX 3090, 6 CPU, 144 GiB | No schedulable node; insufficient GPU, CPU, or memory |
| Any eligible L40 node | 4 L40, 6 CPU, 144 GiB | No schedulable node; insufficient GPU or CPU |
| Any eligible L40S node | 4 L40S, 6 CPU, 144 GiB | No schedulable node; insufficient GPU or CPU |
| US West, two stages simultaneously | 2 A6000 and 2 A40; each 6 CPU, 84 GiB | Neither stage scheduled; insufficient GPU, CPU, or memory |
| Any eligible RTX A4000 node | 12 A4000, 6 CPU, 144 GiB | No schedulable node; insufficient GPU, CPU, or memory |

These results describe only the tested requests at the time of the checks.
They do not imply that the cluster lacks enough installed GPUs or that a
different partition cannot run 70B. The A6000/A40 two-stage layout would
have matched the four-A6000 centralized layout at 192 GB per replica, but it
was not schedulable. The L40/L40S options would also require an image rebuilt
for their Ada architecture before inference, even if capacity opened.

## Next run gate

1. Find a simultaneously schedulable homogeneous and heterogeneous pair.
   Keep each complete 70B replica within one region when comparing to the
   paper's placement strategy. Confirm GPU model, CPU, RAM, node health,
   taints, and storage locality with short Jobs.
2. Stage the pinned checkpoint, converted weights, and prompt bank on an NRP
   volume accessible to every worker at the selected site. Do not download
   them to the submitting machine. Update the manifest's host selectors,
   GPU resource names, and claim names for that actual placement.
3. Run a normal-mode model load and repeated-request smoke test, then the
   paired traffic sweep. Preserve raw per-request and per-rank results in S3.
   A single distributed replica does not test the paper's fleet-level
   throughput advantage from fitting more independent replicas.

NRP guidance: [GPU Jobs](https://nrp.ai/documentation/userdocs/running/gpu-pods/),
[geographical scheduling](https://nrp.ai/documentation/userdocs/tutorial/scheduling/#using-geographical-topology),
[regional storage](https://nrp.ai/documentation/userdocs/storage/ceph/).
