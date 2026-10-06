# 13B fleet placement check, October 6, 2026

At about 16:37-16:41 UTC, the `nautilus` cluster's `nyu-networks` namespace
had no existing Jobs or Pods. `hexgen-model-west-13b` was Bound as a 100 GiB
RWX CephFS claim. The listed quota objects restricted A100/H100/H200/GH200,
not the A10 or RTX 3090 resources used here.

The 13B fleet's source Job template and both placement-gate manifests were
changed to omit `priorityClassName: opportunistic`. With no class specified,
NRP admitted the probes at priority 0. The same omission applies to the
rendered 13B fleet server and client Jobs. No full model run was started.

Each probe requested and limited two `nvidia.com/gpu` devices, three CPUs, and
42 GiB memory **on one node**. The selectors required either an NVIDIA A10
in `us-east` or an NVIDIA GeForce RTX 3090 in `us-west`; the GPU taint was
tolerated. The original gate submitted two East A10 pair Jobs plus one West
RTX 3090 pair Job simultaneously. The alternate gate submitted one East A10
pair plus two West RTX 3090 pair Jobs. Every Job had a 120-second active
deadline and was deleted promptly after its scheduling result was read.

Before using default priority, a non-preempting attempt with explicit
`preemptionPolicy: Never` was rejected by NRP's Pod admission controller,
so no Pods were created. A subsequent `owner-no-preempt` test admitted Pods,
but all three Pods in each gate stayed Pending. They were removed before the
default-priority gates.

Both default-priority gates also left all three Pods Pending and unassigned.
For the A10 requests, the scheduler reported insufficient GPU on 12 nodes,
CPU on three, and memory on three; the other nodes failed the selectors,
taints, or schedulability checks. For the RTX 3090 requests, it reported
insufficient GPU on 23 nodes, CPU on 18-19, and memory on 16, with the other
nodes failing other placement checks. The scheduler's priority-0 preemption
assessment said preemption was not helpful or found no suitable victims.
These counts are scheduler filter counts over all 533 nodes, **not** counts
of candidate nodes with a particular GPU type or free GPUs. None of the
probes received a hostname, so this check did not validate a specific host.

No inference requests were sent, no 13B full fleet arm ran, and no S3 result
was produced. A final namespace check found no Jobs or Pods. These results
show that neither selected comparable fleet layout was schedulable during
this short priority-0 check. They do not establish that NRP has too few
installed GPUs or that capacity will remain unavailable.

## Additional region check

At about 16:49 UTC, the live node inventory also listed A10 nodes in
`us-central` and RTX 3090 nodes in `us-central`, `us-east`, `us-mountain`, and
`us-west`. Installed/allocatable devices are not currently unallocated
devices. The original East/West selection came from earlier validated A10
and RTX 3090 smokes and East access to the staged West CephFS model claim;
the model claim has not yet been mount-tested on a Central GPU host.

To test whether another region offered a fleet placement, short temporary
manifests derived from the two versioned fleet probe files replaced both
region selectors with `us-central`, leaving GPU product, 2-GPU/3-CPU/42-GiB
requests, priority 0, and the 120-second deadline unchanged. One simultaneous
gate asked for two Central A10 pairs and one Central RTX 3090 pair; the other
asked for one Central A10 pair and two Central RTX 3090 pairs. Every Pod stayed
Pending without an assigned hostname. The scheduler reported insufficient
GPU on eligible nodes, with CPU and/or memory failures on some nodes; its
preemption assessment did not find a feasible placement. All six temporary
Jobs and Pods were deleted, and a final namespace check found no Jobs or Pods.

This rules out those two **tested Central-region gates at that time**, not
every possible GPU type, site, or future placement. A Central admission would
still require a model-volume mount test and normal-mode inference before a
full sweep.

## Longer scheduler observation

At about 16:54-16:56 UTC, the original East-A10/West-RTX-3090 simultaneous
gate was submitted again at default priority. Unlike the quick snapshots
above, the three Jobs were left for their full 120-second active deadline.
Each Pod initially and after roughly 30 and 60 seconds remained Pending,
reported `Unschedulable`, and had no assigned node. All three Jobs reached
their deadline as Failed with zero completions; none ran the probe command.
They were then deleted. A final check found no Jobs or Pods in the namespace.
Thus waiting for this bounded interval did not produce the required placement.
`Pending` itself is retried by Kubernetes and could resolve later if capacity
changes; this check gives no guarantee about a future hour.
