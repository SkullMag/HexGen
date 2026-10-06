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
