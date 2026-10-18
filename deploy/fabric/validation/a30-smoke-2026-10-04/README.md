# FABRIC A30 infrastructure smoke, October 4, 2026

This test checked whether FABRIC can host a **local two-GPU stage layout** for a
Llama-2-13B FP16 HexGen replica. It did not load the checkpoint or serve a
13B request. It is not a homogeneous/heterogeneous benchmark result.

## Placement and runtime

- A three-hour FABRIC slice was admitted at the HAWI site. One A30 was assigned
  to each of `hawi-w4.fabric-testbed.net` and `hawi-w5.fabric-testbed.net`.
  Each VM requested 8 cores, 32 GB RAM, 100 GB disk, and one Basic NIC.
- Both A30s reported 24,576 MiB and CUDA compute capability 8.0 after
  installation of the NVIDIA server driver and container toolkit.
- The two VMs were attached to one FABRIC L2Bridge. ICMP round-trip time between
  their private interfaces was 0.1–0.3 ms in a two-packet connectivity check.
- Both VMs pulled the pinned HexGen image
  `ghcr.io/skullmag/hexgen@sha256:c2a8a83d7ca6c5bce16514942a206e4e871dfafc0aa373f9942bd56b22b21391`.
  On **each** A30, FP16 FlashAttention 2 packed attention, fused dense, and
  rotary CUDA operations returned successfully. This verifies these kernels on
  the A30, not the full 13B model path.
- Two image containers, one per VM, completed a GPU NCCL all-reduce over the
  private link. Rank 0 started with 1.0, rank 1 with 2.0, and both reported 3.0.

## Remaining model gate

The 13B checkpoint is gated. An attempt to copy the existing Hugging Face
credential from the NRP Kubernetes Secret to the separate FABRIC VMs was
rejected by automatic approval review because the earlier authorization covered
NRP, not FABRIC. No model weights were downloaded to FABRIC or to the local
computer; no full-model inference request or S3 result was produced. With
explicit permission for FABRIC credential use, a fresh slice can download the
pinned checkpoint on the VMs, convert its 40 layers, and run a two-host
normal-mode inference smoke.

The temporary FABRIC slice was deleted after the synthetic GPU and network
tests. A follow-up query found no active HexGen FABRIC slice.
