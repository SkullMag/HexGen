Historical measurements from September 21, 2026. Raw prompt data, model weights and result archives are retained separately and are not distributed in this repository. Packaging changes to paths and test fixtures have not been rerun on GPUs.

# Native homogeneous RTX result report

All seven rates completed: 3500/3500 requests. Fresh TP=2 base inference reference: 1.164224 s.

| RPS | Requests | Median latency (s) | p95 latency (s) |
|---:|---:|---:|---:|
| 0.125 | 500 | 1.196 | 1.963 |
| 0.25 | 500 | 1.189 | 2.236 |
| 0.5 | 500 | 1.265 | 5.352 |
| 1 | 500 | 36.990 | 60.097 |
| 2 | 500 | 325.259 | 451.534 |
| 4 | 500 | 284.888 | 492.121 |
| 8 | 500 | 454.122 | 509.138 |

The native OCF coordinator with a bounded 1024-stream/512-MiB per-peer HTTP resource override and error-handling corrections, NATS worker coordination, HexGen model partitioning/decoder and original request client were reused. One OpenLLaMA-7B-v2 FP16 replica spans two Quadro RTX 6000 hosts with TP=2 and PP=1 over private Ethernet.

This is a scaled regular-attention starting-point experiment, not a reproduction of the paper's 70B Red numbers. The cross-host topology, GPU type, model, controlled LMSYS subset and reference latency differ. FlashAttention 2.0.8 supplies package components, but its attention kernels are disabled on these Turing GPUs. One finite trial per rate provides no repeat-trial confidence interval; overload latency includes queue drain.

See README.md for exact server configuration, input preparation, timing boundaries, source reuse and all twelve patched runtime source files and the coordinator validation tests. Compact ZIP: native-hexgen-stream1024-red-full.zip (22 verified files). Raw measurements, both-rank logs, provenance and audit receipts remain in this directory.

The measurements support the weekly starting-point deliverable: a live seven-rate Red suite plus an isolated inference reference using the native distributed stack on older RTX 6000 GPUs. At 0.125–0.5 RPS, median end-to-end latency is 1.19–1.27 seconds; at 1 RPS it reaches 36.99 seconds, and at 2–8 RPS delays reach minutes. All 500 requests finishing at a high offered rate does not demonstrate that rate can be sustained indefinitely. The stream-capacity increase removes the earlier admission limit; it does not increase GPU throughput.

At the overloaded rates, client responses arrive in bursts and native rank logs continue advancing between those bursts. The recorded end-to-end metric includes those delivery delays as well as queueing and inference. The 4-RPS median being lower than the 2-RPS median is not a demonstrated improvement: these are single finite runs with different completion timing, and the 4-RPS p95 is worse. No steady-state capacity estimate or causal breakdown of transport delays is claimed.

The overview uses the same deadline range of 0–20 times the 1.164224-second base reference (up to 23.284483 seconds) for every panel. Thus, high-rate curves remain near zero in that view even though all requests eventually succeed. Each experiment’s individual plot shows its full completion-time range. Success here means a valid completed inference response; it is not a measure of linguistic answer quality.

Cleanup verified on both hosts at 16:27 UTC: tracked experiment workers and coordinators exited, temporary private-peer firewall rules removed, no GPU compute processes remain. Leases and server instances were preserved. The monitoring automation was deleted after the authorized suite and cleanup completed.
