FABRIC PROVISIONING, RESOURCE INVENTORIES AND LATENCY ECDFS

From the repository root, create a separate controller environment (Python 3.13):
  python3.13 -m venv .venv-fabric
  .venv-fabric/bin/python -m pip install -r deploy/fabric/reproducibility/requirements.txt
  PY=.venv-fabric/bin/python
  TOOLS=deploy/fabric/reproducibility

1. Validate the example without authentication, network calls or allocations:
  "$PY" "$TOOLS/provision.py" provision --plan "$TOOLS/five-node-plan.json" --name hexgen-NEW-RUN --out runs/NEW-RUN --dry-run

2. After checking placement availability and authenticating, provision a new slice:
  "$PY" "$TOOLS/provision.py" provision --plan "$TOOLS/five-node-plan.json" --name hexgen-NEW-RUN --out runs/NEW-RUN --lease-hours 24 --fabric-rc "$HOME/work/fabric_config/fabric_rc"

Use your own fabric_rc path. The plan pins historical physical hosts and must be
updated using current placement availability. It requests five VMs, eight GPUs and
two single-site L2Bridge networks. Slice names and output directories must be new.
The slice ID is saved before waiting; failed provisioning retains the allocation and
diagnostic inventory. The script never automatically deletes or renews leases.
Resources stay reserved only until the recorded lease expires.

This provisions infrastructure, not the model or GPU drivers. Install the software
stack separately and capture another snapshot immediately before each measurement.
Missing drivers/tools are recorded as unavailable, not interpreted as GPU readiness.
Interface names can change: use the recorded IP/interface mapping. CPU/RAM/disk
units and instance flavors follow FABRIC; requested and observed fields are separate.

The example represents two full 13B FP16 replicas per arm:
  homogeneous: two STAR VMs, each with 2 Tesla T4 GPUs (TP=2).
  heterogeneous: one MAX VM with 2 Quadro RTX 6000 GPUs (TP=2), plus one
    replica pipelined over two GATECH VMs with 1 A30 each (PP=2, TP=1).
Tensor parallelism stays within each VM. RTX 6000 here is the Turing Quadro, not
RTX A6000. Provisioning does not start replicas or configure FlashAttention.

3. Snapshot each experiment's resources before running its workload:
  "$PY" "$TOOLS/provision.py" snapshot --slice-id SLICE_ID --out runs/NEW-RUN/128_0.25 --fabric-rc "$HOME/work/fabric_config/fabric_rc" --host-info

Repeat --slice-id for multiple slices. Existing inventories are not overwritten by
the CLI. Choose a new directory for a later snapshot. --no-ssh captures FABlib only;
--host-info attempts host CPU/NUMA information and records unavailable API access.
Save benchmark results alongside resources.txt and resources.json. To integrate in
Python, put this tools directory on sys.path and use:
  from resources import capture, write_inventory
  write_inventory(run_dir, capture([slice_object], host_info=True))

The inventory records time, script hash, FABlib/Python versions, slice/project/lease,
physical host/site, VM hostname/image/allocations, requested capacity, component
models, PCI/NUMA labels, networks/interfaces and available guest probes. These include
OS/kernel, CPU/NUMA, memory, disks, NICs/routes, NVIDIA UUID/VRAM/driver/GPU details,
GPU topology, CUDA compiler, runtime package versions and source commit. Probes have
bounded timeouts. No token, key contents or environment dumps are collected. Review
inventories before publication: detailed host/network metadata is intended for your
experiment records. Runtime/source probes default to ~/experiment/env and
~/experiment/source/hexgen; adapt GUEST_SCRIPT for a different guest installation.

4. Rebuild raw latency ECDFs for one experiment or a batch:
  "$PY" "$TOOLS/plot_ecdf.py" "$TOOLS/results/2026-10-13"

Each experiment needs homogeneous.json and heterogeneous.json, with a requests list
of success and latency_seconds fields (a bare request list also works). manifest.json
optionally supplies output_tokens and rate. Latency is completion minus scheduled
Poisson arrival, in seconds, including dispatch, transport, queueing and inference.
No reference-time/SLA normalization is used. Every offered request stays in the
denominator; failures are placed at infinity, so a curve with failures ends below 1.
Ties form a single jump. Curves are not smoothed or truncated.

Each experiment receives latency-ecdf.png/.svg/.csv. The batch receives an overview,
counts, versions and input/script hashes. Inputs are unchanged. Individual x-axis
limits use each setting's full range; all y-axes use the same probability scale.

Included measured results (100 requests per arm, 1,200 total; all successful):
  32 output tokens at 1 request/s
  64 output tokens at 0.5 requests/s
  128 output tokens at 0.125, 0.25, 0.5 and 1 request/s
These are the saved Oct 13 LMSYS/Poisson pilot results, using OpenLLaMA 13B, FP16,
FlashAttention and the two-replica placements above. The arms ran sequentially.
The public JSON files retain only success and latency; prompts, outputs, controller
paths and management addresses are omitted. resources.txt/.json label historical
evidence, requested capacity, source hashes and unrecorded fields. They are not fresh
hardware probes or proof of active reservations. Historical GPU UUIDs and complete
CPU/FIM allocation details were not recorded and cannot be recovered retrospectively.
The exporter binds published numeric result hashes to the original saved result hashes.

These are a scaled pilot, not an equal-budget reproduction of the paper's 70B setup.
No equal rental budget has been established for the GPU mixes. End-to-end latency
also includes the native asynchronous callback/transport path; buffered callbacks
can affect the distribution under load. The ECDF alone does not isolate GPU speed.

To export another copy from the original historical result/inventory schema:
  "$PY" "$TOOLS/export_results.py" /path/to/original/results /path/to/NEW-export
This utility only reads local records, uses an explicit field allowlist and refuses
to overwrite an export. It deliberately accepts only historical_recorded_evidence.

5. Run offline checks (no FABRIC login or allocations required):
  "$PY" -m unittest discover -s "$TOOLS" -p 'test_*.py' -v

The FABRIC reproducibility GitHub workflow runs these tests, validates the plan with
python -O, and rebuilds all six plots on PRs. Controller dependencies are pinned;
guest probes use the standard library. Keep local environments and runs untracked.

FABlib API reference: https://fabric-fablib.readthedocs.io/en/latest/
