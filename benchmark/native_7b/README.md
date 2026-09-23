# Native homogeneous 7B experiment

This folder preserves the experiment used for the completed September 21, 2026
RTX 6000 sweep. It reuses HexGen's OCF/libp2p/NATS serving path, rank-based
`LlamaWorker`, model partitioning, collectives, KV cache and decoder. The added
Python code prepares inputs, replays arrivals through the original request client,
and audits/exports measurements. It is not a replacement inference server.

The measured run used upstream commit
`949414d1be18e504752508c0559cb966ddf999f4` with the runtime changes in this PR.
This publication makes paths/endpoints configurable, extracts the previously used
reference and plot helpers, and adjusts one test fixture for laptop RAM limits.
The published packaging has CPU/transport validation; the 3,500 GPU measurements
belong to the archived run, not a new GPU run of this PR commit.

## Fixed experimental configuration

| Item | Recorded setting |
|---|---|
| Model | `openlm-research/open_llama_7b_v2`, revision `e5961def23172a2384543940e773ab676033c963` |
| Precision / decoding | FP16, greedy, no quantization, exactly 32 new tokens |
| Hardware | Two separate hosts, one Quadro RTX 6000 (Turing, 24 GiB) per host |
| Parallelism | One replica; TP=2, PP=1, batch size 1; 32 layers |
| Network | Private Ethernet, NCCL sockets, RDMA disabled; eno1 reported 10 Gb/s |
| Workload | 500 LMSYS first-user prompts, exactly 128 input tokens including BOS |
| Load | Poisson arrivals, rates 0.125, 0.25, 0.5, 1, 2, 4, 8 requests/s |
| Attention | Regular attention; `--use-flash-attn` omitted |
| OCF resource override | Native HTTP protocol per peer: 1024 streams / 512 MiB |

The installed `flash-attn==2.0.8` supplies model/helper components, but its
attention kernels were **not** enabled on Turing. GPU runtime: Python 3.11,
PyTorch 2.0.1+cu118, Transformers 4.36.2, NVIDIA driver 560.35.05, Ubuntu 22.04.5;
NATS Python client 2.16.0 and aiohttp 3.14.3. OCF uses Go 1.20.14 (the pinned QUIC
dependency does not support arbitrarily newer Go versions).

This is a scaled starting-point experiment, not the paper's 70B result. Cross-host
TP, GPU count/type, regular attention, checkpoint and fixed-length subset differ.
The audit is deliberately specific to this two-RTX deployment. An A100 comparison
requires its own explicit hardware configuration and fresh baseline; do not relabel
these results or silently relax the RTX audit.

## Runtime changes

- `hexgen/llama/arguments.py`: configurable checkpoint, converted weights and rank log paths.
- `llama_config_utils.py`: preserve checkpoint KV-head count and rotary base.
- `load_model_parameters_utils/create_separate_state_dicts_llama_7b.py`: parameterize
  the existing weight converter and preserve rotary frequencies.
- `llama_inference.py`: load these paths, enable evaluation mode, count prompt
  **tokens** rather than characters for output length, and record synchronized
  inference timing, input/output IDs and peak allocated memory.
- `benchmark/hexgen_documents/_llama_worker.py`: log request/prompt IDs with native rank evidence.
- `benchmark/utils/_base_rank_based.py`: separate the worker coordinator address
  (`OCF_WORKER_ADDR`) from the torch rendezvous (`MASTER_ADDR`).
- `benchmark/send_request/request.py`: JSON parsing instead of `eval`; preserve
  HTTP failures instead of returning a success-shaped result.
- `benchmark/hexgen_documents/scripts/run_cross_node.sh`: two-host TP=2/PP=1
  regular-attention launcher with required deployment variables.
- OCF `internal/protocol/p2p/host.go`: bounded resource-manager configuration for
  native HTTP; zero streams keeps the original defaults. **Other parent/system/
  protocol limits remain active**, so 1024 is a ceiling, not a guarantee that every
  host can admit 1024 requests. No GPU-throughput increase is implied.
- OCF `internal/server/forward.go`, `request.go`, `internal/common/requests/proxy.go`:
  close response bodies, preserve failed statuses and emit exactly one JSON error.

The preceding native run failed at 2 RPS (314 successful / 186 failed requests).
The final run used the bounded override and corrected error handling, recalibrated
from scratch, and did not combine predecessor measurements with the new results.

## Prepare the workload (CPU only)

Keep all generated assets outside the tracked source tree. `HEXGEN_RUN_DIR` is an
absolute directory for a **new** experiment. The scripts refuse to overwrite the
frozen prompt bank, reference, or controller run. Do not run with Python `-O`:
the scientific audit assertions are intentional.

```bash
export HEXGEN_RUN_DIR=/absolute/path/to/new-experiment
python -m pip install -r benchmark/native_7b/requirements-cpu.txt
```

Accept the official [LMSYS dataset access terms](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
and authenticate locally with Hugging Face. Cache the pinned dataset Parquet and
model tokenizer/checkpoint before running these offline scripts. Dataset and
checkpoint revisions are in `workload_settings.json`; no token or dataset content
is committed here. Preparing the prompt bank itself does not load model weights.

```bash
python benchmark/native_7b/prepare_native_workload.py
```

Selection uses `conversation_a`'s initial user turn, outer-whitespace trimming,
all languages, deduplication and at least 127 text tokens. From 33,000 rows there
are 2,004 eligible prompts; `random.Random(20260919)` selects 500. BOS plus the first
127 text tokens gives 128 inputs. Four truncations cannot round-trip through the
native text API; the original policy replaces only those four, in place, with the
first unused compatible records in source-row order. The resulting SHA-256 is
`a7e1963c0ef88643b9b8013707c33ecf7a932a70bdecfa40064bc5cb2de1e755`.

`selection.py` and `prepare_native_workload.py` retain the original algorithms.
`upstream_workload.py` is copied byte-for-byte from the authors' historical
[`experimental/optimizer/llmsim/workload.py`](https://github.com/Relaxed-System-Lab/HexGen/blob/e86ff85b4d254bfed9e840e63e397b224a53530a/experimental/optimizer/llmsim/workload.py).
It retains the upstream repository license. NumPy seed 20260919 generates a
unit-rate Poisson trace; each offered rate divides the same first 500 offsets by
that rate. The historical synchronous callback runner is not used: asynchronous
calls preserve offered arrivals while prior requests are pending.

## Configure native services and workers

Follow the original [OCF service documentation](../../third_party/ocf/README.md)
and [rank-worker documentation](../hexgen_documents/README.md). Build one Linux
OCF binary with Go 1.20.14 and copy that identical binary to both hosts at
`third_party/ocf/src/ocf-core/build/core`. Use a standalone head coordinator on
host 1 and a worker coordinator on host 2 that bootstraps to the head's private
libp2p address. Each host uses HTTP 8092, NATS 8094, and private libp2p TCP 43905.
For both OCF configuration files add:

```yaml
p2p:
  http_streams_per_peer: 1024
  http_memory_mib_per_peer: 512
```

Save each actual configuration as `$HEXGEN_RUN_DIR/ocf-config.yaml`. Allow only the
private peer traffic required by OCF/NCCL/Gloo, according to your testbed's network
rules. Record your actual temporary rules in `firewall.json` for provenance;
publication does not automatically change firewalls, leases or server instances.

Convert pinned weights with the original utility (its `--help` shows the path
arguments). Set `CHECKPOINT_PATH` and `STATE_DICTS_PATH` to the resulting local
directories on both hosts. Before loading native workers, prepare the independent
three-prompt reference on a free GPU:

```bash
export CHECKPOINT_PATH=/absolute/path/to/pinned/checkpoint
python benchmark/native_7b/prepare_reference.py
python benchmark/native_7b/freeze_source.py
```

The reference is the same eager Transformers FP16 greedy loop used in validation;
it has **no timing role**. `freeze_source.py` records the reviewed git commit,
runtime-file hashes, driver hash, compiled OCF hash and reference hash. Copy
`approved-source.json`, `reference.json` and `workload/` unchanged to host 2.
Use the same source checkout and binary on both hosts.

Set these per-host variables, then use the original launcher:

```bash
export MASTER_ADDR=<host-1-private-ip>
export OCF_WORKER_ADDR=<host-2-private-ip>
export HEAD_NODE=http://<host-1-private-ip>:8092
export NODE_RANK=0  # 1 on host 2
export STATE_DICTS_PATH=/absolute/path/to/converted/weights
export REQUEST_LOG="$HEXGEN_RUN_DIR/worker-rank${NODE_RANK}.jsonl"
# Change both if the private network interface is not eno1.
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
cd benchmark/hexgen_documents
bash scripts/run_cross_node.sh
```

The worker registration is `OpenLLaMA-7B-v2-native-rtx`; native client requests add
`_0`. The controller defaults to head `http://127.0.0.1:8092`; use
`HEXGEN_HEAD_URL` if the client is elsewhere. This is one provider replica, not
multi-replica load balancing or heterogeneous placement optimization.

## Validate, calibrate, replay and audit

Run `capture_provenance.py` on **both** hosts (same `HEXGEN_RUN_DIR` and checkout).
Run `validate_native.py` on host 1. Collect host 2's `provenance.json` and
`worker-rank1.jsonl` into host 1's `$HEXGEN_RUN_DIR/node02/`, preserving the names.
Then run `audit_validation.py`. It must pass tokenization for all 500 prompts and
match both ranks' 32 generated token IDs against the three independent references.
Only then start:

```bash
python benchmark/native_7b/run_native_benchmark.py
```

Sequence: 30 isolated warmups, 100 isolated measured requests, a 50-request pilot
at 0.125 RPS, then seven 500-request rates in ascending order. The controller writes
`status.json`, counts successful completions, drains each rate, and waits at an
`awaiting_result_audit` gate. Do not restart it or reuse the directory.

At each gate, refresh the rank evidence/provenance from both hosts and run:

```bash
python benchmark/native_7b/audit_results.py baseline
# At a rate gate use its directory, for example:
python benchmark/native_7b/audit_results.py 01_32_0.125_red
```

This writes an audit result and a candidate `.release.json` under `checkpoints/`.
For pilot/full rates, review `compact/<directory>/plot.png` and its four-file ZIP
before copying the corresponding passing receipt to `releases/<directory>.json`
on the controller host. The baseline is audit-only. Never fabricate a passing
receipt or overwrite a failed result to continue. After the final gate is released
and status is `complete`, `finalize.py` re-audits all rates, creates the overview,
and verifies a full ZIP with exactly 22 files (shared setup plus seven triplets).

Each compact experiment has only `requests.csv`, `metrics.json`, `plot.png`.
Folders use `expNum_seqLen_reqRate_deploy`, with experiment numbers `01` through
`07` in ascending request-rate order. The 50-request pilot is separate and does
not consume a full-suite number. `seqLen=32` means generated output tokens.
CSV columns: `request_id,prompt_id,prompt_text,success,end_to_end_latency_s`.
`prompt_text` is exactly the submitted truncated text. Keep raw client and both
rank logs outside Git for independent verification.

## Metrics and interpretation

- Base: median synchronized rank-0 inference time from 100 isolated calls after
  30 warmups. Includes prefill, 32-token decode and initial rank synchronization;
  excludes tokenization, HTTP and checkpoint loading. This is a TP=2 reference.
- Deadline: multiplier times that fresh base; attainment is on-time successes /
  all offered requests. It is post-run analysis, not a server cancellation timeout.
- End-to-end: scheduled arrival through full response, including submission lag,
  native waiting, transport and inference. Not TTFT or per-token latency.
- Throughput: completed requests / elapsed time through final drain; not a
  demonstrated steady-state capacity. p95 uses NumPy's percentile interpolation.
- Memory: maximum per-request PyTorch allocation on either GPU, including its
  model shard; not total process/device VRAM.

The archived base was 1.1642241477966309 seconds. All 3,500 full-suite requests
succeeded, but high offered rates accumulated minutes of delay. See [RESULTS.md](RESULTS.md).
The [published Red artifacts](results/2026-09-21-rtx6000-red/) include all seven
per-rate measurement folders, the overview, provenance and a 22-file ZIP.
Public CSVs omit prompt text; the original local experiment export retains it.
The published folders are renumbered in run order; their historical request IDs
remain unchanged for traceability. The frozen original evidence keeps its original
names. The current controller/finalizer use sequential numbering for fresh runs.
One finite trial per rate supplies no repeat-trial confidence intervals. Do not
infer that 4 RPS is faster than 2 RPS from the median alone, or assign a precise
queue/transport delay breakdown that was not instrumented. After collection, stop
only the experiment processes and remove only its recorded temporary rules.

## Validation for this PR

```bash
python -m unittest discover -s benchmark/native_7b/tests -v
cd third_party/ocf/src/ocf-core
# Use Go 1.20.14.
go test ./internal/protocol/p2p ./internal/server ./internal/common/requests -count=1
```

The transport test exercises 500 simultaneous native HTTP-over-libp2p requests.
The 1024/1025 boundary test raises only its **test fixture's parent peer limit**
above the boundary, since that unrelated default scales with host RAM. Production
parent limits are unchanged. On small CI hosts even the 500-request integration
test can encounter other native resource ceilings; use adequate host resources.

Workload preparation was rerun offline and exactly reproduced both the frozen
prompt-bank and arrival hashes. Portable audits were checked against a separate
copy of the archived evidence. No GPU benchmark was launched for publication.
