# September 21, 2026 native RTX Red results

[Read the results report](../../RESULTS.md) ·
[Download the 22-file ZIP](red-results-no-prompts.zip?raw=true)

Seven rates, 500 requests each, 3,500 successful completions. One FP16
OpenLLaMA-7B-v2 replica spans two single-GPU Quadro RTX 6000 hosts with TP=2,
PP=1 and regular attention. These are archived measurements, not a fresh run of
the publication commit or an exact reproduction of the paper's 70B experiment.

Each `expNum_seqLen_reqRate_deploy` folder contains:

- `requests.csv`: `request_id`, `prompt_id`, `success`, `end_to_end_latency_s`.
  Latency runs from scheduled arrival to the complete response. All retained cells
  match the original export; `prompt_text` is omitted from this public copy.
- `metrics.json`: request counts, median/p95 latency, throughput including queue
  drain, peak per-GPU PyTorch allocation, and deadline attainment.
- `plot.png`: the original plot showing the full completion-time range.

`setup.json` records configuration, workload selection, frozen source hashes and
the baseline: 30 warmups followed by 100 isolated measured requests, with median
distributed inference time **1.1642241477966309 seconds**. A deadline is a
multiplier times this baseline; it is an analysis threshold, not a cancellation
timeout. `overview.png` uses a common 0–20 multiplier range across all rates.

`publication-audit.json` records the export checks and SHA-256 hashes. The ZIP
contains only `setup.json` plus the seven CSV/metrics/plot triplets. The report,
overview and publication audit are alongside it in the repository.

Prompt text, generated text, model weights, raw native logs, credentials and
deployment endpoints are excluded. Prompt IDs and the pinned LMSYS dataset
revision allow authorized users to trace the workload. Full independent native
rank validation requires the original locally retained evidence.

All requests eventually finishing at an offered rate does not demonstrate
sustainable throughput. There was one finite trial per rate, and high-load
latencies include substantial queueing and delayed response delivery.
