# NRP Llama-2-13B comparison plot

[`13b_comparison.png`](13b_comparison.png) and
[`13b_comparison.pdf`](13b_comparison.pdf) overlay the completed homogeneous
and heterogeneous runs in six panels, one for each offered request rate. As in
[HexGen's Figure 2](https://arxiv.org/pdf/2311.11514v3), the vertical axis is
SLO attainment: the percentage of requests completed within a deadline. The
horizontal axis is the SLO deadline in **absolute seconds**. Both axes are
linear. Each empirical curve rises by one percentage point for each of the
100 successful measured requests at that rate; the dotted horizontal line
marks 99% attainment.

The paper expresses its deadline as a scale factor of a reference A100
execution latency. We did not measure that reference on NRP, so these panels
use seconds instead of claiming paper-equivalent SLO scale factors. This also
avoids selecting one arbitrary deadline after seeing the data. The SLO metric
and linear presentation follow Figure 2, but the horizontal units differ.

The figure was calculated from the 1,200 successful raw request records in
NRP S3, not from hand-copied summary values. `export_13b_latency.py` reads the
full-sweep Pod subdirectory for each arm, checks six rates and 100/100
successes per rate, verifies the prompt-bank hashes match, and exports only
latency values and object hashes to `13b_latency.json`. It never exports
prompts, responses, model weights, or credentials. To regenerate the figure:

```sh
python3 deploy/nrp/analysis/plot_13b_comparison.py
```

The source runs are:

- Homogeneous: `s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/13b-a10-full-001/centralized-13b-a10-east/`; two A10s on one East node, 48 GiB total VRAM, TP 2.
- Heterogeneous: `s3://hexgen-nrp-results-b97886564c50/hexgen-nrp/13b-hetero-ada-smoke-001/decentralized-13b-a10-rtx5000/`; A10 in East and RTX 5000 Ada in West, 56 GiB total VRAM, PP 2 with 16/24 layers.

Both arms used the same Llama-2-13B FP16 checkpoint, image, prompt bank,
client site, arrival seed, 32 generated tokens, and 100 measured requests per
rate. At 0.125 requests/s, the minimum deadline for 99% attainment was 5.119
seconds for the homogeneous arm and 22.036 seconds for the heterogeneous arm.
The homogeneous curve reaches any given attainment level sooner at all six
rates.

This is one sweep per arm, with no run-to-run uncertainty estimate. The GPU
types, VRAM totals, parallel layout, and network topology differ; the runs
are not cost matched. The paper evaluated Llama-2-70B with many independently
placed replicas at a matched cloud budget and optimized its layout to avoid
cross-region communication within a pipeline. These 13B results demonstrate
that this particular two-site pipeline works and is slower than the local
two-A10 setup. They do not validate or refute the paper's claimed performance
advantage for its optimized fleet layout.
