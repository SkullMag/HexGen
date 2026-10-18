"""Plot empirical SLO attainment for the paired 13B NRP runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
STYLES = {
    "homogeneous": ("Homogeneous · 2×A10, one node", "#16697A"),
    "heterogeneous": ("Heterogeneous · A10 + RTX 5000 Ada, two sites", "#C45134"),
}


def latency_samples(data):
    samples = {}
    for arm in STYLES:
        run = data["runs"][arm]
        samples[arm] = {}
        for rate in RATES:
            latencies = np.asarray(run["rates"][str(rate)]["latencies_s"], dtype=float)
            if len(latencies) != 100 or not np.all(np.isfinite(latencies)) or np.any(latencies < 0):
                raise ValueError(f"Invalid latency sample for {arm} at {rate} requests/s")
            samples[arm][rate] = np.sort(latencies)
    return samples


def main():
    data = json.loads((HERE / "13b_latency.json").read_text())
    samples = latency_samples(data)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    fig.suptitle("Llama-2-13B FP16 · SLO attainment on NRP", fontsize=17, weight="bold", y=0.98)
    for ax, rate in zip(axes.flat, RATES):
        max_latency = max(samples[arm][rate][-1] for arm in STYLES)
        for arm, (label, color) in STYLES.items():
            latencies = samples[arm][rate]
            # Attainment at deadline t is 100 * count(latency <= t) / 100.
            deadlines = np.r_[0.0, latencies, max_latency * 1.05]
            attainment = np.r_[0.0, np.arange(1, len(latencies) + 1), 100.0]
            ax.step(deadlines, attainment, where="post", color=color, linewidth=2.2, label=label)
        ax.axhline(99, color="#68757d", linestyle=":", linewidth=1)
        ax.set_title(f"Offered rate: {rate:g} requests/s")
        ax.set_xlim(0, max_latency * 1.05)
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.grid(True, color="#e3e8eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("SLO deadline (seconds, linear scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SLO attainment (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.937), ncol=2, frameon=False)
    fig.text(
        0.05,
        0.015,
        "Each point is 1 of 100 measured requests; all requests succeeded. 32 generated tokens; one sweep per arm. "
        "The dotted line marks 99% attainment.\n"
        "Deadlines are absolute seconds because the paper's A100 latency baseline was not measured here. "
        "The two 13B layouts differ in GPU memory, placement, and parallelism; this is not a cost-matched 70B replication.",
        fontsize=9,
        color="#37474f",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.905), h_pad=2.1, w_pad=2.0)
    fig.savefig(HERE / "13b_comparison.png", dpi=220, facecolor="white")
    fig.savefig(HERE / "13b_comparison.pdf", facecolor="white")
    print(HERE / "13b_comparison.png")
    print(HERE / "13b_comparison.pdf")


if __name__ == "__main__":
    main()
