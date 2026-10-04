"""Plot the paired 13B NRP runs from sanitized S3 latency records."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
STYLES = {
    "homogeneous": ("Homogeneous", "#16697A"),
    "heterogeneous": ("Heterogeneous", "#C45134"),
}


def metrics(data):
    result = {}
    for arm, run in data["runs"].items():
        values = []
        for rate in RATES:
            latencies = np.asarray(run["rates"][str(rate)]["latencies_s"], dtype=float)
            if len(latencies) != 100 or not np.all(np.isfinite(latencies)):
                raise ValueError(f"Invalid latency sample for {arm} at {rate} requests/s")
            values.append(
                (
                    float(np.quantile(latencies, 0.5)),
                    float(np.quantile(latencies, 0.95)),
                    float(np.sort(latencies)[math.ceil(0.99 * len(latencies)) - 1]),
                )
            )
        result[arm] = np.asarray(values)
    return result


def main():
    data = json.loads((HERE / "13b_latency.json").read_text())
    values = metrics(data)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), sharex=True)
    fig.suptitle("NRP Llama-2-13B FP16: homogeneous vs heterogeneous", fontsize=17, weight="bold", y=0.97)
    axes[0].set_title("End-to-end latency")
    axes[1].set_title("Minimum deadline for 99% SLO attainment")
    for arm, (label, color) in STYLES.items():
        series = values[arm]
        axes[0].plot(RATES, series[:, 0], color=color, marker="o", linewidth=2.5, label=f"{label} · p50")
        axes[0].plot(RATES, series[:, 1], color=color, marker="s", linestyle="--", linewidth=2, label=f"{label} · p95")
        axes[1].plot(RATES, series[:, 2], color=color, marker="o", linewidth=2.5, label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(RATES, [format(rate, "g") for rate in RATES])
        ax.set_ylim(0.8, 800)
        ax.grid(True, which="major", color="#dce3e7", linewidth=0.8)
        ax.grid(True, which="minor", axis="y", color="#edf1f3", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Offered requests per second")
        ax.set_ylabel("Seconds (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].legend(loc="upper left", frameon=False)
    fig.text(
        0.055,
        0.035,
        "100/100 requests succeeded at each rate; 32 generated tokens; one sweep per arm. "
        "The 99% deadline is the 99th-fastest of 100 requests.\n"
        "Heterogeneous: 56 GiB and a cross-region pipeline; homogeneous: 48 GiB on one node. "
        "These runs are not cost-matched or equivalent to the paper's 70B multi-replica layout.",
        fontsize=9,
        color="#37474f",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.92), w_pad=2.5)
    fig.savefig(HERE / "13b_comparison.png", dpi=220, facecolor="white")
    fig.savefig(HERE / "13b_comparison.pdf", facecolor="white")
    print(HERE / "13b_comparison.png")
    print(HERE / "13b_comparison.pdf")


if __name__ == "__main__":
    main()
