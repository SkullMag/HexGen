#!/usr/bin/env python3
"""Audit a paired NRP sweep and calculate SLO attainment from raw request logs."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def records(folder, filename):
    rows = [json.loads(line) for line in (folder / filename).read_text().splitlines()]
    indexed = {row["request_index"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate request index in {folder / filename}")
    return indexed


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compare(central, hetero, slo):
    configs = [read(folder / "config.json") for folder in (central, hetero)]
    keys = ("run_id", "model_name", "image_ref", "model_revision",
            "tokenizer_model_sha256", "prompt_bank_sha256",
            "rates_per_second", "requests_per_rate", "new_tokens",
            "arrival_seed", "unit_rate_offsets_s")
    for key in keys:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Comparison mismatch: {key}")
    if not all(configs[0][key] for key in ("image_ref", "model_revision", "tokenizer_model_sha256")):
        raise ValueError("Missing image or model provenance")
    if configs[0]["variant"] != "centralized" or configs[1]["variant"] != "decentralized":
        raise ValueError("Expected centralized then decentralized")
    report = {"run_id": configs[0]["run_id"], "slo_seconds": slo,
              "image_ref": configs[0]["image_ref"],
              "prompt_bank_sha256": configs[0]["prompt_bank_sha256"],
              "rates": []}
    for rate in configs[0]["rates_per_second"]:
        label = format(rate, "g").replace(".", "p")
        filename = f"rate-{label}.jsonl"
        arms = [records(folder, filename) for folder in (central, hetero)]
        expected = set(range(configs[0]["requests_per_rate"]))
        if any(set(arm) != expected for arm in arms):
            raise ValueError(f"Incomplete request set for rate {rate}")
        for index in expected:
            a, b = (arm[index] for arm in arms)
            if (a["request_id"] != b["request_id"] or
                    a["prompt_id"] != b["prompt_id"] or
                    abs(a["scheduled_offset_s"] - b["scheduled_offset_s"]) > 1e-6):
                raise ValueError(f"Unpaired request at rate {rate}, index {index}")
        result = {"rate_per_second": rate, "offered_requests": len(expected)}
        for name, arm in zip(("centralized", "decentralized"), arms):
            successful = [row for row in arm.values() if row["success"]]
            latencies = [row["end_to_end_latency_s"] for row in successful]
            result[name] = {
                "successes": len(successful),
                "failures": len(expected) - len(successful),
                "slo_attainment": sum(value <= slo for value in latencies) / len(expected),
                "p50_success_latency_s": percentile(latencies, 0.5),
                "p95_success_latency_s": percentile(latencies, 0.95),
            }
        both = [(arms[0][i], arms[1][i]) for i in expected
                if arms[0][i]["success"] and arms[1][i]["success"]]
        result["paired_output_matches"] = sum(a["output_text"] == b["output_text"]
                                               for a, b in both)
        result["paired_successes"] = len(both)
        report["rates"].append(result)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--hetero", type=Path, required=True)
    parser.add_argument("--slo-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.slo_seconds <= 0:
        parser.error("slo-seconds must be positive")
    report = compare(args.central, args.hetero, args.slo_seconds)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
