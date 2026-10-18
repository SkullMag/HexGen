"""Export only latency values from the verified 13B NRP S3 runs."""

import hashlib
import json
import math
import os

import boto3
from botocore.config import Config


RUNS = {
    "homogeneous": ("13b-a10-full-001", "centralized-13b-a10-east"),
    "heterogeneous": ("13b-hetero-ada-smoke-001", "decentralized-13b-a10-rtx5000"),
}
RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        config=Config(s3={"addressing_style": "path"}),
    )

    def read(key):
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    output = {"source_bucket": bucket, "runs": {}}
    for label, (run_id, variant) in RUNS.items():
        prefix = f"hexgen-nrp/{run_id}/{variant}/"
        keys = [
            item["Key"]
            for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix
            )
            for item in page.get("Contents", [])
        ]
        candidates = []
        for key in keys:
            if key.endswith("/summary.json"):
                summary = json.loads(read(key))
                if (
                    [rate["rate_per_second"] for rate in summary.get("rates", [])]
                    == RATES
                    and all(rate["offered_requests"] == 100 for rate in summary["rates"])
                ):
                    candidates.append((key, summary))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one full-sweep summary for {label}; got {len(candidates)}")
        summary_key, summary = candidates[0]
        pod_prefix = summary_key.rsplit("/", 1)[0]
        result = {
            "run_id": run_id,
            "variant": variant,
            "client_pod": pod_prefix.rsplit("/", 1)[-1],
            "summary_sha256": hashlib.sha256(read(summary_key)).hexdigest(),
            "prompt_bank_sha256": summary["prompt_bank_sha256"],
            "rates": {},
        }
        for rate in summary["rates"]:
            key = f"{pod_prefix}/{rate['results_file']}"
            blob = read(key)
            records = [json.loads(line) for line in blob.splitlines()]
            if len(records) != 100 or sum(bool(rec["success"]) for rec in records) != 100:
                raise RuntimeError(f"Expected 100/100 successful requests in {key}")
            latencies = [float(rec["end_to_end_latency_s"]) for rec in records]
            if any(not math.isfinite(value) or value <= 0 for value in latencies):
                raise RuntimeError(f"Invalid latency in {key}")
            result["rates"][str(rate["rate_per_second"])] = {
                "latencies_s": latencies,
                "raw_sha256": hashlib.sha256(blob).hexdigest(),
            }
        output["runs"][label] = result
    if output["runs"]["homogeneous"]["prompt_bank_sha256"] != output["runs"][
        "heterogeneous"
    ]["prompt_bank_sha256"]:
        raise RuntimeError("Prompt banks differ")
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
