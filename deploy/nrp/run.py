#!/usr/bin/env python3
"""Render and apply one NRP comparison arm without storing credentials in Git."""

import argparse
import math
from pathlib import Path
import re
import subprocess

import yaml

HERE = Path(__file__).resolve().parent


def kubectl(args: list[str], context: str, namespace: str, data: str | None = None):
    return subprocess.run(["kubectl", f"--context={context}", "-n", namespace, *args],
                          input=data, text=True, check=True)


def render(variant: str, run_id: str, image: str, arrival_seed: int = 20260919,
           requests: int | None = None, rates: str | None = None,
           warmups: int | None = None, new_tokens: int | None = None) -> list[dict]:
    documents = list(yaml.safe_load_all((HERE / "common.yaml").read_text()))
    documents += list(yaml.safe_load_all((HERE / f"{variant}.yaml").read_text()))
    for document in documents:
        if document["kind"] == "Job":
            document["metadata"]["name"] += f"-{run_id}"
        containers = document.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        for container in containers:
            container["image"] = image
            if container.get("env") is not None:
                container["env"] = [env for env in container["env"] if env["name"] != "IMAGE_REF"]
                container["env"].append({"name": "IMAGE_REF", "value": image})
            for env in container.get("env", []):
                if env["name"] == "RUN_ID":
                    env["value"] = run_id
                if env["name"] == "ARRIVAL_SEED":
                    env["value"] = str(arrival_seed)
                overrides = {"REQUESTS": requests, "RATES": rates,
                             "WARMUPS": warmups, "NEW_TOKENS": new_tokens}
                if env["name"] in overrides and overrides[env["name"]] is not None:
                    env["value"] = str(overrides[env["name"]])
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["render", "apply", "delete"])
    parser.add_argument("--variant", choices=["centralized", "decentralized"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True, help="Immutable GHCR digest or unique SHA tag")
    parser.add_argument("--arrival-seed", type=int, default=20260919,
                        help="Use the same seed for both arms of one paired run")
    parser.add_argument("--requests", type=int, help="Requests per offered rate")
    parser.add_argument("--rates", help="Space-separated offered rates per second")
    parser.add_argument("--warmups", type=int, help="Warmup requests after readiness")
    parser.add_argument("--new-tokens", type=int, help="Generated tokens per request")
    parser.add_argument("--context", default="nautilus")
    parser.add_argument("--namespace", default="nyu-networks")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]{0,30}[a-z0-9])?", args.run_id):
        parser.error("run-id must be a short DNS-safe lowercase label")
    if args.requests is not None and args.requests < 1:
        parser.error("requests must be positive")
    if args.warmups is not None and args.warmups < 0:
        parser.error("warmups cannot be negative")
    if args.new_tokens is not None and args.new_tokens < 1:
        parser.error("new-tokens must be positive")
    if args.rates is not None:
        try:
            if not args.rates.split() or any(
                not math.isfinite(float(rate)) or float(rate) <= 0
                for rate in args.rates.split()
            ):
                raise ValueError
        except ValueError:
            parser.error("rates must be a space-separated list of positive numbers")
    documents = render(args.variant, args.run_id, args.image, args.arrival_seed,
                       args.requests, args.rates, args.warmups, args.new_tokens)
    data = yaml.safe_dump_all(documents, sort_keys=False)
    if args.action == "render":
        print(data, end="")
        return
    if args.action == "apply":
        required_pvcs = {v["persistentVolumeClaim"]["claimName"]
                         for doc in documents if doc["kind"] == "Job"
                         for v in doc["spec"]["template"]["spec"].get("volumes", [])
                         if "persistentVolumeClaim" in v}
        for kind, name in [("secret", "hexgen-nrp-s3"),
                           *(("pvc", name) for name in sorted(required_pvcs))]:
            kubectl(["get", kind, name, "-o", "name"], args.context, args.namespace)
        # Submit the coordinator and GPU workers before the client. The client also
        # retries the native request path while model initialization completes.
        for doc in documents:
            kubectl(["apply", "-f", "-"], args.context, args.namespace,
                    yaml.safe_dump(doc, sort_keys=False))
        print(f"Submitted {args.variant} arm; raw results go under "
              f"hexgen-nrp/{args.run_id}/{args.variant}/")
    else:
        kubectl(["delete", "-f", "-", "--ignore-not-found"],
                args.context, args.namespace, data)


if __name__ == "__main__":
    main()
