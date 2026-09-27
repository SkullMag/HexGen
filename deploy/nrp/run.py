#!/usr/bin/env python3
"""Render and apply one NRP comparison arm without storing credentials in Git."""

import argparse
from pathlib import Path
import re
import subprocess

import yaml

HERE = Path(__file__).resolve().parent


def kubectl(args: list[str], context: str, namespace: str, data: str | None = None):
    return subprocess.run(["kubectl", f"--context={context}", "-n", namespace, *args],
                          input=data, text=True, check=True)


def render(variant: str, run_id: str, image: str) -> list[dict]:
    documents = list(yaml.safe_load_all((HERE / "common.yaml").read_text()))
    documents += list(yaml.safe_load_all((HERE / f"{variant}.yaml").read_text()))
    for document in documents:
        if document["kind"] == "Job":
            document["metadata"]["name"] += f"-{run_id}"
        containers = document.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        if document["kind"] == "Deployment":
            containers = document["spec"]["template"]["spec"]["containers"]
        for container in containers:
            container["image"] = image
            for env in container.get("env", []):
                if env["name"] == "RUN_ID":
                    env["value"] = run_id
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["render", "apply", "delete"])
    parser.add_argument("--variant", choices=["centralized", "decentralized"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True, help="Immutable GHCR digest or unique SHA tag")
    parser.add_argument("--context", default="nautilus")
    parser.add_argument("--namespace", default="nyu-networks")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]{0,30}[a-z0-9])?", args.run_id):
        parser.error("run-id must be a short DNS-safe lowercase label")
    documents = render(args.variant, args.run_id, args.image)
    data = yaml.safe_dump_all(documents, sort_keys=False)
    if args.action == "render":
        print(data, end="")
        return
    if args.action == "apply":
        required_pvcs = {v["persistentVolumeClaim"]["claimName"]
                         for doc in documents if doc["kind"] == "Deployment"
                         for v in doc["spec"]["template"]["spec"]["volumes"]
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
