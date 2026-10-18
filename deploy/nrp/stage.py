#!/usr/bin/env python3
"""Render or submit the remote homogeneous model-staging Job."""

import argparse
from pathlib import Path
import re
import subprocess

import yaml


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "apply", "delete"])
    parser.add_argument("--image", required=True, help="Pinned GHCR SHA tag or digest")
    parser.add_argument("--revision", default="3aba440b59558f995867ba6e1f58f21d0336b5bb")
    parser.add_argument("--run-id", default="llama70b")
    parser.add_argument("--context", default="nautilus")
    parser.add_argument("--namespace", default="nyu-networks")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a 40-character commit SHA")
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]{0,30}[a-z0-9])?", args.run_id):
        parser.error("run-id must be a short DNS-safe lowercase label")
    document = yaml.safe_load((HERE / "model-stage-central.yaml").read_text())
    document["metadata"]["name"] += f"-{args.run_id}"
    container = document["spec"]["template"]["spec"]["containers"][0]
    container["image"] = args.image
    next(item for item in container["env"] if item["name"] == "MODEL_REVISION")["value"] = args.revision
    data = yaml.safe_dump(document, sort_keys=False)
    if args.action == "render":
        print(data, end="")
        return
    command = ["kubectl", f"--context={args.context}", "-n", args.namespace]
    if args.action == "apply":
        for kind, name in (("pvc", "hexgen-model-unl"), ("secret", "hexgen-hf-token")):
            subprocess.run([*command, "get", kind, name, "-o", "name"], check=True)
        command.extend(["apply", "-f", "-"])
    else:
        command.extend(["delete", "-f", "-", "--ignore-not-found"])
    subprocess.run(command, input=data, text=True, check=True)


if __name__ == "__main__":
    main()
