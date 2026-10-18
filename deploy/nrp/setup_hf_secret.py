#!/usr/bin/env python3
"""Check gated model/dataset access and install an NRP-only HF token Secret."""

import argparse
import base64
import getpass
import json
from pathlib import Path
import subprocess

import requests


MODEL_REVISION = "3aba440b59558f995867ba6e1f58f21d0336b5bb"
DATASET_REVISION = "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8"
RESOURCES = {
    "Llama 2 70B": (
        "https://huggingface.co/meta-llama/Llama-2-70b-hf/resolve/"
        f"{MODEL_REVISION}/config.json"
    ),
    "LMSYS prompt dataset": (
        "https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/"
        f"{DATASET_REVISION}/data/train-00000-of-00001-cced8514c7ed782a.parquet"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-saved-token", action="store_true",
                        help="Read ~/.cache/huggingface/token without displaying it")
    parser.add_argument("--context", default="nautilus")
    parser.add_argument("--namespace", default="nyu-networks")
    args = parser.parse_args()
    if args.use_saved_token:
        token_path = Path.home() / ".cache/huggingface/token"
        if not token_path.is_file():
            parser.error("No saved Hugging Face token found")
        token = token_path.read_text().strip()
    else:
        token = getpass.getpass("Hugging Face read token: ").strip()
    if not token:
        parser.error("A Hugging Face token is required")

    for label, url in RESOURCES.items():
        try:
            response = requests.head(
                url, headers={"Authorization": f"Bearer {token}"},
                allow_redirects=False, timeout=30,
            )
        except requests.RequestException as exc:
            raise SystemExit(f"Could not check {label}: {type(exc).__name__}") from None
        if response.status_code not in (200, 302, 303, 307, 308):
            raise SystemExit(
                f"Access to {label} was not confirmed (HTTP {response.status_code}). "
                "Obtain access in Hugging Face before staging the model."
            )
        print(f"Access confirmed: {label}")

    manifest = {
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {"name": "hexgen-hf-token", "namespace": args.namespace},
        "type": "Opaque",
        "data": {"HF_TOKEN": base64.b64encode(token.encode()).decode()},
    }
    result = subprocess.run(
        ["kubectl", f"--context={args.context}", "-n", args.namespace,
         "apply", "--server-side", "--field-manager=hexgen-hf-setup", "-f", "-"],
        input=json.dumps(manifest), text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise SystemExit(f"Could not install NRP HF Secret (exit {result.returncode})")
    print(f"NRP Secret ready: {args.namespace}/hexgen-hf-token")


if __name__ == "__main__":
    main()
