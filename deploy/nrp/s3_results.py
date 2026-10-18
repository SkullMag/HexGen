"""Upload raw measurements from a pod to NRP's S3-compatible Ceph storage."""

import json
import os
from pathlib import Path

import boto3
from botocore.config import Config


def destination(role: str, filename: str) -> tuple[str, str]:
    bucket = os.environ["S3_BUCKET"]
    run = os.environ["RUN_ID"]
    variant = os.environ["VARIANT"]
    pod = os.environ.get("POD_NAME", role)
    return bucket, f"hexgen-nrp/{run}/{variant}/{pod}/{filename}"


def upload(path: Path, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    endpoint = os.environ["S3_ENDPOINT_URL"]
    bucket, key = destination(role, path.name)
    client = boto3.client("s3", endpoint_url=endpoint,
                          config=Config(s3={"addressing_style": "path"}))
    client.upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"


def upload_metadata(directory: Path, role: str, extra: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    path.write_text(json.dumps({
        "run_id": os.environ["RUN_ID"],
        "variant": os.environ["VARIANT"],
        "pod": os.environ.get("POD_NAME"),
        "node": os.environ.get("NODE_NAME"),
        "role": role,
        **extra,
    }, indent=2) + "\n")
    return upload(path, role)
