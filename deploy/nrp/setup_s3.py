#!/usr/bin/env python3
"""Create and verify the NRP results bucket, then install its Kubernetes Secret."""

import argparse
import base64
import getpass
import hashlib
import json
import subprocess

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", help="Existing or new results bucket name")
    parser.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    parser.add_argument("--context", default="nautilus")
    parser.add_argument("--namespace", default="nyu-networks")
    args = parser.parse_args()

    access_key = getpass.getpass("NRP S3 access key ID: ").strip()
    secret_key = getpass.getpass("NRP S3 secret access key: ").strip()
    if not access_key or not secret_key:
        parser.error("both NRP S3 keys are required")

    bucket = args.bucket or (
        "hexgen-nrp-results-" + hashlib.sha256(access_key.encode()).hexdigest()[:12]
    )
    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={"addressing_style": "path"}),
    )
    try:
        try:
            client.head_bucket(Bucket=bucket)
            print(f"Bucket exists: {bucket}")
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status != 404:
                raise
            client.create_bucket(Bucket=bucket)
            print(f"Bucket created: {bucket}")

        test_key = "hexgen-nrp/setup/verification.txt"
        payload = b"NRP HexGen S3 write/read verification\n"
        client.put_object(Bucket=bucket, Key=test_key, Body=payload)
        observed = client.get_object(Bucket=bucket, Key=test_key)["Body"].read()
        if observed != payload:
            raise RuntimeError("S3 readback did not match the uploaded test object")
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"NRP S3 setup failed: {type(exc).__name__}") from None
    print(f"S3 write/read verified: s3://{bucket}/{test_key}")

    values = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "S3_BUCKET": bucket,
        "S3_ENDPOINT_URL": args.endpoint,
    }
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "hexgen-nrp-s3", "namespace": args.namespace},
        "type": "Opaque",
        "data": {
            key: base64.b64encode(value.encode()).decode()
            for key, value in values.items()
        },
    }
    command = [
        "kubectl", "--context", args.context, "--namespace", args.namespace,
        "apply", "--server-side", "--field-manager=hexgen-nrp-s3-setup", "-f", "-",
    ]
    result = subprocess.run(
        command, input=json.dumps(manifest), text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise SystemExit(
            f"Kubernetes Secret setup failed (exit {result.returncode}); "
            "the verified bucket remains available. Check cluster access and retry."
        )
    print(f"Kubernetes Secret ready: {args.namespace}/hexgen-nrp-s3")


if __name__ == "__main__":
    main()
