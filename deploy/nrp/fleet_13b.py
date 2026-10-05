#!/usr/bin/env python3
"""Render two independent, locally parallelized 13B replicas per NRP arm."""

import argparse
from copy import deepcopy
import json
import re
import subprocess

import yaml

from run import render as render_single


VARIANTS = {
    "homogeneous": "centralized-13b-fleet-a10",
    "homogeneous-west": "centralized-13b-fleet-3090",
    "heterogeneous": "decentralized-13b-fleet-a10-3090",
}
ARM_JOB_PREFIX = {"homogeneous": "homo", "homogeneous-west": "west", "heterogeneous": "hete"}


def set_env(container, name, value):
    for entry in container.get("env", []):
        if entry["name"] == name:
            entry["value"] = value
            return
    container.setdefault("env", []).append({"name": name, "value": value})


def render(arm, run_id, image, hosts, client_node, arrival_seed=20260919,
           requests=100, rates="0.125 0.25 0.5 1 2 4", warmups=3, new_tokens=32,
           component="all"):
    if len(hosts) != 2:
        raise ValueError("Exactly two replica hosts are required")
    if arm != "homogeneous-west" and not hosts[0].endswith(".nrp.mghpcc.org"):
        raise ValueError("Replica 0 must be an explicitly selected MGH A10 host")
    if arm == "homogeneous" and not hosts[1].endswith(".nrp.mghpcc.org"):
        raise ValueError("Both homogeneous replicas must stay at MGH")
    variant = VARIANTS[arm]
    source = render_single("centralized-13b-a10-east", run_id, image,
                           arrival_seed, requests, rates, warmups, new_tokens,
                           "all", client_node)
    common = next(doc for doc in source if doc["kind"] == "ConfigMap"
                  and doc["metadata"]["name"] == "hexgen-nrp-ocf")
    code = next(doc for doc in source if doc["kind"] == "ConfigMap"
                and doc["metadata"]["name"].startswith("hexgen-nrp-client-code-"))
    service = next(doc for doc in source if doc["kind"] == "Service")
    server = next(doc for doc in source if doc["kind"] == "Job"
                  and "-head-" in doc["metadata"]["name"])
    client = next(doc for doc in source if doc["kind"] == "Job"
                  and "-client-" in doc["metadata"]["name"])
    documents = [common]
    heads = []
    for index, host in enumerate(hosts):
        name = f"hexgen-13b-{ARM_JOB_PREFIX[arm]}-r{index}-{run_id}"
        if len(name) > 63:
            raise ValueError("run-id is too long for Kubernetes resource names")
        heads.append(f"http://{name}:8092")
        svc = deepcopy(service)
        svc["metadata"]["name"] = name
        svc["spec"]["selector"]["app"] = name
        job = deepcopy(server)
        job["metadata"]["name"] = name
        pod = job["spec"]["template"]
        pod["metadata"]["labels"]["app"] = name
        spec = pod["spec"]
        spec["nodeSelector"]["kubernetes.io/hostname"] = host
        if arm == "homogeneous-west" or (arm == "heterogeneous" and index == 1):
            spec["nodeSelector"].update({
                "topology.kubernetes.io/region": "us-west",
                "nvidia.com/gpu.product": "NVIDIA-GeForce-RTX-3090",
            })
        spec["volumes"].append({"name": "nccl-shm", "emptyDir": {
            "medium": "Memory", "sizeLimit": "2Gi"}})
        for container in spec["containers"]:
            for key, value in {"VARIANT": variant,
                               "HEAD_NODE": heads[-1],
                               "OCF_WORKER_ADDR": name,
                               "MASTER_ADDR": name}.items():
                if container["name"] == "worker":
                    set_env(container, key, value)
            if container["name"] == "worker":
                set_env(container, "NCCL_P2P_DISABLE", "1")
                container["volumeMounts"].append({"name": "nccl-shm", "mountPath": "/dev/shm"})
        documents.extend([svc, job])
    client["metadata"]["name"] = f"hexgen-13b-{ARM_JOB_PREFIX[arm]}-client-{run_id}"
    for container in client["spec"]["template"]["spec"]["containers"]:
        set_env(container, "VARIANT", variant)
        set_env(container, "HEAD_NODE", heads[0])
        set_env(container, "HEAD_NODES", " ".join(heads))
    documents.extend([code, client])
    if component == "servers":
        return [doc for doc in documents if doc is not code and doc is not client]
    if component == "client":
        return [code, client]
    return documents


def kubectl(args, data=None):
    return subprocess.run(["kubectl", "--context=nautilus", "-n", "nyu-networks", *args],
                          input=data, text=True, check=True)


def preflight_servers(arm, hosts):
    jobs = subprocess.run(["kubectl", "--context=nautilus", "-n", "nyu-networks",
                           "get", "jobs", "-o", "json"], capture_output=True,
                          text=True, check=True)
    active = [job["metadata"]["name"] for job in json.loads(jobs.stdout)["items"]
              if job["metadata"]["name"].startswith("hexgen-")
              and job.get("status", {}).get("active", 0)]
    if active:
        raise RuntimeError(f"Existing HexGen Jobs are active: {active}")
    pods = subprocess.run(["kubectl", "--context=nautilus", "-n", "nyu-networks",
                           "get", "pods", "-o", "json"], capture_output=True,
                          text=True, check=True)
    live_pods = [pod["metadata"]["name"] for pod in json.loads(pods.stdout)["items"]
                 if pod["metadata"]["name"].startswith("hexgen-")
                 and pod["status"].get("phase") in ("Pending", "Running")]
    if live_pods:
        raise RuntimeError(f"Existing HexGen Pods are active: {live_pods}")
    nodes = subprocess.run(["kubectl", "--context=nautilus", "-n", "nyu-networks",
                            "get", "nodes", "-o", "json"], capture_output=True,
                           text=True, check=True)
    by_name = {node["metadata"]["name"]: node
               for node in json.loads(nodes.stdout)["items"]}
    for index, host in enumerate(hosts):
        node = by_name.get(host)
        if node is None:
            raise RuntimeError(f"Host missing from current node inventory: {host}")
        spec, labels = node["spec"], node["metadata"]["labels"]
        ready = any(cond["type"] == "Ready" and cond["status"] == "True"
                    for cond in node["status"]["conditions"])
        west = arm == "homogeneous-west" or (arm == "heterogeneous" and index == 1)
        wanted_region = "us-west" if west else "us-east"
        wanted_gpu = ("NVIDIA-GeForce-RTX-3090" if west
                      else "NVIDIA-A10")
        if not ready or spec.get("unschedulable") or (
            labels.get("topology.kubernetes.io/region") != wanted_region
            or labels.get("nvidia.com/gpu.product") != wanted_gpu
        ):
            raise RuntimeError(f"Host is not a Ready {wanted_gpu} in {wanted_region}: {host}")
        blocked = [taint["key"] for taint in spec.get("taints", [])
                   if taint.get("effect") in ("NoSchedule", "NoExecute")]
        if blocked:
            raise RuntimeError(f"Host has blocking taints: {host}: {blocked}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "apply", "delete"))
    parser.add_argument("--arm", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--hosts", nargs=2, required=True, metavar=("EAST_A10", "SECOND"))
    parser.add_argument("--client-node", required=True)
    parser.add_argument("--component", choices=("all", "servers", "client"), default="all")
    parser.add_argument("--arrival-seed", type=int, default=20260919)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--rates", default="0.125 0.25 0.5 1 2 4")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]{0,25}[a-z0-9])?", args.run_id):
        parser.error("run-id must be a short DNS-safe lowercase label")
    if args.requests < 1 or args.warmups < 0 or args.new_tokens < 1:
        parser.error("invalid request count, warmup count, or output length")
    if args.action == "apply" and args.component == "all":
        parser.error("apply servers first, verify both replicas, then apply client")
    documents = render(args.arm, args.run_id, args.image, args.hosts,
                       args.client_node, args.arrival_seed, args.requests,
                       args.rates, args.warmups, args.new_tokens, args.component)
    if args.action == "render":
        print(yaml.safe_dump_all(documents, sort_keys=False), end="")
        return
    if args.action == "apply":
        kubectl(["get", "secret", "hexgen-nrp-s3", "-o", "name"])
        kubectl(["get", "pvc", "hexgen-model-west-13b", "-o", "name"])
        if args.component == "servers":
            preflight_servers(args.arm, args.hosts)
        if args.component == "client":
            for index in range(2):
                name = f"hexgen-13b-{ARM_JOB_PREFIX[args.arm]}-r{index}-{args.run_id}"
                kubectl(["wait", "--for=condition=Ready", "pod", "-l",
                         f"job-name={name}", "--timeout=1s"])
        for document in documents:
            kubectl(["apply", "-f", "-"], yaml.safe_dump(document, sort_keys=False))
    else:
        for document in reversed(documents):
            if document["kind"] == "ConfigMap" and document["metadata"]["name"] == "hexgen-nrp-ocf":
                continue  # Shared coordinator configuration may serve another run.
            kubectl(["delete", "-f", "-", "--ignore-not-found"],
                    yaml.safe_dump(document, sort_keys=False))


if __name__ == "__main__":
    main()
