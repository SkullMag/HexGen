"""Offline safety checks for the two-replica NRP manifests."""

import unittest

from fleet_13b import render


IMAGE = "ghcr.io/skullmag/hexgen@sha256:" + "a" * 64
EAST = "gpu-16.nrp.mghpcc.org"
WEST = "ry-gpu-02.sdsc.optiputer.net"
CLIENT = "usra-sti-01.uah.edu"


class FleetManifestTest(unittest.TestCase):
    def check_arm(self, arm, hosts, second_product, second_region):
        docs = render(arm, "fleet13btest", IMAGE, hosts, CLIENT)
        services = [doc for doc in docs if doc["kind"] == "Service"]
        jobs = [doc for doc in docs if doc["kind"] == "Job"]
        self.assertEqual((len(services), len(jobs)), (2, 3))
        self.assertEqual(len({doc["metadata"]["name"] for doc in services}), 2)
        workers = [doc for doc in jobs if "-client-" not in doc["metadata"]["name"]]
        for index, job in enumerate(workers):
            pod = job["spec"]["template"]
            selector = pod["spec"]["nodeSelector"]
            self.assertEqual(selector["kubernetes.io/hostname"], hosts[index])
            self.assertEqual(selector["nvidia.com/gpu.product"],
                             "NVIDIA-A10" if index == 0 else second_product)
            self.assertEqual(selector["topology.kubernetes.io/region"],
                             "us-east" if index == 0 else second_region)
            self.assertEqual(pod["metadata"]["labels"]["app"],
                             services[index]["spec"]["selector"]["app"])
            worker = next(c for c in pod["spec"]["containers"] if c["name"] == "worker")
            env = {entry["name"]: entry.get("value") for entry in worker["env"]}
            self.assertEqual((env["WORLD_SIZE"], env["HETERO_CONFIG"], env["PP_PARTITION"]),
                             ("2", "2", "40"))
            self.assertEqual(env["HEAD_NODE"], f"http://{services[index]['metadata']['name']}:8092")
        client = next(doc for doc in jobs if "-client-" in doc["metadata"]["name"])
        env = {entry["name"]: entry.get("value")
               for entry in client["spec"]["template"]["spec"]["containers"][0]["env"]}
        self.assertEqual(env["HEAD_NODES"].split(),
                         [f"http://{service['metadata']['name']}:8092" for service in services])

    def test_homogeneous_replicas_stay_east(self):
        self.check_arm("homogeneous", [EAST, "gpu-18.nrp.mghpcc.org"],
                       "NVIDIA-A10", "us-east")

    def test_heterogeneous_replicas_do_not_span_regions(self):
        self.check_arm("heterogeneous", [EAST, WEST],
                       "NVIDIA-GeForce-RTX-3090", "us-west")


if __name__ == "__main__":
    unittest.main()
