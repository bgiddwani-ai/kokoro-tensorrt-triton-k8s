import unittest
from pathlib import Path

import kubernetes_validate
import yaml


class KubernetesTests(unittest.TestCase):
    def test_single_manifest_matches_reference_configuration(self):
        root = Path(__file__).resolve().parents[2]
        documents = list(yaml.safe_load_all((root / "k8s.yaml").read_text()))
        self.assertEqual(
            {doc["kind"] for doc in documents},
            {"Namespace", "PersistentVolumeClaim", "Deployment", "Service"},
        )
        for document in documents:
            kubernetes_validate.validate(
                document, desired_version="1.32.0", strict=True
            )
        deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
        spec = deployment["spec"]["template"]["spec"]
        container = spec["containers"][0]
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "1")
        self.assertTrue(all("hostPath" not in volume for volume in spec["volumes"]))
        self.assertFalse(spec.get("hostIPC", False))
        self.assertFalse(spec.get("hostPID", False))
        settings = {item["name"]: item["value"] for item in container["env"]}
        self.assertEqual(settings["KOKORO_WORKERS"], "4")
        self.assertEqual(settings["KOKORO_MPS"], "1")
        self.assertEqual(settings["KOKORO_WORKSPACE_MIB"], "4096")
