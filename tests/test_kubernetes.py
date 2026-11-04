import os
import shutil
import subprocess
import unittest
from pathlib import Path

import kubernetes_validate
import yaml

ROOT = Path(__file__).resolve().parents[1]


class KubernetesTests(unittest.TestCase):
    def render(self, **overrides):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/kube.sh"), "render"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **overrides},
        )
        return list(yaml.safe_load_all(result.stdout))

    def check_schema(self, documents):
        for document in documents:
            kubernetes_validate.validate(
                document, desired_version="1.32.0", strict=True
            )

    @unittest.skipUnless(
        shutil.which("kubectl"), "install kubectl to test rendered manifests"
    )
    def test_default_manifests_and_gpu_isolation(self):
        documents = self.render()
        self.check_schema(documents)
        deployment = next(item for item in documents if item["kind"] == "Deployment")
        spec = deployment["spec"]["template"]["spec"]
        self.assertFalse(spec.get("hostPID", False))
        self.assertFalse(spec.get("hostIPC", False))
        self.assertTrue(all("hostPath" not in volume for volume in spec["volumes"]))
        container = spec["containers"][0]
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "1")
        config = next(item for item in documents if item["kind"] == "ConfigMap")
        self.assertEqual(
            container["envFrom"][0]["configMapRef"]["name"], config["metadata"]["name"]
        )

    @unittest.skipUnless(
        shutil.which("kubectl"), "install kubectl to test rendered manifests"
    )
    def test_overrides_change_config_and_namespace_consistently(self):
        documents = self.render(
            NAMESPACE="speech-demo",
            IMAGE="registry.example.com/tts:v2",
            WORKERS="4",
            MPS="1",
            WORKSPACE_MIB="4096",
            STORAGE_CLASS="fast-disk",
            RUNTIME_CLASS="nvidia",
        )
        self.check_schema(documents)
        for item in documents:
            if item["kind"] == "Namespace":
                self.assertEqual(item["metadata"]["name"], "speech-demo")
            else:
                self.assertEqual(item["metadata"]["namespace"], "speech-demo")
        config = next(item for item in documents if item["kind"] == "ConfigMap")
        self.assertEqual(config["data"]["KOKORO_WORKERS"], "4")
        self.assertEqual(config["data"]["KOKORO_MPS"], "1")
        deployment = next(item for item in documents if item["kind"] == "Deployment")
        spec = deployment["spec"]["template"]["spec"]
        self.assertEqual(spec["runtimeClassName"], "nvidia")
        self.assertEqual(spec["containers"][0]["image"], "registry.example.com/tts:v2")
        pvc = next(
            item for item in documents if item["kind"] == "PersistentVolumeClaim"
        )
        self.assertEqual(pvc["spec"]["storageClassName"], "fast-disk")

    def test_benchmark_jobs_are_valid(self):
        for name in ("benchmark-job.yaml", "eager-job.yaml"):
            self.check_schema([yaml.safe_load((ROOT / "k8s" / name).read_text())])
