import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from kokoro_triton.run_benchmarks import run


class BenchmarkControlTests(unittest.TestCase):
    def test_failed_baseline_reloads_serving_model(self):
        events = []

        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_model_ready(self, name):
                return True

            def unload_model(self, name):
                events.append("unload")

            def load_model(self, name):
                events.append("load")

        grpc = types.ModuleType("tritonclient.grpc")
        grpc.InferenceServerClient = lambda url: Client()
        parent = types.ModuleType("tritonclient")
        parent.grpc = grpc
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                "sys.modules", {"tritonclient": parent, "tritonclient.grpc": grpc}
            ),
            patch.object(
                Path,
                "read_text",
                return_value="export KOKORO_ENGINE_CACHE=/data/engines/example\n",
            ),
            patch(
                "kokoro_triton.run_benchmarks.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["benchmark"]),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            run(Path(directory), "localhost:8001")
        self.assertEqual(events, ["unload", "load"])
