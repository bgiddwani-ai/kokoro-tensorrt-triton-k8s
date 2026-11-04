"""Prepare models, build engines on the assigned GPU, and render Triton's config."""

import argparse
import fcntl
import json
import os
import shlex
import time
from pathlib import Path

import cupy as cp
import numpy as np
import onnxruntime as ort

from .cache import cache_key
from .contract import DECODER_EN, DECODER_TE, warm_inputs
from .prepare import prepare, sha256
from .settings import Settings


def render_repository(path, workers):
    model = Path(path) / "kokoro"
    (model / "1").mkdir(parents=True, exist_ok=True)
    (model / "1/model.py").write_text(
        "from kokoro_triton.triton_model import TritonPythonModel\n"
    )
    (model / "config.pbtxt").write_text(
        """name: "kokoro"
backend: "python"
max_batch_size: 0
input [
  { name: "tokens" data_type: TYPE_INT64 dims: [1, -1] },
  { name: "style" data_type: TYPE_FP32 dims: [1, 256] },
  { name: "speed" data_type: TYPE_FP32 dims: [1] }
]
output [{ name: "audio" data_type: TYPE_FP32 dims: [-1] }]
instance_group [{ count: """
        + str(workers)
        + """ kind: KIND_GPU gpus: [0] }]
"""
    )


def bootstrap(env_file):
    settings = Settings.from_env()
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("Expose exactly one GPU to this container")
    for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"missing {provider}; use the GPU server image")
    prepared = prepare(settings)
    device = cp.cuda.runtime.getDeviceProperties(0)
    identity = {
        "gpu_name": device["name"].decode(),
        "compute_capability": [device["major"], device["minor"]],
        "gpu_memory_bytes": device["totalGlobalMem"],
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "onnxruntime": ort.__version__,
        "runtime_image": os.getenv("KOKORO_RUNTIME_ID", "custom"),
        "runtime_source": sha256(Path(__file__).with_name("runtime.py")),
        "models": prepared,
        "max_tokens": settings.max_tokens,
        "max_frames": settings.max_frames,
        "workspace_mib": settings.workspace_mib,
        "cache_epoch": os.getenv("KOKORO_CACHE_EPOCH", "1"),
    }
    cache = settings.data / "engines" / cache_key(identity)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["KOKORO_ENGINE_CACHE"] = str(cache)
    from .runtime import Kokoro

    timings = []
    # Serialize preparation and priming when a volume is reused after a restart.
    with open(cache / ".build.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = Kokoro(settings, cache)
        for length in (settings.max_tokens, 4):
            started = time.perf_counter()
            model.predictor.run(warm_inputs(length))
            timings.append(
                {
                    "stage": "predictor",
                    "size": length,
                    "seconds": time.perf_counter() - started,
                }
            )
        for length in (settings.max_frames, 4):
            started = time.perf_counter()
            model.decoder.run(
                {
                    DECODER_EN: np.zeros((1, 640, length), np.float32),
                    DECODER_TE: np.zeros((1, 512, length), np.float32),
                    "style": np.zeros((1, 256), np.float32),
                }
            )
            timings.append(
                {
                    "stage": "decoder",
                    "size": length,
                    "seconds": time.perf_counter() - started,
                }
            )
        model.run(warm_inputs())
        for stage in ("predictor", "decoder"):
            if not list((cache / stage).glob("*.engine")):
                raise RuntimeError(
                    f"no TensorRT engines built for {stage}; refusing fallback-only serving"
                )
        (cache / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        (cache / "priming.json").write_text(json.dumps(timings, indent=2) + "\n")
    render_repository("/tmp/model_repository", settings.workers)
    Path(env_file).write_text(
        "export KOKORO_ENGINE_CACHE=" + shlex.quote(str(cache)) + "\n"
    )
    print("Models and TensorRT engines ready; starting Triton", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="/tmp/kokoro-env")
    bootstrap(parser.parse_args().env_file)
