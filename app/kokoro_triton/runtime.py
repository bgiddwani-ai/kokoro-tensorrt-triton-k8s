"""TensorRT subgraphs, CUDA/cuDNN LSTMs, and GPU-resident stage buffers."""

import os
from pathlib import Path

import cupy as cp
import numpy as np
import onnxruntime as ort

from .contract import (
    DECODER_EN,
    DECODER_TE,
    DURATIONS,
    PREDICTED_EN,
    PREDICTED_TE,
    expansion_indices,
    validate_inputs,
)
from .settings import Settings


class Engine:
    def __init__(self, name, settings, cache_root):
        cache = Path(cache_root) / name
        cache.mkdir(parents=True, exist_ok=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_profiling = os.getenv("KOKORO_PROFILE", "0") == "1"
        options.profile_file_prefix = str(cache / "profile")
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_max_workspace_size": str(settings.workspace_mib << 20),
                    "trt_fp16_enable": False,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache),
                    "trt_op_types_to_exclude": "LSTM",
                    "trt_min_subgraph_size": 5,
                    "trt_builder_optimization_level": 3,
                },
            ),
            (
                "CUDAExecutionProvider",
                {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"},
            ),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(
            str(settings.models / f"hybrid-{name}.onnx"), options, providers=providers
        )
        if self.session.get_providers()[0] != "TensorrtExecutionProvider":
            raise RuntimeError(
                "TensorRT provider failed to initialize; check GPU libraries"
            )
        self.names = [output.name for output in self.session.get_outputs()]

    def run(self, inputs):
        binding = self.session.io_binding()
        owners = []
        for name, value in inputs.items():
            if isinstance(value, cp.ndarray):
                contiguous = cp.ascontiguousarray(value)
                owners.append(contiguous)
                binding.bind_input(
                    name,
                    "cuda",
                    0,
                    contiguous.dtype,
                    contiguous.shape,
                    contiguous.data.ptr,
                )
            else:
                contiguous = np.ascontiguousarray(value)
                owners.append(contiguous)
                binding.bind_cpu_input(name, contiguous)
        for name in self.names:
            binding.bind_output(name, "cuda", 0)
        cp.cuda.get_current_stream().synchronize()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        outputs = {}
        for name, value in zip(self.names, binding.get_outputs()):
            dtype = np.int64 if "int64" in value.data_type() else np.float32
            size = int(np.prod(value.shape())) * np.dtype(dtype).itemsize
            # Retain OrtValue as owner until CuPy releases its view.
            memory = cp.cuda.UnownedMemory(value.data_ptr(), size, value)
            outputs[name] = cp.ndarray(
                value.shape(), dtype, cp.cuda.MemoryPointer(memory, 0)
            )
        return outputs


class Kokoro:
    def __init__(self, settings=None, cache_root=None):
        self.settings = settings or Settings.from_env()
        cache_root = cache_root or os.environ["KOKORO_ENGINE_CACHE"]
        self.predictor = Engine("predictor", self.settings, cache_root)
        self.decoder = Engine("decoder", self.settings, cache_root)

    def run(self, inputs):
        validate_inputs(inputs, self.settings.max_tokens)
        predicted = self.predictor.run(inputs)
        durations = cp.asnumpy(predicted[DURATIONS]).reshape(-1)
        indices = cp.asarray(expansion_indices(durations, self.settings.max_frames))
        en = cp.take(predicted[PREDICTED_EN].transpose(0, 2, 1), indices, axis=2)
        te = cp.take(predicted[PREDICTED_TE], indices, axis=2)
        audio = self.decoder.run(
            {DECODER_EN: en, DECODER_TE: te, "style": inputs["style"]}
        )["audio"]
        audio = cp.asnumpy(audio).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("non-finite or empty audio rejected")
        return audio, durations
