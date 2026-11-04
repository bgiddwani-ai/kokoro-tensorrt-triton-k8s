import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h

from kokoro_triton.graphs import make_portable


class GraphParityTests(unittest.TestCase):
    def compare(self, kind, inputs, output_shape, **attributes):
        node = h.make_node(
            kind, list(inputs), ["output"], domain="com.microsoft", **attributes
        )
        graph = h.make_graph(
            [node],
            "parity",
            [
                h.make_tensor_value_info(name, T.FLOAT, list(value.shape))
                for name, value in inputs.items()
            ],
            [h.make_tensor_value_info("output", T.FLOAT, output_shape)],
        )
        model = h.make_model(
            graph,
            ir_version=10,
            opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            original, portable = (
                Path(temporary) / name for name in ("original.onnx", "portable.onnx")
            )
            onnx.save(model, original)
            make_portable(original, portable)
            expected = ort.InferenceSession(
                str(original), providers=["CPUExecutionProvider"]
            ).run(None, inputs)[0]
            actual = ort.InferenceSession(
                str(portable), providers=["CPUExecutionProvider"]
            ).run(None, inputs)[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_biased_gelu(self):
        self.compare(
            "FastGelu",
            {
                "x": np.linspace(-4, 4, 24, dtype=np.float32).reshape(2, 3, 4),
                "bias": np.array([0.2, -0.1, 0.3, 0.5], np.float32),
            },
            [2, 3, 4],
        )

    def test_scaled_transposed_matmul_2d_and_3d(self):
        rng = np.random.default_rng(42)
        for prefix in ((), (2,)):
            with self.subTest(prefix=prefix):
                self.compare(
                    "FusedMatMul",
                    {
                        "a": rng.normal(size=(*prefix, 4, 3)).astype(np.float32),
                        "b": rng.normal(size=(*prefix, 4, 5)).astype(np.float32),
                    },
                    [*prefix, 3, 5],
                    transA=1,
                    alpha=0.25,
                )

    def test_residual_bias_layer_norm(self):
        rng = np.random.default_rng(42)
        self.compare(
            "SkipLayerNormalization",
            {
                "x": rng.normal(size=(1, 3, 4)).astype(np.float32),
                "skip": rng.normal(size=(1, 3, 4)).astype(np.float32),
                "gamma": np.array([1.0, 0.7, 1.2, 0.9], np.float32),
                "beta": np.array([0.0, 0.1, 0.2, -0.1], np.float32),
                "bias": np.array([0.2, -0.1, 0.3, 0.5], np.float32),
            },
            [1, 3, 4],
            epsilon=1e-5,
        )
