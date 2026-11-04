"""Triton Python backend entry point."""

import numpy as np
import triton_python_backend_utils as pb_utils

from kokoro_triton.contract import warm_inputs
from kokoro_triton.runtime import Kokoro


class TritonPythonModel:
    def initialize(self, args):
        self.model = Kokoro()
        self.model.run(warm_inputs())

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                inputs = {}
                for name in ("tokens", "style", "speed"):
                    tensor = pb_utils.get_input_tensor_by_name(request, name)
                    if tensor is None:
                        raise ValueError(f"missing input: {name}")
                    inputs[name] = tensor.as_numpy()
                audio, _ = self.model.run(inputs)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "audio", audio.astype(np.float32, copy=False)
                            )
                        ]
                    )
                )
            except Exception as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
