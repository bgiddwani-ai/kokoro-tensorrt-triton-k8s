"""Optional full-model CPU smoke check; downloads weights, never builds engines."""

import json

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from .contract import (
    DECODER_EN,
    DECODER_TE,
    DURATIONS,
    PREDICTED_EN,
    PREDICTED_TE,
    expansion_indices,
    warm_inputs,
)
from .graphs import infer_partition_shapes, make_portable, split_model
from .prepare import sha256
from .settings import ONNX_REPOSITORY, ONNX_REVISION, ONNX_SHA256, SAMPLE_RATE


def main():
    source = hf_hub_download(
        ONNX_REPOSITORY, "kokoro-82m-v1.0.onnx", revision=ONNX_REVISION
    )
    if sha256(source) != ONNX_SHA256:
        raise RuntimeError("public model checksum mismatch")
    models = split_model(make_portable(source))
    sessions = {}
    for stage in ("predictor", "decoder"):
        model = infer_partition_shapes(models.pop(stage))
        sessions[stage] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    inputs = warm_inputs()
    predicted = dict(
        zip(
            [value.name for value in sessions["predictor"].get_outputs()],
            sessions["predictor"].run(None, inputs),
        )
    )
    durations = predicted[DURATIONS].reshape(-1)
    indices = expansion_indices(durations)
    audio = sessions["decoder"].run(
        None,
        {
            DECODER_EN: np.take(
                predicted[PREDICTED_EN].transpose(0, 2, 1), indices, axis=2
            ),
            DECODER_TE: np.take(predicted[PREDICTED_TE], indices, axis=2),
            "style": inputs["style"],
        },
    )[0]
    if not audio.size or not np.isfinite(audio).all():
        raise RuntimeError("CPU graph smoke check returned invalid audio")
    print(
        json.dumps(
            {
                "onnx_revision": ONNX_REVISION,
                "sha256_verified": True,
                "finite": True,
                "audio_seconds": audio.size / SAMPLE_RATE,
                "frames": int(durations.sum()),
                "gpu_tested": False,
            }
        )
    )


if __name__ == "__main__":
    main()
