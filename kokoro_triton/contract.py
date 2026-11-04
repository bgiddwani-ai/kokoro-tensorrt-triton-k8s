"""CPU-only validation, independent of the GPU runtime."""

import numpy as np

DURATIONS = "onnx::CumSum_3090"
PREDICTED_EN = "/text_encoder/Where_13"
PREDICTED_TE = "/text_encoder_1/Where_7_output_0"
DECODER_EN = "/MatMul_output_0"
DECODER_TE = "/MatMul_1_output_0"


def validate_inputs(inputs, max_tokens=500):
    tokens, style, speed = (inputs[n] for n in ("tokens", "style", "speed"))
    if tokens.dtype != np.int64 or tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("tokens must be INT64 with shape [1, sequence_length]")
    if not 4 <= tokens.shape[1] <= max_tokens:
        raise ValueError(f"sequence_length must be between 4 and {max_tokens}")
    if np.any(tokens < 0) or np.any(tokens >= 178):
        raise ValueError("token IDs must be in [0, 177]")
    if tokens[0, 0] != 0 or tokens[0, -1] != 0:
        raise ValueError("tokens must include leading and trailing boundary ID 0")
    if (
        style.shape != (1, 256)
        or style.dtype != np.float32
        or not np.isfinite(style).all()
    ):
        raise ValueError("style must be finite FP32 with shape [1, 256]")
    if speed.shape != (1,) or speed.dtype != np.float32:
        raise ValueError("speed must be FP32 with shape [1]")
    if not np.isfinite(speed).all() or speed[0] <= 0:
        raise ValueError("speed must be positive and finite")


def expansion_indices(durations, max_frames=2000):
    durations = np.asarray(durations)
    if durations.ndim != 1 or not np.issubdtype(durations.dtype, np.integer):
        raise ValueError("durations must be an integer vector")
    # Check the bound before allocating or summing potentially huge values.
    if durations.size == 0 or np.any(durations < 1) or np.any(durations > max_frames):
        raise ValueError("invalid predicted durations")
    if int(durations.sum()) > max_frames:
        raise ValueError(
            "utterance exceeds frame limit; shorten text or increase speed"
        )
    return np.repeat(np.arange(durations.size, dtype=np.int64), durations)


def warm_inputs(length=4):
    tokens = np.full((1, length), 16, dtype=np.int64)
    tokens[0, [0, -1]] = 0
    return {
        "tokens": tokens,
        "style": np.zeros((1, 256), np.float32),
        "speed": np.ones((1,), np.float32),
    }
