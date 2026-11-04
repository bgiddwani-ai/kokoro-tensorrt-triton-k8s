"""Text to 24 kHz WAV over Triton gRPC; preprocessing stays on the client."""

import argparse
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpc
from huggingface_hub import hf_hub_download

from .contract import validate_inputs
from .settings import EAGER_REPOSITORY, EAGER_REVISION, SAMPLE_RATE


class Frontend:
    def __init__(self, max_tokens=500):
        if not 4 <= max_tokens <= 500:
            raise ValueError("max_tokens must be between 4 and 500")
        self.max_tokens = max_tokens
        self.pipelines = {}
        self.voices = {}
        config = hf_hub_download(
            EAGER_REPOSITORY, "config.json", revision=EAGER_REVISION
        )
        self.vocab = json.loads(Path(config).read_text())["vocab"]

    def prepare(self, text, voice="af_heart", speed=1.0):
        if not text.strip():
            raise ValueError("text must not be empty")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be positive and finite")
        if not re.fullmatch(r"[a-z]{2}_[a-z0-9_]+", voice):
            raise ValueError("voice must be a model voice name such as af_heart")
        language = voice[0]
        if language not in self.pipelines:
            from kokoro import KPipeline

            self.pipelines[language] = KPipeline(
                lang_code=language, model=False, repo_id=EAGER_REPOSITORY
            )
        pipeline = self.pipelines[language]
        if voice not in self.voices:
            path = hf_hub_download(
                EAGER_REPOSITORY, f"voices/{voice}.pt", revision=EAGER_REVISION
            )
            self.voices[voice] = pipeline.load_voice(path)
        pack = self.voices[voice]
        inputs = []
        for segment in pipeline(text):
            remaining = segment.phonemes
            while remaining:
                limit = self.max_tokens - 2
                cut = len(remaining)
                if cut > limit:
                    cut = remaining.rfind(" ", 0, limit) or limit
                part, remaining = remaining[:cut], remaining[cut:].lstrip()
                ids = [
                    self.vocab[character]
                    for character in part
                    if character in self.vocab
                ]
                if len(ids) < 2:
                    raise ValueError(
                        "provide at least two supported phoneme tokens per chunk"
                    )
                value = {
                    "tokens": np.array([[0, *ids, 0]], np.int64),
                    "style": pack[len(part) - 1]
                    .numpy()
                    .reshape(1, 256)
                    .astype(np.float32),
                    "speed": np.array([speed], np.float32),
                }
                validate_inputs(value, self.max_tokens)
                inputs.append(value)
        if not inputs:
            raise ValueError("text produced no phonemes")
        return inputs


def infer(client, inputs, timeout=120.0):
    tensors = []
    for name, data in inputs.items():
        tensor = grpc.InferInput(
            name, data.shape, "INT64" if data.dtype == np.int64 else "FP32"
        )
        tensor.set_data_from_numpy(data)
        tensors.append(tensor)
    audio = client.infer(
        "kokoro",
        tensors,
        client_timeout=timeout,
        outputs=[grpc.InferRequestedOutput("audio")],
    ).as_numpy("audio")
    if audio is None or audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("server returned empty or non-finite audio")
    return audio.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--out", default="artifacts/sample.wav")
    args = parser.parse_args()
    chunks = Frontend(args.max_tokens).prepare(args.text, args.voice, args.speed)
    with grpc.InferenceServerClient(args.url) as client:
        audio = np.concatenate([infer(client, chunk) for chunk in chunks])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, SAMPLE_RATE, subtype="PCM_16")
    print(
        json.dumps(
            {
                "out": str(output),
                "sample_rate": SAMPLE_RATE,
                "audio_seconds": audio.size / SAMPLE_RATE,
                "chunks": len(chunks),
            }
        )
    )


if __name__ == "__main__":
    main()
