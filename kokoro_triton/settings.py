"""Configuration shared by preparation, warmup and serving."""

import os
from dataclasses import dataclass
from pathlib import Path

ONNX_REPOSITORY = "nvidia/kokoro-82M-onnx-opt"
ONNX_REVISION = "4715915bc1a4af683ceb826a2d365d59ec666d13"
ONNX_SHA256 = "89abd5cc24130a0e3b9047a6c08ab4cde5bf5f83c92aacc02cffc0084fc3ebfa"
EAGER_REPOSITORY = "hexgrad/Kokoro-82M"
EAGER_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SAMPLE_RATE = 24000


@dataclass(frozen=True)
class Settings:
    data: Path
    workers: int
    max_tokens: int
    max_frames: int
    workspace_mib: int
    threads: int

    @classmethod
    def from_env(cls):
        value = cls(
            data=Path(os.getenv("KOKORO_DATA_DIR", "/data")),
            workers=int(os.getenv("KOKORO_WORKERS", "1")),
            max_tokens=int(os.getenv("KOKORO_MAX_TOKENS", "500")),
            max_frames=int(os.getenv("KOKORO_MAX_FRAMES", "2000")),
            workspace_mib=int(os.getenv("KOKORO_WORKSPACE_MIB", "2048")),
            threads=int(os.getenv("KOKORO_CPU_THREADS", "4")),
        )
        if not 1 <= value.workers <= 16:
            raise ValueError("KOKORO_WORKERS must be between 1 and 16")
        if not 4 <= value.max_tokens <= 500:
            raise ValueError("KOKORO_MAX_TOKENS must be between 4 and 500")
        if not 4 <= value.max_frames <= 2000:
            raise ValueError("KOKORO_MAX_FRAMES must be between 4 and 2000")
        if value.workspace_mib < 256 or value.threads < 1:
            raise ValueError(
                "workspace must be >=256 MiB; CPU threads must be positive"
            )
        return value

    @property
    def models(self):
        return self.data / "models"
