"""One command for isolated eager/hybrid comparisons, serving load, and quality."""

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run(output, url):
    import tritonclient.grpc as grpc

    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    _, cache = (
        Path("/tmp/kokoro-env")
        .read_text()
        .strip()
        .removeprefix("export ")
        .split("=", 1)
    )
    environment["KOKORO_ENGINE_CACHE"] = shlex.split(cache)[0]

    def benchmark(backend, *arguments):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kokoro_triton.benchmark",
                "--backend",
                backend,
                "--url",
                url,
                *arguments,
            ],
            env=environment,
            check=True,
        )

    with grpc.InferenceServerClient(url) as client:
        if not client.is_model_ready("kokoro"):
            raise RuntimeError("Wait for the model to become ready before benchmarking")
        client.unload_model("kokoro")
        try:
            for backend in ("eager", "hybrid"):
                benchmark(
                    backend,
                    "--out",
                    str(output / f"{backend}.json"),
                    "--audio-dir",
                    str(output / f"{backend}-audio"),
                )
        finally:
            # Each GPU comparison runs in a separate process. Its CUDA allocations
            # are released before Triton's workers are loaded again.
            client.load_model("kokoro")
    benchmark(
        "triton",
        "--mode",
        "matrix",
        "--concurrency",
        "1,2,4,8,16,32",
        "--warmup",
        "1",
        "--out",
        str(output / "matrix.json"),
        "--audio-dir",
        str(output / "served-audio"),
    )
    benchmark(
        "triton",
        "--mode",
        "sustained",
        "--concurrency",
        "4",
        "--duration",
        "30",
        "--warmup",
        "1",
        "--out",
        str(output / "sustained.json"),
    )
    subprocess.run(
        [
            os.getenv("KOKORO_QUALITY_PYTHON", sys.executable),
            "-m",
            "kokoro_triton.quality",
            "--eager-dir",
            str(output / "eager-audio"),
            "--triton-dir",
            str(output / "served-audio"),
            "--out",
            str(output / "quality.json"),
        ],
        env=environment,
        check=True,
    )
    print(f"Benchmark complete. JSON results and audio: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/data/benchmark"))
    parser.add_argument("--url", default="localhost:8001")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.out, args.url)


if __name__ == "__main__":
    main()
