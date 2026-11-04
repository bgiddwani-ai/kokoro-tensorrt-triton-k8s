"""Fixed-workload latency, serving matrix, and sustained benchmarks."""

import argparse
import concurrent.futures
import itertools
import json
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpc

from .client import Frontend, infer
from .settings import EAGER_REPOSITORY, EAGER_REVISION, SAMPLE_RATE


def eager_runner(inputs):
    import torch
    from huggingface_hub import hf_hub_download
    from kokoro import KModel

    if not torch.cuda.is_available():
        raise RuntimeError("The eager comparison requires CUDA PyTorch")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = (
        KModel(
            repo_id=EAGER_REPOSITORY,
            config=hf_hub_download(
                EAGER_REPOSITORY, "config.json", revision=EAGER_REVISION
            ),
            model=hf_hub_download(
                EAGER_REPOSITORY, "kokoro-v1_0.pth", revision=EAGER_REVISION
            ),
        )
        .eval()
        .cuda()
    )

    def tensors(value):
        return (
            torch.from_numpy(value["tokens"]).cuda(),
            torch.from_numpy(value["style"]).cuda(),
            float(value["speed"][0]),
        )

    prepared = [tensors(value) for value in inputs]

    def evaluate(value):
        with torch.inference_mode():
            audio, _ = model.forward_with_tokens(*value)
            result = audio.float().cpu().numpy().reshape(-1)
            torch.cuda.synchronize()
        return result

    return lambda index: evaluate(prepared[index]), lambda value: evaluate(
        tensors(value)
    )


def statistics(rows, elapsed, concurrency):
    good = [row for row in rows if row["error"] is None]
    latencies = [row["latency_ms"] for row in good]
    return {
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "completed": len(good),
        "errors": len(rows) - len(good),
        "requests_per_second": len(good) / elapsed,
        "audio_seconds_per_second": sum(row["audio_seconds"] for row in good) / elapsed,
        "latency_ms": {
            name: float(np.percentile(latencies, q)) if latencies else None
            for name, q in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
        },
        "raw": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("eager", "hybrid", "triton"), default="triton"
    )
    parser.add_argument(
        "--mode", choices=("latency", "matrix", "sustained"), default="latency"
    )
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument(
        "--corpus", default=str(Path(__file__).resolve().parents[1] / "texts.json")
    )
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default="/data/benchmark/benchmark.json")
    parser.add_argument("--audio-dir")
    args = parser.parse_args()
    counts = [int(value) for value in args.concurrency.split(",")]
    if min(counts) < 1 or args.duration <= 0 or args.warmup < 0:
        parser.error("invalid concurrency, duration, or warmup")
    if args.backend != "triton" and (args.mode != "latency" or counts != [1]):
        parser.error("direct GPU comparisons use latency mode and concurrency 1")

    # Load CUDA/cuDNN for the hybrid backend before importing the text frontend.
    if args.backend == "hybrid":
        from .runtime import Kokoro

        model = Kokoro()
    frontend = Frontend()
    inputs, metadata = [], []
    corpus = json.loads(Path(args.corpus).read_text())
    for case in corpus:
        for chunk, value in enumerate(frontend.prepare(case["text"], case["voice"])):
            inputs.append(value)
            metadata.append(
                {
                    "case": case["id"],
                    "chunk": chunk,
                    "length": case["length"],
                    "voice": case["voice"],
                    "tokens": value["tokens"].shape[1],
                }
            )
    if not inputs:
        parser.error("corpus contains no inputs")
    clients, local = [], threading.local()
    clients_lock = threading.Lock()
    if args.backend == "eager":
        run, dynamic = eager_runner(inputs)
    elif args.backend == "hybrid":

        def dynamic(value):
            return model.run(value)[0]

        def run(index):
            return dynamic(inputs[index])
    else:

        def dynamic(value):
            if not hasattr(local, "client"):
                local.client = grpc.InferenceServerClient(args.url)
                with clients_lock:
                    clients.append(local.client)
            return infer(local.client, value)

        def run(index):
            return dynamic(inputs[index])

    def timed(index):
        started = time.perf_counter()
        try:
            audio = run(index)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("invalid audio")
            return {
                "input": index,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "audio_seconds": audio.size / SAMPLE_RATE,
                "error": None,
            }
        except Exception as error:
            return {
                "input": index,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "audio_seconds": 0.0,
                "error": str(error),
            }

    report = {
        "backend": args.backend,
        "mode": args.mode,
        "sample_rate": SAMPLE_RATE,
        "eager_revision": EAGER_REVISION,
        "cases": metadata,
        "measurements": [],
        "timing": "prepared tensors to complete host audio; Triton includes RPC and queueing",
    }
    try:
        for index in range(len(inputs)):
            for _ in range(max(1, args.warmup)):
                audio = run(index)
            if not audio.size or not np.isfinite(audio).all():
                raise RuntimeError("invalid warmup output")
            if args.audio_dir:
                path = Path(args.audio_dir)
                path.mkdir(parents=True, exist_ok=True)
                sf.write(
                    path
                    / f"{metadata[index]['case']:02d}_{metadata[index]['chunk']}.wav",
                    audio,
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
        for count in counts:
            started = time.perf_counter()
            if args.mode == "latency":
                rows = [timed(index) for index in range(len(inputs)) for _ in range(8)]
            elif args.mode == "matrix":
                requests = max(96, count * 12)
                with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                    rows = list(
                        pool.map(
                            timed, (index % len(inputs) for index in range(requests))
                        )
                    )
            else:
                barrier = threading.Barrier(count + 1)
                sequence = itertools.count()
                deadline = [0.0]

                def worker():
                    output = []
                    barrier.wait()
                    while time.perf_counter() < deadline[0]:
                        output.append(timed(next(sequence) % len(inputs)))
                    return output

                with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                    pending = [pool.submit(worker) for _ in range(count)]
                    started = time.perf_counter()
                    deadline[0] = started + args.duration
                    barrier.wait()
                    rows = [row for future in pending for row in future.result()]
            measured = statistics(rows, time.perf_counter() - started, count)
            report["measurements"].append(measured)
            print(
                json.dumps(
                    {key: value for key, value in measured.items() if key != "raw"}
                ),
                flush=True,
            )
        if args.mode == "latency":
            good = [
                row for row in report["measurements"][0]["raw"] if row["error"] is None
            ]
            report["core_p50_ms"] = {
                group: float(
                    np.median(
                        [
                            row["latency_ms"]
                            for row in good
                            if metadata[row["input"]]["length"] == group
                        ]
                    )
                )
                for group in ("short", "medium", "long")
            }
        if args.backend != "hybrid" and args.mode != "sustained":
            text_rows = []
            for case in corpus:
                for rep in range(3):
                    started = time.perf_counter()
                    for value in frontend.prepare(case["text"], case["voice"]):
                        dynamic(value)
                    text_rows.append(
                        {
                            "case": case["id"],
                            "length": case["length"],
                            "rep": rep,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
            report["text_to_audio_raw"] = text_rows
            report["text_to_audio_p50_ms"] = {
                group: float(
                    np.median(
                        [
                            row["latency_ms"]
                            for row in text_rows
                            if row["length"] == group
                        ]
                    )
                )
                for group in ("short", "medium", "long")
            }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if any(measured["errors"] for measured in report["measurements"]):
            raise SystemExit("benchmark failed; inspect raw request errors")
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    main()
