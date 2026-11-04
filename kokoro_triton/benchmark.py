"""Matched-input eager and Triton benchmarks with queue-inclusive RPC timing."""

import argparse
import concurrent.futures
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
        raise RuntimeError(
            "eager GPU benchmark requires a CUDA-capable PyTorch installation"
        )
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
    tensors = [
        (
            torch.from_numpy(x["tokens"]).cuda(),
            torch.from_numpy(x["style"]).cuda(),
            float(x["speed"][0]),
        )
        for x in inputs
    ]

    def run(index):
        with torch.inference_mode():
            audio, _ = model.forward_with_tokens(*tensors[index])
            result = audio.float().cpu().numpy().reshape(-1)
            torch.cuda.synchronize()
        return result

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "eager"), default="triton")
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--corpus", default="benchmarks/texts.json")
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--out", default="artifacts/benchmark.json")
    parser.add_argument("--audio-dir", help="save one output WAV per corpus chunk")
    parser.add_argument(
        "--json-stdout", action="store_true", help="include raw results in pod logs"
    )
    args = parser.parse_args()
    concurrency = [int(value) for value in args.concurrency.split(",")]
    if any(value < 1 for value in concurrency) or args.duration <= 0 or args.warmup < 0:
        parser.error(
            "concurrency and duration must be positive; warmup must be nonnegative"
        )
    if args.backend == "eager" and concurrency != [1]:
        parser.error("run the eager latency baseline with --concurrency 1")
    frontend = Frontend(args.max_tokens)
    inputs, metadata = [], []
    for case in json.loads(Path(args.corpus).read_text()):
        chunks = frontend.prepare(case["text"], case["voice"], 1.0)
        for chunk, value in enumerate(chunks):
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
    local = threading.local()
    clients = []
    clients_lock = threading.Lock()
    if args.backend == "eager":
        run = eager_runner(inputs)
    else:

        def run(index):
            if not hasattr(local, "client"):
                local.client = grpc.InferenceServerClient(args.url)
                with clients_lock:
                    clients.append(local.client)
            return infer(local.client, inputs[index])

    try:
        for index in range(len(inputs)):
            for _ in range(max(1, args.warmup)):
                audio = run(index)
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
        report = {
            "backend": args.backend,
            "sample_rate": SAMPLE_RATE,
            "eager_revision": EAGER_REVISION,
            "timing": "prepared tensors to complete host audio; Triton includes RPC and queueing",
            "g2p_included": False,
            "cases": metadata,
            "measurements": [],
        }
        for count in concurrency:
            barrier = threading.Barrier(count + 1)
            deadline = [0.0]

            def worker(worker_id):
                rows = []
                sequence = worker_id
                barrier.wait()
                while time.perf_counter() < deadline[0]:
                    index = sequence % len(inputs)
                    started = time.perf_counter()
                    try:
                        output = run(index)
                        if not np.isfinite(output).all():
                            raise RuntimeError("non-finite output")
                        row = {
                            "input": index,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                            "audio_seconds": output.size / SAMPLE_RATE,
                            "error": None,
                        }
                    except Exception as error:
                        row = {
                            "input": index,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                            "audio_seconds": 0.0,
                            "error": str(error),
                        }
                    rows.append(row)
                    sequence += count
                return rows

            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                pending = [pool.submit(worker, worker_id) for worker_id in range(count)]
                started = time.perf_counter()
                deadline[0] = started + args.duration
                barrier.wait()
                rows = [row for future in pending for row in future.result()]
                elapsed = time.perf_counter() - started
            successful = [row for row in rows if row["error"] is None]
            latencies = [row["latency_ms"] for row in successful]
            measurement = {
                "concurrency": count,
                "elapsed_seconds": elapsed,
                "completed": len(successful),
                "errors": len(rows) - len(successful),
                "requests_per_second": len(successful) / elapsed,
                "audio_seconds_per_second": sum(
                    row["audio_seconds"] for row in successful
                )
                / elapsed,
                "latency_ms": {
                    name: float(np.percentile(latencies, percentile))
                    if latencies
                    else None
                    for name, percentile in (
                        ("p50", 50),
                        ("p95", 95),
                        ("p99", 99),
                        ("max", 100),
                    )
                },
                "raw": rows,
            }
            report["measurements"].append(measurement)
            if not args.json_stdout:
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in measurement.items()
                            if key != "raw"
                        }
                    ),
                    flush=True,
                )
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.json_stdout:
            print(json.dumps(report, allow_nan=False), flush=True)
        if any(row["errors"] for row in report["measurements"]):
            raise SystemExit("benchmark failed: inspect errors in raw results")
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    main()
