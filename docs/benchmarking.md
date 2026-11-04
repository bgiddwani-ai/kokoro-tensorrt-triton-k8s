# Benchmarking and quality qualification

Compare the same text, voice, speed, token limits, model revisions, and GPU allocation. The supplied corpus has 12 English passages, two voices, and short/medium/long groups. Run eager and Triton GPU measurements separately so they do not compete for GPU memory or execution time.

## Triton benchmark on Kubernetes

Build the optional benchmark image. It adds the text frontend, CUDA PyTorch for the eager comparison, and the corpus to the server image. Choose a registry your cluster can pull:

```bash
export BENCHMARK_IMAGE=registry.example.com/your-team/kokoro-benchmark:v1
make benchmark-image IMAGE="$IMAGE" BENCHMARK_IMAGE="$BENCHMARK_IMAGE"
```

Wait until the service is ready, then submit the CPU client Job. It sends requests to the service and does not request a second GPU:

```bash
kubectl -n kokoro wait --for=condition=available deployment/kokoro --timeout=30m
kubectl -n kokoro delete job kokoro-benchmark --ignore-not-found
kubectl set image --local -f k8s/benchmark-job.yaml \
  benchmark="$BENCHMARK_IMAGE" -o yaml | kubectl -n kokoro apply -f -
kubectl -n kokoro wait --for=condition=complete job/kokoro-benchmark --timeout=30m
mkdir -p artifacts
kubectl -n kokoro logs job/kokoro-benchmark > artifacts/triton-job.log
```

The final JSON line includes per-request timings and summaries for concurrency 1, 2, 4, 8, 16, and 32. If a Job fails, inspect its logs and pod events. Warnings from dependencies may precede the JSON; do not treat the whole log as a JSON document. Use the same private image pull configuration as the server if needed.

Extract the final report:

```bash
python - <<'PY'
import json
from pathlib import Path
for line in reversed(Path('artifacts/triton-job.log').read_text().splitlines()):
    try:
        report = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(report, dict) and 'measurements' in report:
        Path('artifacts/triton.json').write_text(json.dumps(report, indent=2) + '\n')
        break
else:
    raise SystemExit('No completed benchmark report in Job logs')
PY
```

Use your chosen namespace instead of `kokoro` when you set `NAMESPACE` during deployment. Also change the Job's `--url` if benchmarking a Service in another namespace.

## Eager baseline on Kubernetes

The eager Job requests one GPU. On a cluster with only one available GPU, release the server's allocation first:

```bash
kubectl -n kokoro scale deployment/kokoro --replicas=0
kubectl -n kokoro wait --for=delete pod -l app.kubernetes.io/name=kokoro --timeout=2m
kubectl -n kokoro delete job kokoro-eager --ignore-not-found
kubectl set image --local -f k8s/eager-job.yaml \
  benchmark="$BENCHMARK_IMAGE" -o yaml | kubectl -n kokoro apply -f -
kubectl -n kokoro wait --for=condition=complete job/kokoro-eager --timeout=30m
kubectl -n kokoro logs job/kokoro-eager > artifacts/eager-job.log
kubectl -n kokoro scale deployment/kokoro --replicas=1
kubectl -n kokoro rollout status deployment/kokoro --timeout=30m
```

If your cluster needs `runtimeClassName`, node affinity, or tolerations for GPU pods, add the same rules to the eager Job. In a heterogeneous cluster, deliberately schedule eager and Triton to the same GPU model; a benchmark against a different GPU is not a controlled comparison.

## Local benchmark and WAV comparison

With the client requirements installed and the service forwarded:

```bash
python -m kokoro_triton.benchmark --backend triton \
  --concurrency 1,2,4,8,16,32 --duration 30 \
  --out artifacts/triton.json --audio-dir artifacts/triton-audio
```

On a CUDA machine, install a CUDA-compatible PyTorch build and the client requirements. Stop competing GPU workloads, then run:

```bash
python -m kokoro_triton.benchmark --backend eager --concurrency 1 \
  --duration 30 --out artifacts/eager.json --audio-dir artifacts/eager-audio
```

`Dockerfile.benchmark` includes CUDA PyTorch if you prefer a container. For example, on a Linux NVIDIA Docker host, this performs the eager run and saves generated audio/results to a generic local artifacts directory:

```bash
mkdir -p artifacts
docker run --rm --gpus 1 \
  -v "$PWD/artifacts:/results" \
  "$BENCHMARK_IMAGE" --backend eager --concurrency 1 \
  --out /results/eager.json --audio-dir /results/eager-audio
```

Transfer the generated artifacts between your own machines as needed so eager and Triton WAV directories are together. Then run an ASR-based regression check on the same files:

```bash
python -m pip install -r requirements-quality.txt
python -m kokoro_triton.quality \
  --eager-dir artifacts/eager-audio \
  --triton-dir artifacts/triton-audio \
  --out artifacts/quality.json
```

The checker verifies mono 24 kHz finite audio, measures duration changes, transcribes both sets with the same pinned Whisper model on CPU, and reports **signed** `triton - eager` WER/CER deltas. Default acceptance requires no increase in aggregate WER/CER and at most 1% duration change per passage. Set explicit thresholds for your application. Listen to matched samples as well: ASR scores do not measure voice identity or prosody. The small supplied corpus is a smoke/regression set, not a broad speech quality evaluation.

## What the timing means

- Text normalization, G2P, and voice preparation happen before timing.
- Triton timing covers prepared CPU tensors, RPC marshalling/network, server queueing/inference, and complete audio received on the client.
- Eager timing starts with tensors already on the GPU and includes complete audio copied back to host. It excludes RPC. Keep this difference explicit when interpreting speedups.
- Every corpus chunk is warmed before measurement. Each concurrency level runs a closed-loop workload for the requested duration; elapsed time includes draining outstanding requests.
- `requests_per_second` counts successful chunks, not original multi-chunk passages. `audio_seconds_per_second` is generated audio duration divided by wall time. Neither is token throughput.
- p50/p95/p99/max describe successful-request latency. Errors are counted and make the benchmark exit unsuccessfully. Inspect raw rows, not only percentiles.
- The service returns complete chunks. These are not streaming time-to-first-audio measurements.

## Tuning procedure

Measure workers 1, 2, and 4, both with and without MPS when supported. For each setting, rerun the complete concurrency matrix, retain raw JSON, check memory and errors, and compare tail latency as well as throughput. Confirm the chosen candidate with a longer sustained run, for example `--duration 300`.

Only change precision or model transformations after establishing the eager baseline and a quality gate on the same held-out workload. Retain the last accepted image/configuration so you can roll back. This repository defaults to FP32 and does not silently enable quantization, FP16, or dynamic batching.

Triton's `/metrics` endpoint provides server counters and GPU metrics when available. Record the public software versions, GPU model, worker/profile configuration, and workload alongside your own results. Generated artifacts, weights, caches, and credentials are ignored by Git.
