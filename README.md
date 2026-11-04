# Kokoro: TensorRT + Triton on Kubernetes

**One image. One Kubernetes file. One benchmark command.**

The image downloads pinned public models, builds engines on the assigned GPU, and serves 24 kHz speech. The selected runtime uses TensorRT subgraphs, CUDA/cuDNN LSTMs, four Triton workers, and MPS. Implementation code is in `app/`.

You need Docker, kubectl connected to a Linux x86_64 GPU cluster, an NVIDIA driver compatible with CUDA 12.8, the NVIDIA device plugin/runtime, and a default StorageClass. Use a dedicated full NVIDIA GPU (compute capability 7.5+) with enough memory for four workers. All paths below are container paths or relative to your checkout.

## 1. Build the container

```bash
git clone https://github.com/bgiddwani-ai/kokoro-tensorrt-triton-k8s.git
cd kokoro-tensorrt-triton-k8s

export IMAGE=registry.example.com/your-team/kokoro:v1
docker login registry.example.com
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
```

Replace the registry/team with your own. Your cluster must be able to pull the image.

## 2. Run on Kubernetes

```bash
sed "s#CONTAINER_IMAGE#$IMAGE#g" k8s.yaml | kubectl apply -f -
kubectl -n kokoro rollout status deployment/kokoro --timeout=30m
export POD=$(kubectl -n kokoro get pods -l app=kokoro -o jsonpath='{.items[0].metadata.name}')
```

The first startup downloads weights and builds engines; allow several minutes. View progress with `kubectl -n kokoro logs -f "$POD"`. `k8s.yaml` requests one GPU, creates model/cache storage, and exposes the service as `kokoro:8001`. If your cluster requires a specific StorageClass, RuntimeClass, or image pull secret, set it in this single manifest.

Generate a WAV without installing anything locally:

```bash
kubectl -n kokoro exec "$POD" -- python3 -m kokoro_triton.client \
  --text "Hello! The speech service is ready." --out /data/hello.wav
kubectl -n kokoro cp "$POD:/data/hello.wav" ./hello.wav
```

## 3. Reproduce the benchmark

```bash
kubectl -n kokoro exec "$POD" -- python3 -m kokoro_triton.run_benchmarks
kubectl -n kokoro cp "$POD:/data/benchmark" ./benchmark-output
```

Run this without production traffic: it temporarily unloads the serving model to measure eager and hybrid inference separately, then reloads it before the serving benchmark. It uses the same 12 English passages and two voices, three warmups/eight model repetitions, a concurrency sweep of 1/2/4/8/16/32, a 30-second sustained run at concurrency 4, and matched ASR quality checks. JSON results and WAVs are saved to `benchmark-output/`. The one image contains all required clients and benchmark dependencies.

## Our reference results

| Complete-audio model p50 | Eager | Optimized hybrid | Speedup |
| --- | ---: | ---: | ---: |
| Short | 28.39 ms | 14.61 ms | 1.94× |
| Medium | 40.46 ms | 27.10 ms | 1.49× |
| Long | 66.87 ms | 48.22 ms | 1.39× |

Four workers at concurrency 4 sustained **64.98 requests/s**, **679.66 audio-seconds/s**, **105.61 ms p95**, and **zero errors across 1,954 requests**. Eager/served WER was **2.614%** and CER **1.131%**, with zero signed deltas on this corpus.

- [HTML benchmark report](results/benchmark_report.html) — download and open in a browser; charts and tables work offline.
- [Summary JSON](results/summary.json) · [Full historical matrix CSV](results/matrix.csv) · [Per-request evidence](results/)

These are reference model/Triton container measurements from before Kubernetes packaging, not measured Kubernetes performance. Device/infrastructure identifiers are omitted. Rerun on your hardware; results vary. The minimal application reproduces the selected hybrid configuration and eager comparison; the report also retains historical tuning results. Eager model timing starts with GPU tensors; hybrid timing includes input marshalling; serving timing includes RPC/queueing. All return complete audio, not first-audio latency.

## Stop

```bash
kubectl -n kokoro scale deployment/kokoro --replicas=0
```

This releases the application's GPU allocation and preserves the PVC/cache. Public models are pinned in `app/kokoro_triton/settings.py`; weights and compiled engines are generated at runtime. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
