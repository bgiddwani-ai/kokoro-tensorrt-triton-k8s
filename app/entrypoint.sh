#!/usr/bin/env bash
set -euo pipefail

export KOKORO_DATA_DIR="${KOKORO_DATA_DIR:-/data}"
export HF_HOME="${HF_HOME:-${KOKORO_DATA_DIR}/huggingface}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${KOKORO_DATA_DIR}/cuda-cache}"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/kokoro-mps/pipe
export CUDA_MPS_LOG_DIRECTORY="${KOKORO_DATA_DIR}/logs/mps"
mkdir -p "$HF_HOME" "$CUDA_CACHE_PATH" "${CUPY_CACHE_DIR:-/data/cupy-cache}"
child_pid=""
mps_started=0
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    if [[ "$mps_started" == 1 ]]; then
        printf 'quit\n' | nvidia-cuda-mps-control || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "${KOKORO_MPS:-1}" == 1 ]]; then
    command -v nvidia-cuda-mps-control >/dev/null || {
        echo "MPS binary unavailable. Set KOKORO_MPS=0 or enable NVIDIA utility capabilities." >&2
        exit 1
    }
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    mps_started=1
fi

python3 -m kokoro_triton.bootstrap --env-file /tmp/kokoro-env &
child_pid=$!
wait "$child_pid"
child_pid=""
source /tmp/kokoro-env

tritonserver --model-repository=/tmp/model_repository \
    --strict-model-config=true --exit-timeout-secs=30 \
    --model-control-mode=explicit --load-model=kokoro \
    --allow-http=true --allow-grpc=true --allow-metrics=true \
    --log-verbose=0 &
child_pid=$!
wait "$child_pid"
