FROM nvcr.io/nvidia/tritonserver:25.03-py3

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KOKORO_RUNTIME_ID=triton25.03-ort1.22.0-cuda12.8 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    KOKORO_DATA_DIR=/data \
    HF_HOME=/data/huggingface \
    CUDA_CACHE_PATH=/data/cuda-cache \
    CUPY_CACHE_DIR=/data/cupy-cache \
    CUDA_MPS_PIPE_DIRECTORY=/tmp/kokoro-mps/pipe \
    KOKORO_QUALITY_PYTHON=/opt/quality/bin/python
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --no-cache-dir -r requirements.txt
# ASR depends on CPU ONNX Runtime; isolate it so it cannot replace the GPU package.
RUN python3 -m venv /opt/quality \
    && /opt/quality/bin/pip install --no-cache-dir numpy==1.26.4 onnxruntime==1.22.0 \
       soundfile==0.14.0 huggingface-hub==0.34.4 faster-whisper==1.2.1 jiwer==4.0.0
COPY app/ /app/
RUN chmod 755 /app/entrypoint.sh && mkdir -p /data && chmod 777 /data
EXPOSE 8000 8001 8002
ENTRYPOINT ["/app/entrypoint.sh"]
