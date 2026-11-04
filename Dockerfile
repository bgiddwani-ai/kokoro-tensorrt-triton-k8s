FROM nvcr.io/nvidia/tritonserver:25.03-py3

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KOKORO_RUNTIME_ID=triton25.03-ort1.22.0-cuda12.8 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY requirements-runtime.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements-runtime.txt
COPY kokoro_triton /app/kokoro_triton
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN chmod 755 /app/scripts/entrypoint.sh && mkdir -p /data && chmod 777 /data
EXPOSE 8000 8001 8002
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
