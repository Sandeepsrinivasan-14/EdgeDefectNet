# syntax=docker/dockerfile:1
# Gradio demo + CLI for EdgeDefectNet (CPU, ONNX Runtime).
#   docker build -t edgedefectnet .
#   docker run --rm -p 7860:7860 edgedefectnet            # http://localhost:7860
#   docker run --rm -v "$PWD/wafers:/data:ro" edgedefectnet python -m edgedefectnet.predict /data/wafer.png
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Dependencies first so this layer is cached until pyproject.toml or the package changes.
COPY pyproject.toml README.md LICENSE ./
COPY edgedefectnet ./edgedefectnet
RUN pip install ".[demo]"

COPY app.py app.css ./
COPY models ./models
COPY examples ./examples

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV HOST=0.0.0.0 \
    PORT=7860 \
    EDGEDEFECTNET_MODEL=/app/models/edgedefectnet.onnx

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ['PORT'], timeout=4)"

CMD ["python", "app.py"]
