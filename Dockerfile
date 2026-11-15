# Standalone Qwen-Image MCP sidecar — fully self-contained (no z-image-studio).
#
# Base is plain CPython: the pinned torch wheels bundle the complete CUDA 13
# runtime as nvidia-* pip packages, so no cuda base image is needed — only a
# recent NVIDIA driver on the host. All versions below are the exact matrix
# verified to load and run Qwen-Image-2512 (bnb-4bit + Lightning LoRA).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    IMAGE_DIR=/srv/images-qwen

# torch first, from the official cu130 wheel index (self-contained CUDA runtime)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 \
    && python -c "import torch; print('torch', torch.__version__)"

# Everything else: the exact verified version matrix
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import bitsandbytes, diffusers, transformers, accelerate, peft, fastapi, uvicorn, mcp, PIL; print('deps ok')"

COPY server.py ./
RUN mkdir -p /srv/images-qwen

EXPOSE 8000
ENTRYPOINT ["python", "server.py"]
