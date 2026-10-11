# syntax=docker/dockerfile:1.6
# MSSD: Marine Surface Debris Surveillance
#
# Targets:
#   cpu   Random Forest training/evaluation and CPU inference
#   gpu   MSSD-Net training and GPU inference (requires nvidia-container-toolkit)
#
# Build:
#   docker build -t msdd:cpu --target cpu .
#   docker build -t msdd:gpu --target gpu .
#
# Run:
#   docker run --rm -it --gpus all \
#       -v "$PWD/msdd_net/data:/workspace/msdd_net/data" \
#       -v "$PWD/trained_models:/workspace/msdd_net/trained_models" \
#       msdd:gpu python train_marida.py --model mssd --variant baseline

# =====================================================================
# Shared base: system deps + python + repo layout
# =====================================================================
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        proj-data \
        proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# =====================================================================
# CPU target: PyTorch CPU build, small and portable
# =====================================================================
FROM base AS cpu

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /workspace/requirements.txt

COPY . /workspace

WORKDIR /workspace/msdd_net

ENV OMP_NUM_THREADS=4

CMD ["python", "train_rf.py", "--features", "ss_si_glcm"]

# =====================================================================
# GPU target: PyTorch CUDA 12.1 build (adjust to your driver if needed)
# =====================================================================
FROM base AS gpu

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /workspace/requirements.txt

COPY . /workspace

WORKDIR /workspace/msdd_net

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "train_marida.py", "--model", "mssd", "--variant", "baseline"]