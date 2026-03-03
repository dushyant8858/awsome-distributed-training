# Dockerfile for QLoRA Fine-tuning of Qwen3-8B
# =============================================
# Base Image: NVIDIA CUDA 12.8 with cuDNN 9
# Python: 3.10
# Key Libraries: PyTorch, Transformers, PEFT, BitsAndBytes, DeepSpeed
#
# Build:
#   docker build -t qwen3-qlora-training:latest .
#
# If you encounter CUBLAS errors at runtime (typically caused by CUDA
# library conflicts on the host), switch the torch index URL below to
# cu126 as a fallback — see docs/TROUBLESHOOTING.md.

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    git-lfs \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Stage 2: Install Python dependencies
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support
# Note: torch 2.10+ has a breaking LR scheduler change (strict zip) that is
# incompatible with some DeepSpeed/transformers versions. Pin to <2.10 until
# upstream libraries catch up.
RUN pip install --no-cache-dir 'torch>=2.7.0,<2.10.0' --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final image with application code
FROM dependencies AS final

# Copy source code and entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
COPY src/ /app/src/
COPY configs/ /app/configs/

# Create directories for outputs and cache
RUN mkdir -p /workspace/outputs /workspace/hf_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/workspace/hf_cache
ENV PYTHONUNBUFFERED=1
# Do NOT set CUDA_VISIBLE_DEVICES here — let torchrun / K8s manage GPU visibility
# DeepSpeed / NCCL settings for multi-GPU communication
ENV NCCL_DEBUG=INFO
ENV NCCL_SOCKET_IFNAME=^lo

# Entrypoint reads PET_* env vars set by the Kubeflow Training Operator
# and launches torchrun with the correct number of processes per node.
# Falls back to single-process mode outside PyTorchJob.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["src/train.py", "--config_path=configs/training_config_zero2.yaml"]
