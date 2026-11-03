# pod-base: general-purpose base (CUDA 12.8 + cuDNN + Python 3.10 + torch 2.7.1)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/workspace/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget ca-certificates ffmpeg libgl1 libglib2.0-0 util-linux \
        build-essential cmake ninja-build \
        python3.10 python3.10-dev python3.10-venv python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.7.1 torchvision==0.22.1 \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install huggingface_hub safetensors numpy pillow

WORKDIR /workspace
CMD ["sleep", "infinity"]
