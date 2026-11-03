# pod-base: general-purpose base (CUDA 12.8 + Python 3.10 + torch 2.7.1), built in two stages.
#
#   builder  CUDA devel + compilers. Published as pod-base-builder so derived images can
#            compile their CUDA extensions on top of it (FROM ${BUILDER} AS builder).
#   runtime  No compilers and no nvcc. Published as pod-base.
#            torch brings its own CUDA libraries (cuDNN, cuBLAS, NCCL, ...) as pip nvidia-*
#            wheels, so the runtime stage starts from the CUDA *base* image (cudart only,
#            0.1 GB compressed) instead of cudnn-runtime (2.86 GB, a second copy of the same
#            libraries). Every CUDA extension is compiled in a builder.
ARG BUILDER_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.8.1-base-ubuntu22.04

FROM ${BUILDER_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/workspace/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget ca-certificates util-linux \
        build-essential cmake ninja-build \
        python3.10 python3.10-dev python3.10-venv python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.7.1 torchvision==0.22.1 \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install huggingface_hub safetensors numpy pillow \
    && find /usr/local/lib/python3.10/dist-packages -name __pycache__ -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/dist-packages -name '*.pyc' -delete

WORKDIR /workspace
CMD ["sleep", "infinity"]


FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf

# Only what running needs (RunPod needs bash, python3, base64, tar and setsid; all present)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv \
        git git-lfs curl ca-certificates ffmpeg libgl1 libglib2.0-0 util-linux \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /workspace
CMD ["sleep", "infinity"]
