# pod-base: general-purpose base (CUDA 12.8 + Python 3.10 + torch), built in two stages.
# CUDA and torch are build arguments; CI builds three variants:
#   pod-base:cu128           CUDA 12.8, torch 2.7.1 / torchvision 0.22.1 (pod-lyra2)
#   pod-base:cu128-torch280  CUDA 12.8, torch 2.8.0 / torchvision 0.23.0 (pod-comfy:cu128, pod-train)
#   pod-base:cu130           CUDA 13.0, torch 2.9.1 / torchvision 0.24.1 (pod-comfy:cu130; needs a
#                            driver for CUDA 13.0, 580 or later. torch 2.8 has no cu130 build)
# A derived image uses the builder and runtime of the same variant.
#
#   builder  CUDA devel + compilers. Published as pod-base-builder so derived images can
#            compile their CUDA extensions on top of it (FROM ${BUILDER} AS builder).
#   runtime  No compilers and no nvcc. Published as pod-base.
#            torch brings its own CUDA libraries (cuDNN, cuBLAS, NCCL, ...) as pip nvidia-*
#            wheels, so the runtime stage starts from the CUDA *base* image (cudart only,
#            0.1 GB compressed) instead of cudnn-runtime (2.86 GB, a second copy of the same
#            libraries). Every CUDA extension is compiled in a builder. gcc and Python.h stay:
#            triton compiles a small C helper at run time (torch.compile, bitsandbytes, ...).
ARG BUILDER_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.8.1-base-ubuntu22.04

FROM ${BUILDER_IMAGE} AS builder
ARG TORCH=2.7.1
ARG TORCHVISION=0.22.1
# PyTorch's wheel index for the CUDA build (cu128, cu130); its +cuXYZ version sorts above PyPI's
ARG TORCH_INDEX=cu128

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
    && python -m pip install "torch==${TORCH}" "torchvision==${TORCHVISION}" \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
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

# Only what running needs (RunPod needs bash, python3, base64, tar and setsid; all present).
# gcc, libc6-dev and python3.10-dev: triton's run-time compile of its C helper
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev gcc libc6-dev \
        git git-lfs curl ca-certificates ffmpeg libgl1 libglib2.0-0 util-linux \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# CPU-only check: torch imports, and a C file including Python.h compiles (what triton does)
RUN gcc --version | head -n 1 \
    && printf '#include <Python.h>\nint main(void) { return 0; }\n' > /tmp/t.c \
    && gcc $(python3.10-config --includes) /tmp/t.c -o /tmp/t && rm -f /tmp/t /tmp/t.c \
    && python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, 'CUDA', torch.version.cuda)" \
    && find /usr/local/lib/python3.10/dist-packages -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /workspace
CMD ["sleep", "infinity"]
