# pod-lyra2: inference for Lyra 2.0 (Lyra-2 in nv-tlabs/lyra)
# Lyra-2/INSTALL.md (conda + CUDA 12.8 + torch 2.7.1) translated onto pod-base.
# Weights are not baked in (start.sh downloads them into /workspace).
ARG BASE=pod-base:cu128
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeigen3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Source and submodules (only the two needed for inference; skip GUI deps such as OpenXR and DLSS)
RUN git clone --depth 1 https://github.com/nv-tlabs/lyra.git /opt/lyra \
    && cd /opt/lyra \
    && git submodule update --init --depth 1 \
        Lyra-2/lyra_2/_src/inference/depth_anything_3 \
        Lyra-2/lyra_2/_src/inference/vipe

WORKDIR /opt/lyra/Lyra-2

# For building without a GPU: pin target archs and expose headers/libs from the pip nvidia-* wheels (INSTALL.md step 4)
ENV NVTE_CUDA_ARCHS="86;89;90" \
    MAX_JOBS=4 \
    CPATH=/usr/local/cuda/include:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/include:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/include \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Python dependencies
RUN pip install --no-deps -r requirements.txt \
    && pip install "git+https://github.com/microsoft/MoGe.git"

# Transformer Engine (+ symlink cuda_runtime as cudart, as INSTALL.md does)
# Pinned to 2.13.0, the latest release when Lyra-2/INSTALL.md was written (2026-04). Later releases
# depend on transformer_engine_cu13 (2.14+) and need cuDNN >= 9.12 headers (2.20+), newer than this base.
RUN pip install --no-build-isolation "transformer_engine[pytorch]==2.13.0" \
    && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && if [ ! -e "$SITE/nvidia/cudart" ]; then ln -s "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"; fi

# flash-attn: upstream builds 2.6.3 from source (~30 min). Use the official prebuilt 2.8.3.post2 wheel instead.
# If it causes problems, switch back to the source build below.
RUN pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# RUN pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# ViPE: build the vendored source (Lyra imports vipe_ext).
# nvidia-vipe on PyPI is sdist-only (it compiles too), so the vendored source is the default to avoid version skew.
RUN USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e lyra_2/_src/inference/vipe
# RUN pip install --no-build-isolation nvidia-vipe

# Depth Anything 3 (with gsplat)
RUN pip install --no-build-isolation -e "lyra_2/_src/inference/depth_anything_3[gs]"

COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

ENV PYTHONPATH=/opt/lyra/Lyra-2
WORKDIR /workspace
CMD ["sleep", "infinity"]
