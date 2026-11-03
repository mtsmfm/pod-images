# pod-lyra2: inference for Lyra 2.0 (Lyra-2 in nv-tlabs/lyra)
# Lyra-2/INSTALL.md (conda + CUDA 12.8 + torch 2.7.1) translated onto pod-base, in two stages:
#
#   builder  pod-base-builder (CUDA devel + nvcc). Everything Lyra adds goes into the venv
#            /opt/venv (--system-site-packages, so torch comes from the base), and every CUDA
#            extension (Transformer Engine, ViPE, gsplat) is compiled here. Steps: install.sh.
#   runtime  pod-base (no nvcc). Only /opt/venv and /opt/lyra are copied over, so the image is
#            pod-base's layers plus Lyra's own. A final CPU-only import check fails the build
#            when a dependency is missing or something would need a JIT compile.
#
# lyra2-rp/Dockerfile builds the same on RunPod's PyTorch image; keep the two in step.
# Weights are not baked in (start.sh downloads them into /workspace).
ARG BUILDER=pod-base-builder:cu128
ARG BASE=pod-base:cu128

FROM ${BUILDER} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG LYRA_REF=9fffc9adc37004091ecf26ef03abfb3abdf4d59a
ARG MOGE_REF=74fbce054ebed49800de42d0ad0e83495065719a
ARG GSPLAT_REF=0b4dddf04cb687367602c01196913cde6a743d70

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeigen3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Source and submodules (only the two needed for inference; skip GUI deps such as OpenXR and DLSS)
RUN git init -q /opt/lyra && cd /opt/lyra \
    && git remote add origin https://github.com/nv-tlabs/lyra.git \
    && git fetch -q --depth 1 origin "${LYRA_REF}" && git checkout -q FETCH_HEAD \
    && git submodule update --init --depth 1 \
        Lyra-2/lyra_2/_src/inference/depth_anything_3 \
        Lyra-2/lyra_2/_src/inference/vipe

# Build without a GPU for A100, A6000/3090, 4090/L40S and H100. MAX_JOBS=2: with 4 parallel jobs, the
# ViPE compile ran the 16 GB runner out of memory
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    NVTE_CUDA_ARCHS="80;86;89;90" \
    MAX_JOBS=2

COPY requirements-extra.txt pip-check-allow.txt install.sh /opt/pod/
RUN /opt/pod/install.sh venv
RUN /opt/pod/install.sh te
RUN /opt/pod/install.sh flash-attn
RUN /opt/pod/install.sh vipe
RUN /opt/pod/install.sh da3
RUN /opt/pod/install.sh finish


FROM ${BASE} AS runtime

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/lyra /opt/lyra
COPY --from=builder /opt/pod /opt/pod
COPY start.sh check_imports.py /opt/

# LD_LIBRARY_PATH: what INSTALL.md asks to keep in the shell profile
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/lyra/Lyra-2 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# CPU-only checks (no GPU and no nvcc here): every dependency is present, nothing needs a JIT compile,
# and the inference CLIs reach argparse. flash_attn / transformer_engine need a GPU: /opt/start.sh checks them.
RUN /opt/pod/install.sh link \
    && chmod +x /opt/start.sh \
    && cd /opt/lyra/Lyra-2 \
    && python /opt/check_imports.py \
    && python /opt/check_imports.py --help-of lyra_2._src.inference.lyra2_custom_traj_inference > /dev/null \
    && python /opt/check_imports.py --help-of lyra_2._src.inference.lyra2_zoomgs_inference > /dev/null \
    && python /opt/check_imports.py --help-of lyra_2._src.inference.vipe_da3_gs_recon > /dev/null \
    && find /opt/venv /opt/lyra -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /workspace
CMD ["sleep", "infinity"]
