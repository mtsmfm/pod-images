# pod-lyra2: inference for Lyra 2.0 (Lyra-2 in nv-tlabs/lyra)
# Lyra-2/INSTALL.md (conda + CUDA 12.8 + torch 2.7.1) translated onto pod-base, in two stages:
#
#   builder  pod-base-builder (CUDA devel + nvcc). Everything Lyra adds goes into the venv
#            /opt/venv (--system-site-packages, so torch comes from the base), and every CUDA
#            extension (Transformer Engine, ViPE, gsplat) is compiled here.
#   runtime  pod-base (no nvcc). Only /opt/venv and /opt/lyra are copied over, so the image is
#            pod-base's layers plus Lyra's own. A final CPU-only import check fails the build
#            when a dependency is missing or something would need a JIT compile.
#
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

WORKDIR /opt/lyra/Lyra-2

# Build without a GPU: pin target archs (A100, A6000/3090, 4090/L40S, H100) and expose headers/libs
# from the pip nvidia-* wheels (INSTALL.md step 4)
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    NVTE_CUDA_ARCHS="80;86;89;90" \
    MAX_JOBS=4 \
    CPATH=/usr/local/cuda/include:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/include:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/include \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY requirements-extra.txt pip-check-allow.txt /opt/pod/

# Python dependencies.
# Lyra's requirements.txt goes in with --no-deps (as upstream does), minus what inference does not use:
# opencv-python (opencv-python-headless is also listed), tensorboard. numpy is pinned to 1.26.4,
# the version Lyra's tested environment ends with (Depth Anything 3 requires numpy<2).
# Its pins then constrain everything else; requirements-extra.txt lists the deps --no-deps leaves out.
RUN python -m venv --system-site-packages /opt/venv \
    && sed -E -e '/^(opencv-python|tensorboard)==/d' -e 's/^numpy>=2.*/numpy==1.26.4/' requirements.txt > /opt/pod/requirements-lyra.txt \
    && sed -E 's/\[[^]]*\]//' /opt/pod/requirements-lyra.txt > /opt/pod/constraints.txt \
    && pip install --no-deps -r /opt/pod/requirements-lyra.txt \
    && pip install -c /opt/pod/constraints.txt -r /opt/pod/requirements-extra.txt \
    && pip install --no-deps "git+https://github.com/microsoft/MoGe.git@${MOGE_REF}"

# Transformer Engine. Pinned to 2.13.0, the latest release when Lyra-2/INSTALL.md was written (2026-04).
# Later releases depend on transformer_engine_cu13 (2.14+) and need cuDNN >= 9.12 headers (2.20+).
RUN pip install --no-build-isolation -c /opt/pod/constraints.txt "transformer_engine[pytorch]==2.13.0"

# flash-attn: upstream builds 2.6.3 from source (~30 min). Use the official prebuilt 2.8.3.post2 wheel instead.
# If it causes problems, switch back to the source build below.
RUN pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# RUN pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

# ViPE: build the vendored source; vipe_ext is compiled here, so `import vipe` does not JIT at run time.
# nvidia-vipe on PyPI is sdist-only (it compiles too), so the vendored source is the default to avoid version skew.
RUN USE_SYSTEM_EIGEN=1 pip install --no-build-isolation --no-deps -e lyra_2/_src/inference/vipe \
    && python -c "import torch, vipe_ext"
# RUN pip install --no-build-isolation --no-deps nvidia-vipe

# Depth Anything 3, and gsplat (its [gs] extra) with the CUDA extension compiled now:
# gsplat falls back to a JIT compile on first import when gsplat.csrc is missing.
RUN pip install --no-build-isolation --no-deps -e lyra_2/_src/inference/depth_anything_3 \
    && pip install --no-build-isolation --no-deps "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_REF}" \
    && python -c "import torch; from gsplat import csrc"

# Fail on unexpected `pip check` lines (expected ones are listed in pip-check-allow.txt), then drop bytecode
RUN pip check > /opt/pod/pip-check.txt 2>&1 || true \
    && grep -v -e '^#' -e '^$' /opt/pod/pip-check-allow.txt > /opt/pod/pip-check-allow.re \
    && if grep -v -E -f /opt/pod/pip-check-allow.re /opt/pod/pip-check.txt; then \
         echo "unexpected pip check output (see above)"; exit 1; fi \
    && find /opt/venv /opt/lyra -name __pycache__ -prune -exec rm -rf {} + \
    && find /opt/venv /opt/lyra -name '*.pyc' -delete


FROM ${BASE} AS runtime

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/lyra /opt/lyra
COPY --from=builder /opt/pod /opt/pod
COPY start.sh check_imports.py /opt/

# Symlink cuda_runtime as cudart for Transformer Engine (INSTALL.md step 5), and the
# LD_LIBRARY_PATH INSTALL.md asks to keep in the shell profile
RUN SITE=/usr/local/lib/python3.10/dist-packages \
    && if [ ! -e "$SITE/nvidia/cudart" ]; then ln -s "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"; fi \
    && chmod +x /opt/start.sh

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/lyra/Lyra-2 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# CPU-only checks (no GPU and no nvcc here): every dependency is present, nothing needs a JIT compile,
# and the inference CLIs reach argparse. flash_attn / transformer_engine need a GPU: /opt/start.sh checks them.
RUN cd /opt/lyra/Lyra-2 \
    && python /opt/check_imports.py \
    && python -m lyra_2._src.inference.lyra2_custom_traj_inference --help > /dev/null \
    && python -m lyra_2._src.inference.lyra2_zoomgs_inference --help > /dev/null \
    && python -m lyra_2._src.inference.vipe_da3_gs_recon --help > /dev/null \
    && find /opt/venv /opt/lyra -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /workspace
CMD ["sleep", "infinity"]
