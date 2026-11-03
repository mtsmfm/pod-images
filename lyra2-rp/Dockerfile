# pod-lyra2-rp: Lyra 2.0 inference on RunPod's official PyTorch image (an experiment).
# Same install as lyra2/ (its install.sh, requirements-extra.txt, check_imports.py and start.sh,
# taken from the named build context `lyra2`), but on runpod/pytorch instead of pod-base, so that a
# RunPod host that already has that image only pulls Lyra's own layers. Whether that helps is to be
# measured; the image itself is larger than pod-lyra2 (the base alone is 10.5 GB compressed).
#
# Conditions for the base (check them again when changing the tag; the first RUN enforces 1 and 2):
#   1. nvcc and torch's wheel share the CUDA major version (CUDA extensions do not build otherwise)
#   2. torch 2.7.1 (Lyra's pin)
#   3. Python 3.10-3.12, with a matching prebuilt flash-attn wheel (cp310/cp312, cu12torch2.7cxx11abiTRUE)
# 1.6.0-cu1281-torch271-ubuntu2204 (2026-11-03): CUDA 12.8.1 devel, torch 2.7.1 from the cu128 index,
# Python 3.12. The cu1300 tags pair nvcc 13.0 with the same cu128 torch, which fails condition 1.
ARG BASE=runpod/pytorch:1.6.0-cu1281-torch271-ubuntu2204

FROM ${BASE} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG LYRA_REF=9fffc9adc37004091ecf26ef03abfb3abdf4d59a
ARG MOGE_REF=74fbce054ebed49800de42d0ad0e83495065719a
ARG GSPLAT_REF=0b4dddf04cb687367602c01196913cde6a743d70

RUN nvcc --version | tail -1 && python --version \
    && python -c "import re, subprocess, torch; \
nvcc = re.search(r'release (\d+)\.', subprocess.check_output(['nvcc', '--version'], text=True)).group(1); \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| nvcc major', nvcc); \
assert torch.__version__.startswith('2.7.1'), 'torch must be 2.7.1'; \
assert torch.version.cuda.split('.')[0] == nvcc, 'nvcc and torch CUDA major versions differ'"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeigen3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git init -q /opt/lyra && cd /opt/lyra \
    && git remote add origin https://github.com/nv-tlabs/lyra.git \
    && git fetch -q --depth 1 origin "${LYRA_REF}" && git checkout -q FETCH_HEAD \
    && git submodule update --init --depth 1 \
        Lyra-2/lyra_2/_src/inference/depth_anything_3 \
        Lyra-2/lyra_2/_src/inference/vipe

# tensorstore==0.1.45 has no Python 3.12 wheel; Lyra's code does not use it (only megatron-core lists it)
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    NVTE_CUDA_ARCHS="80;86;89;90" \
    MAX_JOBS=2 \
    REQUIREMENTS_DROP="opencv-python|tensorboard|tensorstore"

COPY --from=lyra2 requirements-extra.txt pip-check-allow.txt install.sh /opt/pod/
COPY pip-check-allow.txt /opt/pod/pip-check-allow-rp.txt
RUN cat /opt/pod/pip-check-allow-rp.txt >> /opt/pod/pip-check-allow.txt
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
COPY --from=lyra2 start.sh check_imports.py /opt/

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/lyra/Lyra-2 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64

# Same CPU-only checks as lyra2 (nvcc exists in this base, but the extensions are already compiled)
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
