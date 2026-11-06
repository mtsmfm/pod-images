# pod-comfy: ComfyUI at a fixed commit with its pip dependencies, in two stages.
#
#   builder  pod-base-builder. ComfyUI's requirements.txt (minus torch, torchvision and torchaudio,
#            with torch / torchvision pinned to the base's) and the custom nodes' pip dependencies
#            (requirements-extra.txt) go into the venv /opt/venv (--system-site-packages, so torch
#            comes from the base).
#   runtime  pod-base (no nvcc). Only /opt/venv and /opt/ComfyUI are copied over.
#
# Two variants (tags), from the pod-base they are built on:
#   cu128  pod-base:cu128-torch280 (torch 2.8.0 cu128). Runs on any host whose driver supports
#          CUDA 12.8. comfy-kitchen is built from source with this CUDA: the PyPI wheel's kernels are
#          built with CUDA 13.0 and link cudart statically, so they need a CUDA 13 driver (580+).
#   cu130  pod-base:cu130 (torch 2.9.1 cu130). Needs a CUDA 13.0 driver (580+); ComfyUI then uses
#          comfy-kitchen's CUDA kernels (it turns them off with a cu128 torch). PyPI wheel as is.
#
# Not in the image: custom node code (the user sends it at start) and models (downloaded into
# /workspace). /opt/ComfyUI is meant to stay read-only: copy it to /workspace before writing to it.
ARG BUILDER=pod-base-builder:cu128-torch280
ARG BASE=pod-base:cu128-torch280

FROM ${BUILDER} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Users of the image assume this commit (v0.44.0); change it together with COMFY_VERSION below
ARG COMFY_REF=2c75dc1b379a59e0880619d029607af5169a62ef
# 7-day cooldown for what ComfyUI's requirements leave unpinned: pip only takes releases uploaded
# before this (7 days before COMFY_REF was last changed). cooldown-exempt.txt lists the exceptions,
# taken with the cutoff EXEMPT_UPLOADED_PRIOR_TO.
ARG UPLOADED_PRIOR_TO=2026-10-30T00:00:00Z
ARG EXEMPT_UPLOADED_PRIOR_TO=2026-11-04T00:00:00Z
# comfy-kitchen source build (CUDA 12 only). The default architectures add 100f / 120f, which need
# nvcc 12.9+; 120 covers the RTX 5090 / PRO 6000, and 86-real the 3090 / A6000 / A40
ARG COMFY_KITCHEN_CUDA_ARCHS="75-real;80-real;86-real;89;90a-real;120"

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY requirements-extra.txt constraints-build.txt cooldown-exempt.txt check_cuda.py check_nodes.py \
    required-nodes.txt /opt/pod/
RUN mkdir -p /opt/ComfyUI \
    && curl -fsSL "https://github.com/Comfy-Org/ComfyUI/archive/${COMFY_REF}.tar.gz" \
        | tar -xz -C /opt/ComfyUI --strip-components=1 \
    && /usr/bin/python3 -m pip freeze | grep -E '^(torch|torchvision)==' > /opt/pod/constraints-torch.txt \
    && cat /opt/pod/constraints-torch.txt \
    && python -m venv --system-site-packages /opt/venv \
    && python -m pip install --no-cache-dir pip==26.2.2

# comfy-kitchen: the version ComfyUI pins, built against the image's CUDA when torch is CUDA 12.x.
# Installed first, so the requirements below find comfy-kitchen==<pin> satisfied and keep it.
# Build tools (pyproject's build-system) are pinned by constraints-build.txt.
RUN cuda_major=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])') \
    && version=$(sed -nE 's/^comfy-kitchen==([0-9.]+).*/\1/p' /opt/ComfyUI/requirements.txt) \
    && echo "torch CUDA major ${cuda_major}, comfy-kitchen ${version}" \
    && if [ "$cuda_major" = 12 ]; then \
        git clone -q --recursive --depth 1 --shallow-submodules --branch "v${version}" \
            https://github.com/Comfy-Org/comfy-kitchen.git /tmp/comfy-kitchen \
        && PIP_CONSTRAINT=/opt/pod/constraints-build.txt COMFY_CUDA_ARCHS="${COMFY_KITCHEN_CUDA_ARCHS}" \
            pip wheel --no-deps -w /tmp/wheels /tmp/comfy-kitchen \
        && pip install --no-deps /tmp/wheels/comfy_kitchen-*.whl \
        && rm -rf /tmp/comfy-kitchen /tmp/wheels; \
    fi

RUN grep -v -E '^(torch|torchvision|torchaudio)([^A-Za-z0-9_.-]|$)' /opt/ComfyUI/requirements.txt \
        > /opt/pod/requirements-comfy.txt \
    && if grep -q '^[A-Za-z]' /opt/pod/cooldown-exempt.txt; then \
        pip install --no-deps --uploaded-prior-to "${EXEMPT_UPLOADED_PRIOR_TO}" -r /opt/pod/cooldown-exempt.txt; \
    fi \
    && pip install --uploaded-prior-to "${UPLOADED_PRIOR_TO}" -c /opt/pod/constraints-torch.txt \
        -r /opt/pod/requirements-comfy.txt -r /opt/pod/requirements-extra.txt \
    && pip check \
    && find /opt/venv /opt/ComfyUI -name __pycache__ -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete


FROM ${BASE} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# libgl1, libglib2.0-0 and ffmpeg (for checking CreateVideo / SaveVideo output) come with pod-base
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/ComfyUI /opt/ComfyUI
COPY --from=builder /opt/pod /opt/pod

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    COMFY_HOME=/opt/ComfyUI

# torch's CUDA version (12.8 or 13.0) and ComfyUI's version (COMFY_REF's), checked below
ARG TORCH_CUDA=12.8
ARG COMFY_VERSION=0.44.0

# CPU-only checks (no GPU and no nvcc here): triton's compiler needs are present, torch is the expected
# CUDA build and no CUDA code needs a newer driver than it (check_cuda.py), the dependencies import,
# ComfyUI is COMFY_VERSION, it loads every built-in node (comfy_extras, comfy_api_nodes) without
# IMPORT FAILED, and its server lists the nodes in required-nodes.txt in /object_info.
# The tests run in a copy, as users do (/opt/ComfyUI -> /workspace/ComfyUI).
RUN gcc --version | head -n 1 && test -f /usr/include/python3.10/Python.h && ffmpeg -version | head -n 1 \
    && python /opt/pod/check_cuda.py "${TORCH_CUDA}" \
    && cd /opt/ComfyUI \
    && version=$(python -c "import comfyui_version as v; print(v.__version__)") \
    && echo "ComfyUI ${version} (expected ${COMFY_VERSION})" && [ "$version" = "${COMFY_VERSION}" ] \
    && python -c "import torch, torchvision, transformers, safetensors, av, kornia, spandrel, torchsde, timm; print(torch.__version__)" \
    && cp -a /opt/ComfyUI /tmp/comfy-ci && cd /tmp/comfy-ci \
    && python main.py --cpu --quick-test-for-ci 2>&1 | tee /tmp/comfy-ci.log | tail -n 5 \
    && if grep -E 'IMPORT FAILED|Cannot import' /tmp/comfy-ci.log; then exit 1; fi \
    && python /opt/pod/check_nodes.py /tmp/comfy-ci /opt/pod/required-nodes.txt \
    && pip check \
    && cd / && rm -rf /tmp/comfy-ci /tmp/comfy-ci.log /tmp/comfy-server.log /root/.cache \
    && find /opt/venv /opt/ComfyUI -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /workspace
CMD ["sleep", "infinity"]
