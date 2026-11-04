# pod-comfy: ComfyUI at a fixed commit with its pip dependencies, in two stages.
#
#   builder  pod-base-builder (torch 2.8.0). ComfyUI's requirements.txt (minus torch, torchvision and
#            torchaudio, with torch / torchvision pinned to the base's) and the custom nodes' pip
#            dependencies (requirements-extra.txt) go into the venv /opt/venv
#            (--system-site-packages, so torch comes from the base).
#   runtime  pod-base (no nvcc). Only /opt/venv and /opt/ComfyUI are copied over.
#
# Not in the image: custom node code (the user sends it at start) and models (downloaded into
# /workspace). /opt/ComfyUI is meant to stay read-only: copy it to /workspace before writing to it.
ARG BUILDER=pod-base-builder:cu128-torch280
ARG BASE=pod-base:cu128-torch280

FROM ${BUILDER} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Users of the image assume this commit; change both together
ARG COMFY_REF=19ae172bff6c9e4c738e7f2e00fe087f5b5b46ab

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY requirements-extra.txt /opt/pod/
RUN mkdir -p /opt/ComfyUI \
    && curl -fsSL "https://github.com/Comfy-Org/ComfyUI/archive/${COMFY_REF}.tar.gz" \
        | tar -xz -C /opt/ComfyUI --strip-components=1 \
    && /usr/bin/python3 -m pip freeze | grep -E '^(torch|torchvision)==' > /opt/pod/constraints-torch.txt \
    && python -m venv --system-site-packages /opt/venv \
    && grep -v -E '^(torch|torchvision|torchaudio)([^A-Za-z0-9_.-]|$)' /opt/ComfyUI/requirements.txt \
        > /opt/pod/requirements-comfy.txt \
    && pip install -c /opt/pod/constraints-torch.txt \
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

# CPU-only checks (no GPU and no nvcc here): triton's compiler needs are present, the dependencies
# import, and ComfyUI loads every built-in node (comfy_extras, comfy_api_nodes) without IMPORT FAILED.
# The quick test runs in a copy, as users do (/opt/ComfyUI -> /workspace/ComfyUI).
RUN gcc --version | head -n 1 && test -f /usr/include/python3.10/Python.h && ffmpeg -version | head -n 1 \
    && cd /opt/ComfyUI \
    && python -c "import torch, torchvision, transformers, safetensors, av, kornia, spandrel, torchsde, timm; print(torch.__version__)" \
    && cp -a /opt/ComfyUI /tmp/comfy-ci && cd /tmp/comfy-ci \
    && python main.py --cpu --quick-test-for-ci 2>&1 | tee /tmp/comfy-ci.log | tail -n 5 \
    && if grep -E 'IMPORT FAILED|Cannot import' /tmp/comfy-ci.log; then exit 1; fi \
    && cd / && rm -rf /tmp/comfy-ci /tmp/comfy-ci.log /root/.cache \
    && find /opt/venv /opt/ComfyUI -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /workspace
CMD ["sleep", "infinity"]
