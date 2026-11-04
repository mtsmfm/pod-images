# pod-train: ai-toolkit and sd-scripts at fixed commits with their pip dependencies, in two stages.
#
#   builder  pod-base-builder (torch 2.8.0). Both apps' requirements go into one venv /opt/venv
#            (--system-site-packages, so torch comes from the base); where they pin different
#            versions (diffusers, transformers, accelerate, ...) ai-toolkit's win. Steps: install.sh.
#   runtime  pod-base (no nvcc). Only /opt/venv, /opt/ai-toolkit and /opt/sd-scripts are copied over.
#
# Not in the image: ai-toolkit's web UI (ui/) and models (downloaded into /workspace).
# /opt/<app> is meant to stay read-only: copy it to /workspace before writing to it.
ARG BUILDER=pod-base-builder:cu128-torch280
ARG BASE=pod-base:cu128-torch280

FROM ${BUILDER} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Users of the image assume these commits; change both together
ARG AITK_REF=04733873cf1da8060fce01316ec45daedbd4a918
ARG SD_SCRIPTS_REF=068bcd7ffe76b2cd5012fb680a2c94e295398bbc
# Built for the base's torch 2.8
ARG TORCHAUDIO=2.8.0
ARG TORCHCODEC=0.7.0
ARG FLA_VERSION=0.5.2
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY install.sh check_deps.py optional-imports.txt /opt/pod/
RUN /opt/pod/install.sh fetch
RUN /opt/pod/install.sh ai-toolkit
RUN /opt/pod/install.sh sd-scripts
RUN /opt/pod/install.sh finish


FROM ${BASE} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# libgl1 and libglib2.0-0 (OpenCV), and FFmpeg's libraries (torchcodec), come with pod-base
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/ai-toolkit /opt/ai-toolkit
COPY --from=builder /opt/sd-scripts /opt/sd-scripts
COPY --from=builder /opt/pod /opt/pod

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    AITK_HOME=/opt/ai-toolkit \
    SD_SCRIPTS_HOME=/opt/sd-scripts \
    HF_HOME=/workspace/hf

# CPU-only checks (no GPU and no nvcc here): triton's compiler needs are present, the main modules
# import, the CLIs reach argparse, and every module imported anywhere in either app's source is
# installed unless optional-imports.txt lists it.
RUN gcc --version | head -n 1 && test -f /usr/include/python3.10/Python.h \
    && cd /opt/ai-toolkit \
    && python -c "import toolkit, lycoris, optimum.quanto, peft, diffusers, accelerate, transformers, bitsandbytes, torchaudio; from torchcodec.decoders import AudioDecoder; import torch; print(torch.__version__)" \
    && python run.py --help > /dev/null \
    && python /opt/pod/check_deps.py --allow-file /opt/pod/optional-imports.txt \
        --exclude ui --exclude testing --exclude notebooks /opt/ai-toolkit \
    && cd /opt/sd-scripts \
    && python anima_train_network.py --help > /dev/null \
    && python /opt/pod/check_deps.py --allow-file /opt/pod/optional-imports.txt \
        --exclude tests --exclude bitsandbytes_windows /opt/sd-scripts \
    && cd / && rm -rf /root/.cache \
    && find /opt/venv /opt/ai-toolkit /opt/sd-scripts -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /workspace
CMD ["sleep", "infinity"]
