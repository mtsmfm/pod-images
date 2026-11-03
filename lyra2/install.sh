#!/usr/bin/env bash
# Install Lyra 2.0's dependencies into the venv /opt/venv, one step per Dockerfile RUN.
# Shared by lyra2/Dockerfile and lyra2-rp/Dockerfile (which differ only in the base image).
#
#   install.sh venv        create /opt/venv (--system-site-packages: torch comes from the base) and
#                          install Lyra's requirements.txt, requirements-extra.txt and MoGe
#   install.sh te          Transformer Engine
#   install.sh flash-attn  flash-attn (prebuilt wheel for this Python)
#   install.sh vipe        ViPE, with its CUDA extension compiled now
#   install.sh da3         Depth Anything 3 and gsplat, with gsplat's CUDA extension compiled now
#   install.sh finish      fail on unexpected `pip check` lines, drop bytecode
#   install.sh link        (runtime stage) symlink cuda_runtime as cudart for Transformer Engine
#
# Environment: LYRA_DIR (default /opt/lyra/Lyra-2), REQUIREMENTS_DROP (extended regex of package
# names to leave out of Lyra's requirements.txt), and the ARGs MOGE_REF / GSPLAT_REF.
set -euo pipefail

POD=/opt/pod
LYRA_DIR=${LYRA_DIR:-/opt/lyra/Lyra-2}
# Not needed for inference: opencv-python (opencv-python-headless is also listed), tensorboard
REQUIREMENTS_DROP=${REQUIREMENTS_DROP:-opencv-python|tensorboard}

# site-packages of the base image's Python, where torch and its nvidia-* wheels live
system_site() { python -c 'import os, torch; print(os.path.dirname(os.path.dirname(torch.__file__)))'; }

# Headers and libraries of the pip nvidia-* wheels for building without a GPU (INSTALL.md step 4)
build_env() {
  local site; site=$(system_site)
  export CPATH="/usr/local/cuda/include:$site/nvidia/cudnn/include:$site/nvidia/nccl/include${CPATH:+:$CPATH}"
  export LD_LIBRARY_PATH="$site/torch/lib:$site/nvidia/cuda_runtime/lib:$site/nvidia/cudnn/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

case "${1:?usage: install.sh <step>}" in
  venv)
    python -m venv --system-site-packages /opt/venv
    cd "$LYRA_DIR"
    # numpy: Lyra's tested environment ends with 1.26 (Depth Anything 3 requires numpy<2)
    sed -E -e "/^(${REQUIREMENTS_DROP})==/d" -e 's/^numpy>=2.*/numpy==1.26.4/' requirements.txt > "$POD/requirements-lyra.txt"
    sed -E 's/\[[^]]*\]//' "$POD/requirements-lyra.txt" > "$POD/constraints.txt"   # constraints take no extras
    pip install --no-deps -r "$POD/requirements-lyra.txt"
    pip install -c "$POD/constraints.txt" -r "$POD/requirements-extra.txt"
    pip install --no-deps "git+https://github.com/microsoft/MoGe.git@${MOGE_REF:?}"
    ;;
  te)
    build_env
    # Pinned to 2.13.0, the latest release when Lyra-2/INSTALL.md was written (2026-04).
    # Later releases depend on transformer_engine_cu13 (2.14+) and need cuDNN >= 9.12 headers (2.20+).
    pip install --no-build-isolation -c "$POD/constraints.txt" "transformer_engine[pytorch]==2.13.0"
    ;;
  flash-attn)
    # Upstream builds 2.6.3 from source (~30 min). Use the official prebuilt 2.8.3.post2 wheel instead;
    # if it causes problems, build from source:
    #   pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
    py=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.7cxx11abiTRUE-${py}-${py}-linux_x86_64.whl"
    ;;
  vipe)
    build_env
    cd "$LYRA_DIR"
    # Vendored source, so `import vipe` finds vipe_ext and does not JIT-compile at run time.
    # nvidia-vipe on PyPI is sdist-only (it compiles too); the vendored source avoids version skew:
    #   pip install --no-build-isolation --no-deps nvidia-vipe
    USE_SYSTEM_EIGEN=1 pip install --no-build-isolation --no-deps -e lyra_2/_src/inference/vipe
    python -c "import torch, vipe_ext"
    ;;
  da3)
    build_env
    cd "$LYRA_DIR"
    pip install --no-build-isolation --no-deps -e lyra_2/_src/inference/depth_anything_3
    # gsplat (DA3's [gs] extra) falls back to a JIT compile on first import when gsplat.csrc is missing
    pip install --no-build-isolation --no-deps "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_REF:?}"
    python -c "import torch; from gsplat import csrc"
    ;;
  finish)
    pip check > "$POD/pip-check.txt" 2>&1 || true
    grep -v -e '^#' -e '^$' "$POD/pip-check-allow.txt" > "$POD/pip-check-allow.re"
    if grep -v -E -f "$POD/pip-check-allow.re" "$POD/pip-check.txt"; then
      echo "unexpected pip check output (above); expected lines go in pip-check-allow.txt" >&2
      exit 1
    fi
    find /opt/venv /opt/lyra -name __pycache__ -prune -exec rm -rf {} +
    find /opt/venv /opt/lyra -name '*.pyc' -delete
    ;;
  link)
    site=$(system_site)
    if [ ! -e "$site/nvidia/cudart" ]; then ln -s "$site/nvidia/cuda_runtime" "$site/nvidia/cudart"; fi
    ;;
  *)
    echo "unknown step: $1" >&2; exit 1 ;;
esac
