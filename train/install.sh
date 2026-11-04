#!/usr/bin/env bash
# Install ai-toolkit and sd-scripts into the venv /opt/venv, one step per Dockerfile RUN.
#
#   install.sh fetch       download both at AITK_REF / SD_SCRIPTS_REF into /opt, create the venv
#                          (--system-site-packages: torch comes from the base)
#   install.sh ai-toolkit  ai-toolkit's requirements.txt, torchaudio, and the accelerators its own
#                          installer adds (flash-linear-attention, flash-attn)
#   install.sh sd-scripts  sd-scripts' requirements.txt, keeping ai-toolkit's versions where they differ
#   install.sh finish      pip check, drop bytecode
#
# torch / torchvision stay the base's (constraints from its pip freeze); torchaudio and torchcodec
# are pinned to the releases built for that torch (TORCHAUDIO, TORCHCODEC).
set -euo pipefail

POD=${POD:-/opt/pod}
AITK=${AITK:-/opt/ai-toolkit}
SD=${SD:-/opt/sd-scripts}
# sd-scripts pins these too; ai-toolkit's versions win (anima_train_network.py --help is checked with them)
SD_DROP='accelerate|transformers|diffusers|huggingface-hub|huggingface_hub|safetensors'

fetch() { # fetch <github repo> <commit> <dir>
  mkdir -p "$3"
  curl -fsSL "https://github.com/$1/archive/$2.tar.gz" | tar -xz -C "$3" --strip-components=1
}

case "${1:?usage: install.sh <step>}" in
  fetch)
    fetch ostris/ai-toolkit "${AITK_REF:?}" "$AITK"
    rm -rf "$AITK/ui"   # the web UI (Node.js) is not used; run.py is
    fetch kohya-ss/sd-scripts "${SD_SCRIPTS_REF:?}" "$SD"
    { /usr/bin/python3 -m pip freeze | grep -E '^(torch|torchvision)=='
      echo "torchaudio==${TORCHAUDIO:?}"; } > "$POD/constraints-torch.txt"
    cat "$POD/constraints-torch.txt"
    python -m venv --system-site-packages /opt/venv
    ;;
  ai-toolkit)
    # torchaudio: toolkit/config_modules.py imports it at the top. Same CUDA build as the base's torch.
    pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 "torchaudio==${TORCHAUDIO:?}"
    # torchcodec: ai-toolkit pins the release for its own torch (2.13); use the one built for ours
    sed -E "s/^torchcodec==.*/torchcodec==${TORCHCODEC:?}/" "$AITK/requirements_base.txt" > "$POD/requirements-aitk-base.txt"
    { echo "-r requirements-aitk-base.txt"; grep -v '^-r' "$AITK/requirements.txt"; } > "$POD/requirements-aitk.txt"
    pip install -c "$POD/constraints-torch.txt" -r "$POD/requirements-aitk.txt"
    # What ai-toolkit's installer (manager/spec.py) adds on Linux + CUDA: flash-linear-attention (pure
    # Python, triton; its code is in fla-core) and a prebuilt flash-attn. NATTEN has no build for this
    # torch and is left out.
    pip install -c "$POD/constraints-torch.txt" "flash-linear-attention==${FLA_VERSION:?}"
    pip install --no-deps "${FLASH_ATTN_WHEEL:?}"
    ;;
  sd-scripts)
    # Drop the pins ai-toolkit decides, comments and `-e .` (the scripts import library/ from their
    # own directory); keep ai-toolkit's exact pins as constraints so nothing of it changes
    sed -E -e 's/#.*//' -e "/^(${SD_DROP})([^A-Za-z0-9_.-]|$)/d" -e '/^-e /d' -e '/^[[:space:]]*$/d' \
      "$SD/requirements.txt" > "$POD/requirements-sd.txt"
    { cat "$POD/constraints-torch.txt"
      grep -h -E '^[A-Za-z0-9_.-]+==' "$POD/requirements-aitk-base.txt" "$AITK/requirements.txt"; } \
      > "$POD/constraints-sd.txt"
    pip install -c "$POD/constraints-sd.txt" -r "$POD/requirements-sd.txt"
    ;;
  finish)
    pip check
    find /opt/venv "$AITK" "$SD" -name __pycache__ -prune -exec rm -rf {} +
    find /opt/venv -name '*.pyc' -delete
    ;;
  *)
    echo "unknown step: $1" >&2; exit 1 ;;
esac
