#!/usr/bin/env bash
# Run by hand inside the pod: download weights into /workspace and check imports and the CLI.
#   bash /opt/start.sh
# The weights are under NVIDIA's research & development license (see the model card at https://huggingface.co/nvidia/Lyra-2.0).
set -euo pipefail

CKPT_ROOT=/workspace/lyra-ckpt
LYRA_DIR=/opt/lyra/Lyra-2

# 1. Weights: checkpoints/* is about 97 GB (attach a RunPod volume at /workspace to keep them for next time)
if [ ! -d "$CKPT_ROOT/checkpoints/model" ]; then
  echo "== downloading nvidia/Lyra-2.0 checkpoints (~97 GB) to $CKPT_ROOT"
  mkdir -p "$CKPT_ROOT"
  # Xet keeps its own chunk cache, which would store the weights twice on disk
  export HF_HUB_DISABLE_XET=1
  df -h /workspace
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir "$CKPT_ROOT"
  else
    hf download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir "$CKPT_ROOT"
  fi
  df -h /workspace
fi

# 2. Point the repo's checkpoints directory at them
if [ -e "$LYRA_DIR/checkpoints" ] && [ ! -L "$LYRA_DIR/checkpoints" ]; then
  echo "error: $LYRA_DIR/checkpoints is a real directory; move it away first" >&2
  exit 1
fi
ln -sfn "$CKPT_ROOT/checkpoints" "$LYRA_DIR/checkpoints"

cd "$LYRA_DIR"
export PYTHONPATH="$LYRA_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 3. Import check
echo "== import check"
python - <<'PY'
import torch, flash_attn, transformer_engine.pytorch, depth_anything_3.api, moge.model.v1
try:
    import vipe_ext  # vendored source build
    vipe_src = "vipe_ext (source)"
except ImportError:
    import vipe      # nvidia-vipe from pip
    vipe_src = "vipe (pip)"
print("torch:", torch.__version__, "| flash_attn:", flash_attn.__version__, "|", vipe_src)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("all imports OK")
PY

# 4. Inference script CLI
echo "== lyra2_zoomgs_inference --help"
python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
echo "== OK"
