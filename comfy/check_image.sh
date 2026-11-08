#!/usr/bin/env bash
# The runtime stage's checks, run in a pulled pod-comfy image (CI runs it in the zstd tag; not copied
# into the image). Mount comfy/ at /tmp/ctx:
#   docker run --rm -v $PWD/comfy:/tmp/ctx:ro IMAGE bash /tmp/ctx/check_image.sh TORCH_CUDA
# The expected ComfyUI version is the Dockerfile's COMFY_VERSION.
set -euo pipefail
expected=$(sed -nE 's/^ARG COMFY_VERSION=(.*)/\1/p' /tmp/ctx/Dockerfile)
python /opt/pod/check_cuda.py "$1"
cd /opt/ComfyUI
version=$(python -c "import comfyui_version as v; print(v.__version__)")
echo "ComfyUI ${version} (expected ${expected})"
[ "$version" = "$expected" ]
cp -a /opt/ComfyUI /tmp/comfy-ci && cd /tmp/comfy-ci
python main.py --cpu --quick-test-for-ci 2>&1 | tee /tmp/comfy-ci.log | tail -n 5
if grep -E 'IMPORT FAILED|Cannot import' /tmp/comfy-ci.log; then exit 1; fi
echo "quick test: exit 0, no IMPORT FAILED"
python /opt/pod/check_nodes.py /tmp/comfy-ci /opt/pod/required-nodes.txt
pip check
