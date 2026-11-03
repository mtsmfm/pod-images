# pod-images

Custom Docker images for RunPod (and local use), built by GitHub Actions and published to ghcr.io.
This repository holds image definitions only.

| Image | Contents |
| --- | --- |
| `ghcr.io/mtsmfm/pod-base:cu128` | General-purpose base: Ubuntu 22.04, Python 3.10, torch 2.7.1 / torchvision 0.22.1 (CUDA 12.8 libraries from the torch wheels), huggingface_hub. No compilers, no nvcc |
| `ghcr.io/mtsmfm/pod-base-builder:cu128` | The same plus CUDA 12.8.1 devel + cuDNN and compilers. Only used as the builder stage of derived images |
| `ghcr.io/mtsmfm/pod-lyra2:cu128` | Inference environment for [Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2) (flash-attn, Transformer Engine, ViPE, Depth Anything 3 + gsplat, MoGe) on pod-base |
| `ghcr.io/mtsmfm/pod-lyra2-rp:cu128` | Experiment: the same Lyra install on RunPod's `runpod/pytorch:1.6.0-cu1281-torch271-ubuntu2204` (Python 3.12). See below |

Each image also gets a pinned tag `cu128-<run_number>` (`:cu128` always points to the latest build; older builds stay reachable by their run tag).

### How the images are built

Every image has two stages. The **builder** stage (CUDA devel, nvcc, compilers) installs the Python
dependencies and compiles every CUDA extension for `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
(A100, A6000/3090, 4090/L40S, H100). The **runtime** stage, the one that is pushed, has no compilers
and no nvcc and only copies the results over:

- pod-base's runtime starts from `nvidia/cuda:12.8.1-base-ubuntu22.04` (cudart only, 0.1 GB) and gets
  torch's site-packages from the builder. torch brings its own CUDA libraries (cuDNN, cuBLAS, NCCL, ...)
  as pip `nvidia-*` wheels, so `cudnn-runtime` (2.86 GB compressed) would only add a second copy.
- pod-lyra2 installs everything Lyra adds into the venv `/opt/venv` (`--system-site-packages`, so
  torch comes from the base) and copies only `/opt/venv` and `/opt/lyra` onto pod-base. Its layers
  are pod-base's plus Lyra's own.
- Nothing compiles at run time: gsplat and ViPE would JIT-compile their extension on first import
  when it is missing, and the runtime stage has no nvcc to do it.

The workflow prints each image's compressed size (sum of the amd64 manifest's layers) in the run summary.

| Image | Compressed (2026-11-04) | Before the two-stage builds |
| --- | --- | --- |
| pod-base:cu128 | 4.20 GB | 9.99 GB |
| pod-base-builder:cu128 | 9.83 GB | — |
| pod-lyra2:cu128 | 5.61 GB (pod-base + 1.4 GB) | 12.05 GB |
| pod-lyra2-rp:cu128 | 11.90 GB (runpod/pytorch 10.5 GB + 1.4 GB) | — |

## Usage

### Building

- **Automatic**: a push to `main` that touches `base/**`, `lyra2/**`, `lyra2-rp/**` or `.github/` builds and pushes base (builder and runtime), then lyra2; lyra2-rp builds in parallel.
- **Manual**: Actions → `build` → *Run workflow*, and pick `all` / `base` / `lyra2` / `lyra2-rp` for `image`. With `lyra2` alone, base is skipped and lyra2 is built on the `pod-base-builder:cu128` / `pod-base:cu128` already on ghcr.io.
- **Local**: `./build.sh base <user>`, `./build.sh lyra2 <user>` (set `PUSH=1` to push as well). No GPU needed.

After the first push, make each package (pod-base, pod-base-builder, pod-lyra2, pod-lyra2-rp) **Public** on ghcr.io: *Package settings → Change visibility*. This is separate from the repository's visibility.

### Running Lyra 2.0

```bash
docker run --gpus all -it -v $PWD/workspace:/workspace ghcr.io/mtsmfm/pod-lyra2:cu128 bash /opt/start.sh
```

`/opt/start.sh`:

1. Downloads `checkpoints/*` from `nvidia/Lyra-2.0` on Hugging Face unless `/workspace/lyra-ckpt/checkpoints/model` exists (with `HF_HUB_DISABLE_XET=1`, so Xet's cache does not store the weights a second time; `df -h /workspace` before and after)
2. Symlinks `/opt/lyra/Lyra-2/checkpoints` to it
3. Checks the imports `torch, flash_attn, transformer_engine.pytorch, depth_anything_3.api, moge.model.v1` and `vipe_ext` (or `vipe`), and prints `torch.cuda.is_available()` and the GPU name
4. Runs `python -m lyra_2._src.inference.lyra2_zoomgs_inference --help`

**Disk**: `checkpoints/*` is about **97 GB** (recon 13 GB, text_encoder 11 GB, the main model in many 2.1 GB shards).
Give the container **220 GB or more** (image + 97 GB + working space). To use it again, put `/workspace` on a
RunPod volume: the second start skips the download.

The image build already checks, on CPU, that every module the inference code imports is installed
(`/opt/check_imports.py`, which also follows imports inside functions) and that the inference CLIs reach
argparse. `start.sh` checks what needs a GPU (flash_attn, Transformer Engine, CUDA).

For running inference, see the [Lyra-2 README](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2). The source is at `/opt/lyra/Lyra-2` (`PYTHONPATH` is set).

License: Lyra's code is Apache 2.0; the weights are under NVIDIA's research & development license (see the Hugging Face model card). The weights are not included in the image.

Differences from the upstream install guide:

- Installed on top of pod-base (system Python 3.10, CUDA 12.8 from the torch wheels) instead of conda
- Lyra, MoGe and gsplat are pinned to commits (`LYRA_REF`, `MOGE_REF`, `GSPLAT_REF` in `lyra2/Dockerfile`)
- Dependencies: Lyra's `requirements.txt` goes in with `--no-deps` as upstream does, without `opencv-python` (`opencv-python-headless` is also listed) and `tensorboard`. The deps `--no-deps` leaves out are listed in `lyra2/requirements-extra.txt` (including `addict`, which Depth Anything 3 imports but does not declare). Depth Anything 3, MoGe, ViPE and gsplat also go in with `--no-deps`, leaving out `open3d` (only DA3's benchmarks), `gradio` and `flex-gemm` (only MoGe's demo and v2/v3 models) and `opencv-python`. `boto3` stays: Lyra imports it at start-up
- numpy is pinned to 1.26.4: Lyra and MoGe ask for numpy>=2 but Depth Anything 3 for numpy<2, and Lyra's tested environment ends with 1.26 (DA3 is installed last). `pip check` must pass except for the lines in `lyra2/pip-check-allow.txt`
- Transformer Engine is pinned to 2.13.0 (upstream installs the latest; newer releases target CUDA 13 / cuDNN >= 9.12)
- flash-attn uses the official prebuilt 2.8.3.post2 wheel instead of building 2.6.3 from source. If it causes problems, switch to the source-build line in `lyra2/Dockerfile`
- ViPE is built from the vendored source (as upstream does). A commented-out line switches to `nvidia-vipe` from PyPI (which is also an sdist and compiles)

### pod-lyra2-rp (experiment)

RunPod hosts may already have RunPod's official images, so an image built on one would only pull
our own layers. Whether that is faster than pulling the slimmer pod-lyra2 is to be measured (pod
creation to runtime ready, a few times each); if it makes no difference, drop `lyra2-rp/`.
The base alone is 10.5 GB compressed, and the layers only help when the host has *the same digest*
(RunPod rebuilds its tags).

- Same install as lyra2 (`lyra2/install.sh`, `requirements-extra.txt`, `check_imports.py`, `start.sh`, via the build context `lyra2`)
- Conditions for the base, enforced by the first `RUN` for 1 and 2: (1) nvcc and torch's wheel share the CUDA major version, (2) torch 2.7.1, (3) Python 3.10-3.12 with a matching prebuilt flash-attn wheel. The `cu1300` tags pair nvcc 13.0 with a cu128 torch and fail (1)
- Python 3.12: `tensorstore==0.1.45` has no wheel for it and is left out (Lyra's code does not use it)

## What RunPod needs

The images contain nothing RunPod-specific.

- `bash`, `python3`, `base64`, `tar` and `setsid` (util-linux) are included. Startup logic is injected by overriding CMD with RunPod's `dockerStartCmd`; the image CMD is `sleep infinity`
- HTTP is exposed on the RunPod side with `ports: 8000/http`
- RunPod injects the pod ID and API key environment variables (used for auto-stopping pods). Nothing is needed in the image
- No RunPod SDK, runpodctl, SSH or Jupyter
- With public packages, RunPod needs no registry credentials. If a package is private, register it under Settings → Container Registry
- Attach a volume at `/workspace` to keep the weights (`/workspace/lyra-ckpt`) and the HF cache (`HF_HOME=/workspace/hf`) across runs

## Rules

- **One folder per purpose.** Derived images build in `FROM ${BUILDER} AS builder` and ship `FROM ${BASE} AS runtime`, copying only what they add. To add one, create a folder in the same shape and add a job to the workflow
- **Build on CPU only** (GitHub's free runners). CUDA extensions are compiled ahead of time in the builder stage with `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` (A100, A6000/3090, 4090/L40S, H100); nothing may JIT-compile at run time
- **Catch missing dependencies in the build**: import what the image is for (and what those modules import lazily) at the end of the Dockerfile, without a GPU
- **Never bake in weights or large data.** Download them into `/workspace` after start
- Tags name the CUDA/torch generation (`cu128`). The run number is appended at push time
- **Check for leaks** before committing: no names from other projects and no credentials

  ```bash
  grep -rn -i --exclude-dir=.git -e '<other project names>' -e 'tok[e]n' -e 'secr[e]t' -e 'api[_]key' .
  ```

  The only expected hits are the two ghcr.io login lines in the workflow (the credential Actions provides automatically). The brackets like `[e]` keep this line from matching itself.
