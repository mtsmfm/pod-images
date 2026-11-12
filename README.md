# pod-images

Custom Docker images for RunPod (and local use), built by GitHub Actions and published to ghcr.io.
This repository holds image definitions only.

Each image is for one purpose. Apps live in `/opt/<app>` at a fixed commit, with every pip dependency
installed; models and weights are never in the image.

| Purpose | Image | torch (base) | Apps (commit) | Compressed | Needed after start |
| --- | --- | --- | --- | --- | --- |
| Base | `ghcr.io/mtsmfm/pod-base:cu128` | 2.7.1 / torchvision 0.22.1 | — | 4.28 GB | — |
| Base | `ghcr.io/mtsmfm/pod-base:cu128-torch280` | 2.8.0 / torchvision 0.23.0 | — | 4.34 GB | — |
| Base | `ghcr.io/mtsmfm/pod-base:cu130` | 2.9.1 / torchvision 0.24.1 (CUDA 13.0) | — | 3.19 GB | — |
| Lyra 2.0 inference | `ghcr.io/mtsmfm/pod-lyra2:cu128` | 2.7.1 (`pod-base:cu128`) | [Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2) `9fffc9a` with flash-attn, Transformer Engine, ViPE, Depth Anything 3 + gsplat, MoGe | 5.69 GB | Weights (97 GB) in `/workspace/lyra-ckpt`: `/opt/start.sh` downloads them |
| Generation | `ghcr.io/mtsmfm/pod-comfy:cu128` | 2.8.0 (`pod-base:cu128-torch280`) | [ComfyUI](https://github.com/Comfy-Org/ComfyUI) v0.45.0 (`989d93d`) and its requirements, plus the pip dependencies of custom nodes (not their code); comfy-kitchen built with CUDA 12.8 | 5.13 GB | Custom nodes and models under `/workspace/ComfyUI` (or `extra_model_paths.yaml`) |
| Generation (fast) | `ghcr.io/mtsmfm/pod-comfy:cu130` | 2.9.1 cu130 (`pod-base:cu130`) | The same; comfy-kitchen from PyPI | 4.01 GB | The same |
| Training | `ghcr.io/mtsmfm/pod-train:cu128` | 2.8.0 (`pod-base:cu128-torch280`) | [ai-toolkit](https://github.com/ostris/ai-toolkit) `0473387` (CLI, no web UI) and [sd-scripts](https://github.com/kohya-ss/sd-scripts) `068bcd7`, in one venv | 5.12 GB | Base models, text encoders and VAEs in `/workspace` |
| (experiment) | `ghcr.io/mtsmfm/pod-lyra2-rp:cu128` | 2.7.1 (`runpod/pytorch:1.6.0-cu1281-torch271-ubuntu2204`, Python 3.12) | Same as pod-lyra2. See below | 11.90 GB | Same as pod-lyra2 |

Measured 2026-11-04; the workflow summary has the current numbers.

**`cu128` or `cu130`** (the tag names the CUDA the image is built with):

- `cu128` runs on any host whose driver supports CUDA 12.8 (driver 570 or later), which is every RunPod host seen so far.
- `cu130` is for hosts whose driver supports CUDA 13.0 (580 or later). It is faster: ComfyUI uses comfy-kitchen's CUDA kernels (int8 / nvfp4 matmuls, int8 attention) only with a cu130 torch.
- pod-comfy also has `:cu128-zstd` and `:cu130-zstd` (and `-zstd-<run_number>`): the same build with
  zstd layers (level 3), smaller and faster to unpack. They need Docker 23.0+ / containerd 1.5+ on the
  host; fall back to the gzip tag where the pull fails. CI checks that every layer is zstd, that the
  `rootfs.diff_ids` match the gzip tag's, and runs `comfy/check_image.sh` in the pulled zstd image

Builder stages, only used to build derived images: `ghcr.io/mtsmfm/pod-base-builder:cu128`,
`:cu128-torch280` and `:cu130` (the matching pod-base plus CUDA devel, cuDNN and compilers; 9.83 / 9.90 / 7.17 GB).

Each image also gets a pinned tag `<tag>-<run_number>` (`cu128-123`, `cu128-torch280-123`); `:cu128`
always points to the latest build and older builds stay reachable by their run tag.

**Apps in `/opt/<app>` are meant to stay read-only.** For work that writes to the app's directory
(custom nodes, models, outputs, configs), copy it first, e.g. `rsync -a /opt/ComfyUI/ /workspace/ComfyUI/`
or `rsync -a /opt/ai-toolkit/ /workspace/ai-toolkit/`, and run it from there. The Python with all the
dependencies is `/opt/venv/bin/python`, first on `PATH` (`python`, `python3`, `pip`); `/usr/bin/python3`
has torch only.

### How the images are built

Every image has two stages. The **builder** stage (CUDA devel, nvcc, compilers) installs the Python
dependencies and compiles every CUDA extension for `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
(A100, A6000/3090, 4090/L40S, H100). The **runtime** stage, the one that is pushed, has no nvcc
and only copies the results over:

- pod-base's runtime starts from `nvidia/cuda:12.8.1-base-ubuntu22.04` (cudart only, 0.1 GB) and gets
  torch's site-packages from the builder. torch brings its own CUDA libraries (cuDNN, cuBLAS, NCCL, ...)
  as pip `nvidia-*` wheels, so `cudnn-runtime` (2.86 GB compressed) would only add a second copy.
  It keeps `gcc`, `libc6-dev` and `python3.10-dev`: triton compiles a small C helper against
  `Python.h` at run time (torch.compile, bitsandbytes, triton kernels). The build checks both.
- CUDA and torch are build arguments of `base/Dockerfile` (`BUILDER_IMAGE`, `RUNTIME_IMAGE`, `TORCH`,
  `TORCHVISION`, `TORCH_INDEX`); the workflow builds the three variants above. cu130 uses
  `nvidia/cuda:13.0.3-*-ubuntu22.04` so that it keeps Python 3.10 like the others. A derived image uses the builder and runtime of the same variant.
- Derived images install everything they add into the venv `/opt/venv` (`--system-site-packages`, so
  torch comes from the base, pinned with constraints so pip never replaces it) and copy only
  `/opt/venv` and `/opt/<app>` onto pod-base. Their layers are pod-base's plus their own.
- Nothing compiles at run time: gsplat and ViPE would JIT-compile their extension on first import
  when it is missing, and the runtime stage has no nvcc to do it.
- The last `RUN` of each Dockerfile checks the image on CPU (no GPU in CI) and fails the build
  otherwise: imports, the apps' CLIs (`--help`, ComfyUI's `--quick-test-for-ci`) and, where it applies,
  a scan for modules imported anywhere in the app's source that are not installed.

The workflow writes each image's compressed size (sum of the amd64 manifest's layers) to the run
summary, with a warning above the target (pod-base 5 GB, pod-lyra2 7 GB, pod-comfy and pod-train 6 GB):
above about 6 GB, RunPod's image extraction gets noticeably slower.

| Image | Compressed (2026-11-05) | Before the two-stage builds |
| --- | --- | --- |
| pod-base:cu128 | 4.28 GB (4.20 GB before gcc and Python.h) | 9.99 GB |
| pod-base:cu128-torch280 | 4.34 GB | — |
| pod-base-builder:cu128 | 9.83 GB | — |
| pod-base-builder:cu128-torch280 | 9.90 GB | — |
| pod-base:cu130 | 3.19 GB | — |
| pod-base-builder:cu130 | 7.17 GB | — |
| pod-comfy:cu128 | 5.13 GB (pod-base + 0.79 GB, ComfyUI v0.45.0) | — |
| pod-comfy:cu130 | 4.01 GB (pod-base + 0.82 GB, ComfyUI v0.45.0) | — |
| pod-comfy:cu128-zstd | 4.46 GB (the same files as :cu128, zstd level 3) | — |
| pod-comfy:cu130-zstd | 3.71 GB (the same files as :cu130, zstd level 3) | — |
| pod-train:cu128 | 5.12 GB (pod-base + 0.78 GB) | — |
| pod-lyra2:cu128 | 5.69 GB (pod-base + 1.4 GB) | 12.05 GB |
| pod-lyra2-rp:cu128 | 11.90 GB (runpod/pytorch 10.5 GB + 1.4 GB) | — |

## Usage

### Building

- **Automatic**: a push to `main` that touches an image folder or `.github/` builds pod-base (both variants, builder and runtime), then pod-lyra2, pod-comfy and pod-train; pod-lyra2-rp builds in parallel. Each build is `.github/workflows/image.yml`, called from `build.yml`.
- **Manual**: Actions → `build` → *Run workflow*, and pick `all` / `base` / `lyra2` / `comfy` / `train` / `lyra2-rp` for `image`. With one derived image alone, base is skipped and the image is built on the pod-base-builder / pod-base already on ghcr.io.
- **Pull request**: a PR that touches the same paths builds everything and pushes it only under `:<tag>-pr<number>` (for example `pod-comfy:cu130-pr3`), on the base images of the same PR. The real tags do not change until the merge, whose build reuses the PR's cached layers.
- **Local**: `./build.sh base <user> [cu128-torch280|cu130]`, `./build.sh comfy <user>` and so on (set `PUSH=1` to push as well). No GPU needed.

After the first push, make each package (pod-base, pod-base-builder, pod-lyra2, pod-comfy, pod-train, pod-lyra2-rp) **Public** on ghcr.io: *Package settings → Change visibility*. This is separate from the repository's visibility.

### Updating an app's commit

Each app's commit is an `ARG *_REF` with its default in the Dockerfile: `COMFY_REF` (comfy/),
`AITK_REF` and `SD_SCRIPTS_REF` (train/), `LYRA_REF`, `MOGE_REF` and `GSPLAT_REF` (lyra2/). Change
the default and push; the build fails if the new commit's dependencies do not install or import.
Whatever uses the image (custom nodes, configs, scripts sent at start) should move to the same commit,
since `:cu128` follows the new build. Pin a `cu128-<run_number>` tag to stay on the old one.

### Generation: pod-comfy

```bash
docker run --gpus all -it -p 8188:8188 -v $PWD/workspace:/workspace ghcr.io/mtsmfm/pod-comfy:cu128 bash -c '
  rsync -a /opt/ComfyUI/ /workspace/ComfyUI/ && cd /workspace/ComfyUI && python main.py --listen 0.0.0.0'
```

- ComfyUI comes from the GitHub tarball of `COMFY_REF` (no `.git`) at `/opt/ComfyUI` (`COMFY_HOME`).
  Its `requirements.txt` is installed except `torch`, `torchvision` and `torchaudio` (the base's are
  kept; ComfyUI does not import torchaudio). `COMFY_VERSION` (the runtime stage's ARG) must match
  `comfyui_version.py` at `COMFY_REF`; the build checks it
- 7-day cooldown: pip takes only releases uploaded before `UPLOADED_PRIOR_TO` (`--uploaded-prior-to`;
  set it 7 days before the day you change `COMFY_REF`). `comfy/cooldown-exempt.txt` lists exceptions
  (pinned packages ComfyUI needs sooner), installed first with `--no-deps` and the cutoff
  `EXEMPT_UPLOADED_PRIOR_TO`; empty it once `UPLOADED_PRIOR_TO` has passed their upload date
- comfy-kitchen (ComfyUI's quantized kernels): its PyPI wheels are built with CUDA 13.0 and link
  cudart statically, so with them even the cu128 image would need a CUDA 13 driver for int8 attention
  (`detect_k_anchor kernel launch failed: CUDA driver version is insufficient`). For cu128 it is built
  from source (the git tag of the version ComfyUI pins, with its submodules) with the image's CUDA 12.8,
  for `75-real;80-real;86-real;89;90a-real;120` (`COMFY_KITCHEN_CUDA_ARCHS`; upstream's `100f` / `120f`
  need nvcc 12.9+). Its cuBLASLt-based ops look for cuBLAS 13 only and stay unavailable on cu128
  (ComfyUI turns comfy-kitchen's CUDA backend off with a cu128 torch anyway). cu130 uses the PyPI wheel
- The build checks with `comfy/check_cuda.py` that torch is the expected CUDA build, that the
  comfy-kitchen kernels were built with the same CUDA major version, and that no library in `/opt/venv`
  was built with a newer CUDA than torch's (it reads nvcc's `release X.Y` from the binaries)
- `comfy/requirements-extra.txt`: the pip dependencies of custom nodes sent at start (`timm`).
  The nodes' code is not in the image; put it in `custom_nodes/` of the copy
- Models are not in the image: put them in the copy's `models/` or point `extra_model_paths.yaml` at `/workspace`
- `libgl1`, `libglib2.0-0` and the `ffmpeg` command (for checking `CreateVideo` / `SaveVideo` output) come with pod-base
- The build runs `python main.py --cpu --quick-test-for-ci` in a copy, which loads every built-in node
  (`comfy_extras`, `comfy_api_nodes`) and fails on `IMPORT FAILED`. Then `comfy/check_nodes.py` starts
  the server on the CPU and fails unless `/object_info` has every node in `comfy/required-nodes.txt`

### Training: pod-train

```bash
docker run --gpus all -it -v $PWD/workspace:/workspace ghcr.io/mtsmfm/pod-train:cu128 bash -c '
  rsync -a /opt/ai-toolkit/ /workspace/ai-toolkit/ && cd /workspace/ai-toolkit && python run.py config/my_job.yaml'
# sd-scripts (Anima LoRA):
#   cd /opt/sd-scripts && python anima_train_network.py ...   (or a copy in /workspace)
```

- ai-toolkit (`AITK_HOME=/opt/ai-toolkit`) and sd-scripts (`SD_SCRIPTS_HOME=/opt/sd-scripts`) from
  their GitHub tarballs, in one venv. ai-toolkit's web UI (`ui/`, Node.js) is removed; `run.py` is the entry point
- Both `requirements.txt` are installed with torch / torchvision kept at the base's. Where they pin
  different versions (diffusers, transformers, accelerate, huggingface-hub, safetensors) ai-toolkit's
  win; sd-scripts' other pins are installed only if they keep ai-toolkit's exact pins. sd-scripts'
  `-e .` is skipped: its scripts import `library/` from their own directory
- Built for torch 2.8.0 (ai-toolkit's own installer targets 2.13): `torchaudio==2.8.0` (ai-toolkit
  imports it) and `torchcodec==0.7.0` instead of ai-toolkit's 0.15.0, which needs torch 2.11+
- What ai-toolkit's installer adds on Linux + CUDA is included: `flash-linear-attention` and the
  prebuilt flash-attn 2.8.3.post2 wheel for torch 2.8. NATTEN (no build for torch 2.8) is not
- The build imports ai-toolkit's main modules, runs `run.py --help` and `anima_train_network.py --help`,
  and scans both apps' source for imports that are not installed (`train/check_deps.py`); the
  optional ones (other optimizers and loggers, xformers, captioning tools, ...) are listed in
  `train/optional-imports.txt`. Training itself needs a GPU and is not run in CI
- `HF_HOME=/workspace/hf`, so downloaded models land in `/workspace`

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

- **One folder per purpose.** Derived images build in `FROM ${BUILDER} AS builder` and ship `FROM ${BASE} AS runtime`, copying only what they add. To add one, create a folder in the same shape and add a job to `build.yml`
- **Build on CPU only** (GitHub's free runners). CUDA extensions are compiled ahead of time in the builder stage with `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` (A100, A6000/3090, 4090/L40S, H100); nothing may JIT-compile at run time
- **Catch missing dependencies in the build**: import what the image is for (and what those modules import lazily) at the end of the Dockerfile, without a GPU
- **Never bake in weights or large data.** Download them into `/workspace` after start
- Tags name the CUDA/torch generation (`cu128`, `cu128-torch280`). The run number is appended at push time
- **Keep compressed sizes at 6 GB or less** (the summary warns above the target)
- **Check for leaks** before committing: no names from other projects and no credentials

  ```bash
  grep -rn -i --exclude-dir=.git -e '<other project names>' -e 'tok[e]n' -e 'secr[e]t' -e 'api[_]key' .
  ```

  The only expected hit is the ghcr.io login line in `.github/workflows/image.yml` (the credential Actions provides automatically). The brackets like `[e]` keep this line from matching itself.
