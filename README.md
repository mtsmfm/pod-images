# pod-images

Custom Docker images for RunPod (and local use), built by GitHub Actions and published to ghcr.io.
This repository holds image definitions only.

| Image | Contents |
| --- | --- |
| `ghcr.io/mtsmfm/pod-base:cu128` | General-purpose base: CUDA 12.8.1 + cuDNN (devel), Ubuntu 22.04, Python 3.10, torch 2.7.1 / torchvision 0.22.1, huggingface_hub |
| `ghcr.io/mtsmfm/pod-lyra2:cu128` | Inference environment for [Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2) (flash-attn, Transformer Engine, ViPE, Depth Anything 3 + gsplat, MoGe) |

Each image also gets a pinned tag `cu128-<run_number>` (`:cu128` always points to the latest build).

## Usage

### Building

- **Automatic**: a push to `main` that touches `base/**`, `lyra2/**` or `.github/workflows/build.yml` builds and pushes base, then lyra2.
- **Manual**: Actions → `build` → *Run workflow*, and pick `all` / `base` / `lyra2` for `image`. With `lyra2` alone, base is skipped and lyra2 is built on the `pod-base:cu128` already on ghcr.io.
- **Local**: `./build.sh base <user>`, `./build.sh lyra2 <user>` (set `PUSH=1` to push as well). No GPU needed.

After the first push, make each package (pod-base, pod-lyra2) **Public** on ghcr.io: *Package settings → Change visibility*. This is separate from the repository's visibility.

### Running Lyra 2.0

```bash
docker run --gpus all -it -v $PWD/workspace:/workspace ghcr.io/mtsmfm/pod-lyra2:cu128 bash /opt/start.sh
```

`/opt/start.sh`:

1. Downloads `checkpoints/*` from `nvidia/Lyra-2.0` on Hugging Face unless `/workspace/lyra-ckpt/checkpoints/model` exists
2. Symlinks `/opt/lyra/Lyra-2/checkpoints` to it
3. Checks the imports `torch, flash_attn, transformer_engine.pytorch, depth_anything_3.api, moge.model.v1` and `vipe_ext` (or `vipe`), and prints `torch.cuda.is_available()` and the GPU name
4. Runs `python -m lyra_2._src.inference.lyra2_zoomgs_inference --help`

For running inference, see the [Lyra-2 README](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2). The source is at `/opt/lyra/Lyra-2` (`PYTHONPATH` is set).

License: Lyra's code is Apache 2.0; the weights are under NVIDIA's research & development license (see the Hugging Face model card). The weights are not included in the image.

Differences from the upstream install guide:

- Installed on top of pod-base (system CUDA 12.8 and Python 3.10) instead of conda
- Transformer Engine is pinned to 2.13.0 (upstream installs the latest; newer releases target CUDA 13 / cuDNN >= 9.12)
- flash-attn uses the official prebuilt 2.8.3.post2 wheel instead of building 2.6.3 from source. If it causes problems, switch to the source-build line in `lyra2/Dockerfile`
- ViPE is built from the vendored source (as upstream does). A commented-out line switches to `nvidia-vipe` from PyPI (which is also an sdist and compiles)

## What RunPod needs

The images contain nothing RunPod-specific.

- `bash`, `python3`, `base64`, `tar` and `setsid` (util-linux) are included. Startup logic is injected by overriding CMD with RunPod's `dockerStartCmd`; the image CMD is `sleep infinity`
- HTTP is exposed on the RunPod side with `ports: 8000/http`
- RunPod injects the pod ID and API key environment variables (used for auto-stopping pods). Nothing is needed in the image
- No RunPod SDK, runpodctl, SSH or Jupyter
- With public packages, RunPod needs no registry credentials. If a package is private, register it under Settings → Container Registry
- Attach a volume at `/workspace` to keep the weights (`/workspace/lyra-ckpt`) and the HF cache (`HF_HOME=/workspace/hf`) across runs

## Rules

- **One folder per purpose.** Derived images take the base via `ARG BASE` and use `FROM ${BASE}`. To add one, create a folder in the same shape and add a job to the workflow
- **Build on CPU only** (GitHub's free runners). CUDA extensions are compiled ahead of time with `TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"` (A6000/3090, 4090/L40S, H100)
- **Never bake in weights or large data.** Download them into `/workspace` after start
- Tags name the CUDA/torch generation (`cu128`). The run number is appended at push time
- **Check for leaks** before committing: no names from other projects and no credentials

  ```bash
  grep -rn -i --exclude-dir=.git -e '<other project names>' -e 'tok[e]n' -e 'secr[e]t' -e 'api[_]key' .
  ```

  The only expected hits are the two ghcr.io login lines in the workflow (the credential Actions provides automatically). The brackets like `[e]` keep this line from matching itself.
