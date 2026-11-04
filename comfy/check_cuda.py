"""Check that the image's CUDA code needs no newer driver than its torch does (CPU only, imports only torch).

    python check_cuda.py EXPECTED_TORCH_CUDA        e.g. 12.8 or 13.0

1. torch.version.cuda is EXPECTED_TORCH_CUDA.
2. comfy-kitchen's kernel library (backends/cuda/_C*.so) says which nvcc built it ("Cuda compilation
   tools, release X.Y"), and X is torch's CUDA major version. comfy-kitchen links cudart statically, so
   a kernel built with CUDA 13 needs a CUDA 13 driver (580+) even when torch is cu128.
3. No shared library under /opt/venv was built with a CUDA newer than torch's (the same trap elsewhere).
"""
import importlib.util
import pathlib
import re
import sys

import torch

RELEASE = re.compile(rb"Cuda compilation tools, release (\d+)\.(\d+)")


def releases(path):
    return sorted({(int(a), int(b)) for a, b in RELEASE.findall(path.read_bytes())})


def main():
    expected = sys.argv[1]
    ok = True
    torch_cuda = torch.version.cuda
    print(f"torch {torch.__version__}, CUDA {torch_cuda} (expected {expected})")
    if torch_cuda != expected:
        print("error: torch's CUDA version is not the expected one")
        ok = False
    major, minor = (int(x) for x in torch_cuda.split(".")[:2])

    spec = importlib.util.find_spec("comfy_kitchen")
    kernels = sorted(pathlib.Path(spec.submodule_search_locations[0]).glob("backends/cuda/_C*.so"))
    if not kernels:
        print("error: comfy-kitchen has no CUDA kernel library")
        ok = False
    for so in kernels:
        found = releases(so)
        print(f"comfy-kitchen kernel {so.name}: built with CUDA {', '.join(f'{a}.{b}' for a, b in found) or '?'}")
        if not found or any(a != major for a, b in found):
            print(f"error: the kernel is not built with CUDA {major}.x")
            ok = False

    newer = []
    count = 0
    for so in pathlib.Path(sys.prefix).rglob("*.so*"):
        if not so.is_file() or so.is_symlink():
            continue
        count += 1
        found = [r for r in releases(so) if r > (major, minor)]
        if found:
            newer.append(f"{so}: CUDA {', '.join(f'{a}.{b}' for a, b in found)}")
    print(f"checked {count} shared libraries under {sys.prefix} for CUDA newer than {major}.{minor}")
    for line in newer:
        print(f"error: {line}")
    sys.exit(0 if ok and not newer else 1)


if __name__ == "__main__":
    main()
