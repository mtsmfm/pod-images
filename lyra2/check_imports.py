"""Build-time dependency check for pod-lyra2 (runs without a GPU).

1. Imports the inference entry points (and the modules the brief names).
2. Walks every Lyra / Depth Anything 3 / ViPE / MoGe source file that got loaded and collects
   all imports in it, including the ones inside functions (lazy imports such as `addict`
   only fail at run time otherwise).
3. Fails if any third-party top-level module cannot be found, except the known optional ones.

Usage:
  python check_imports.py                    # the import check
  python check_imports.py --help-of MODULE   # run `python -m MODULE --help` (with the same no-GPU stub)
"""
import ast
import importlib
import importlib.util
import os
import runpy
import sys

ENTRY_POINTS = [
    "addict",
    "depth_anything_3.api",
    "moge.model.v1",
    "gsplat",
    "vipe",
    "lyra_2._src.inference.lyra2_custom_traj_inference",
    "lyra_2._src.inference.lyra2_zoomgs_inference",
    "lyra_2._src.inference.vipe_da3_gs_recon",
]

# Imported inside try/except or only for debugging / training / other backends.
OPTIONAL = {
    "debugpy", "ipdb", "xformers", "flash_attn_3", "straggler", "dill", "cloudpickle",
    "boto3", "botocore", "aioboto3", "aiobotocore", "tensorboard", "wandb", "mlflow",
    "gradio", "spaces", "apex", "deepspeed",
    # utils3d: fallback when utils3d_moge is missing; imaginaire: only with --context_parallel_size > 1 (upstream bug)
    "utils3d", "imaginaire",
    # GPU-only: checked by /opt/start.sh on a GPU
    "flash_attn", "transformer_engine", "vipe_ext",
}

LOCAL_PACKAGES = ("lyra_2", "depth_anything_3", "vipe", "moge")


def stub_triton_driver_without_gpu() -> None:
    """Let modules that use @triton.autotune at import time (Transformer Engine) import without a GPU.

    triton's Autotuner asks the active driver for a benchmarker when the decorator runs, and there
    is no active driver without a GPU ("0 active drivers"). Only the import check needs this; the
    kernels are never run here. Anything else asked of the driver fails loudly.
    """
    import torch

    if torch.cuda.is_available():
        return
    try:
        from triton.runtime import driver
    except ImportError:
        return

    class NoGpuDriver:
        def get_benchmarker(self):
            return lambda *args, **kwargs: [0.0]

        def __getattr__(self, name):
            raise RuntimeError(f"check_imports: no GPU here (triton driver.{name} was used at import time)")

    driver.set_active(NoGpuDriver())


def main() -> int:
    stub_triton_driver_without_gpu()
    failed = []
    for name in ENTRY_POINTS:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            failed.append(f"import {name}: {type(e).__name__}: {e}")

    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py") and (getattr(mod, "__name__", "") or "").split(".")[0] in LOCAL_PACKAGES:
            files.add(f)

    wanted = {}
    for path in sorted(files):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for n in names:
                wanted.setdefault(n.split(".")[0], path)

    for top, path in sorted(wanted.items()):
        if top in sys.stdlib_module_names or top in LOCAL_PACKAGES or top in OPTIONAL:
            continue
        try:
            found = importlib.util.find_spec(top) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            failed.append(f"missing module {top!r} (imported in {os.path.relpath(path)})")

    print(f"checked {len(files)} loaded source files, {len(wanted)} imported top-level modules")
    for line in failed:
        print("FAIL", line)
    return 1 if failed else 0


def run_help(module: str) -> int:
    """`python -m MODULE --help` without a GPU: the CLI must get as far as argparse."""
    stub_triton_driver_without_gpu()
    sys.argv = [module, "--help"]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:  # argparse exits 0 after printing the help
        return int(e.code or 0)
    print(f"FAIL {module} --help did not exit")
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--help-of":
        sys.exit(run_help(sys.argv[2]))
    sys.exit(main())
