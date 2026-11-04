"""Fail when a module imported anywhere in an app's source is not installed (CPU only, imports nothing).

    python check_deps.py [--allow-file FILE] [--exclude DIR]... ROOT [ROOT ...]

Every .py file under each ROOT is parsed, every `import` (top level or inside a function) is collected,
and each top-level name that is neither part of the app (a package or module under the ROOT) nor in the
standard library must be found by importlib.util.find_spec. The --allow-file lists (one per line, # comments)
the optional imports that the app guards (try/except, a flag) or only uses on other platforms.
"""
import argparse
import ast
import importlib.util
import pathlib
import sys


def local_names(root):
    names = set()
    for p in root.rglob("*.py"):
        names.add(p.stem)
    for p in root.rglob("*"):
        if p.is_dir():
            names.add(p.name)
    return names


def imports_of(path):
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except SyntaxError as e:
        print(f"skip (syntax error): {path}: {e}", file=sys.stderr)
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-file", type=pathlib.Path)
    ap.add_argument("--exclude", action="append", default=[], metavar="DIR", help="directory name to skip")
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    args = ap.parse_args()
    allow = set()
    if args.allow_file:
        for line in args.allow_file.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                allow.add(line)

    missing, files = {}, 0
    for root in args.roots:
        local = local_names(root)
        for path in sorted(root.rglob("*.py")):
            if set(path.relative_to(root).parts) & set(args.exclude):
                continue
            files += 1
            for name in imports_of(path):
                if name in local or name in sys.stdlib_module_names or name in allow:
                    continue
                if importlib.util.find_spec(name) is None:
                    missing.setdefault(name, []).append(str(path))
    print(f"checked {files} files under {', '.join(map(str, args.roots))}")
    for name, paths in sorted(missing.items()):
        print(f"missing: {name}  (e.g. {paths[0]}, {len(paths)} files)")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
