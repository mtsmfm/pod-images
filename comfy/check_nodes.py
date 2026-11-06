"""Start ComfyUI on the CPU and check that /object_info has every node in a list (no GPU needed).

    python check_nodes.py COMFY_DIR NODE_LIST

Runs `main.py --cpu --listen 127.0.0.1 --port 8188` in COMFY_DIR, waits up to 5 minutes for
/object_info, prints "missing: [...]" and stops the server. Exits 1 when a node is missing or the
server does not answer.
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8188/object_info"


def main():
    comfy_dir, node_list = sys.argv[1], sys.argv[2]
    need = [line.strip() for line in pathlib.Path(node_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    log = open("/tmp/comfy-server.log", "w")
    server = subprocess.Popen([sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", "8188"],
                              cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT)
    info = None
    try:
        for _ in range(150):
            if server.poll() is not None:
                break
            try:
                with urllib.request.urlopen(URL, timeout=10) as r:
                    info = json.load(r)
                break
            except OSError:
                time.sleep(2)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    if info is None:
        print("error: ComfyUI did not answer /object_info; the end of its log:")
        print("".join(open("/tmp/comfy-server.log").readlines()[-30:]))
        sys.exit(1)
    missing = [n for n in need if n not in info]
    print(f"{len(info)} nodes in /object_info")
    print("missing:", missing)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
