"""Get the videos into the Colab runtime (not into Drive).

official: the FF++ download script you received after the access request (keep it in Drive,
          never in the repo). Tries the servers EU, EU2, CA in turn.
kaggle:   fallback mirror xdxd003/ff-c23 (same c23 videos) if the official servers are down.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .data import METHODS


def _run_live(cmd) -> tuple[int, str]:
    """Run the script, confirm the ToS prompt with a newline and show its progress live
    (tqdm redraws its bar with carriage returns, so we split on both \\r and \\n)."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    p.stdin.write("\n"); p.stdin.close()
    buf, tail = "", []
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        if ch in "\r\n":
            if buf.strip():
                tail = (tail + [buf])[-15:]
                if "%|" in buf:                       # progress bar: overwrite the same line
                    print("\r" + buf[:120], end="", flush=True)
                elif not buf.startswith(("By pressing", "http", "***", "Press any key")):
                    print(buf, flush=True)
            buf = ""
        else:
            buf += ch
    rc = p.wait()
    print()
    return rc, "\n".join(tail)


def official(script, out_dir, methods=METHODS, compression="c23", num_videos=0,
             servers=("EU", "EU2", "CA")) -> dict:
    """Run the official script for 'original' + each method. Passing a newline confirms the
    Terms-of-Use prompt, which you accepted with the access request."""
    status = {}
    for ds in ["original", *methods]:
        status[ds] = False
        for server in servers:
            cmd = [sys.executable, str(script), str(out_dir), "-d", ds, "-c", compression, "-t", "videos",
                   "--server", server]
            if num_videos:
                cmd += ["--num_videos", str(num_videos)]
            print(f"== {ds} (server {server})", flush=True)
            rc, tail = _run_live(cmd)
            if rc == 0:
                status[ds] = True
                servers = (server,) + tuple(s for s in servers if s != server)   # keep using the server that works
                break
            last = tail.strip().splitlines()[-1] if tail.strip() else "no output"
            print(f"   server {server} failed: {last}")
    return status


def kaggle(dataset="xdxd003/ff-c23") -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download(dataset))
