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
            r = subprocess.run(cmd, input="\n", text=True, capture_output=True)
            if r.returncode == 0:
                status[ds] = True
                break
            print((r.stdout + r.stderr)[-600:])
    return status


def kaggle(dataset="xdxd003/ff-c23") -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download(dataset))
