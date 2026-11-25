"""Get the videos into the Colab runtime (not into Drive).

official: the FF++ download script you received after the access request (keep it in Drive,
          never in the repo). Tries the servers EU, EU2, CA in turn.
kaggle:   fallback mirror xdxd003/ff-c23 (same c23 videos) if the official servers are down.
fetch:    single videos by name (e.g. the FaceShifter test split, demo videos). Uses the file
          layout and server list of the same official script, so nothing about the servers is
          stored in this repository.
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


# ---------------------------------------------------------------- single videos
def _load_script(script):
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location("ff_download_script", str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    urls = re.findall(r"server_url\s*=\s*'(http[^']+)'", Path(script).read_text())
    servers = dict(zip(mod.SERVERS, urls))          # same order as in the script's if/elif chain
    return mod, servers


def fetch(script, out_dir, items, compression="c23", servers=("EU2", "EU", "CA")) -> list[Path]:
    """items: [(dataset, name)], e.g. ("FaceShifter", "000_003") or ("original", "851").
    Saved in the official layout under out_dir, so data.build_manifest finds them. Existing files
    are skipped. Returns the paths that exist afterwards."""
    mod, urls = _load_script(script)
    out_dir, done = Path(out_dir), []
    order = [s for s in servers if s in urls]
    for ds, name in items:
        rel = f"{mod.DATASETS[ds]}/{compression}/videos/{name}.mp4"
        target = out_dir / rel
        if target.exists() and target.stat().st_size > 0:
            done.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        for srv in order:
            try:
                mod.download_file(urls[srv] + "v3/" + rel, str(target))
                if target.exists() and target.stat().st_size > 0:
                    order = [srv] + [x for x in order if x != srv]      # keep the server that works
                    done.append(target)
                    break
            except Exception as e:                                     # try the next server
                print(f"   {name}: server {srv} failed ({type(e).__name__})")
        else:
            print(f"   {ds}/{name}: not downloaded")
    print(f"{len(done)}/{len(items)} videos in {out_dir}")
    return done


def split_pairs(meta_dir, split="test", exclude_ids=()) -> list[str]:
    """Fake video names (target_source, both directions) of an official split, without excluded ids."""
    import json
    pairs = json.load(open(Path(meta_dir) / "splits" / f"{split}.json"))
    names = []
    for a, b in pairs:
        if a in exclude_ids or b in exclude_ids:
            continue
        names += [f"{a}_{b}", f"{b}_{a}"]
    return names
