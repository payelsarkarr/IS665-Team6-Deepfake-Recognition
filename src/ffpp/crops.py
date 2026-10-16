"""Cut face clips out of videos once and cache them (one .npz file per video).

Per video: K clips, evenly spaced over the video. A clip = T frames at about TARGET_FPS
(stride = round(fps / TARGET_FPS)), so a clip covers the same time span in 25 and 30 fps videos.
The face box is found once per clip and kept fixed for all T frames: any change between
frames then comes from the video, not from our cropping.

Cache file layout (no pickle): jpeg = all JPEG bytes concatenated (uint8), offsets = start of
each frame (K*T + 1 entries), meta = [n_clips, T, size], starts = first frame of each clip.
Files are written atomically, so an interrupted run can simply be restarted: finished videos
are skipped.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEFAULTS = dict(clips_per_video=4, frames_per_clip=16, size=224, target_fps=25.0,
                box_scale=1.3, jpeg_quality=90, detect_max_side=640)

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("OpenCV without Haar cascades (OpenCV 5). Install: pip install 'opencv-python-headless<5'")
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def detect_face(bgr, max_side=640):
    """Largest frontal face (x, y, w, h) in full-frame coordinates, or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape
    f = min(1.0, max_side / max(h0, w0))
    small = cv2.resize(gray, (int(w0 * f), int(h0 * f)), interpolation=cv2.INTER_AREA) if f < 1 else gray
    faces = _cascade().detectMultiScale(small, 1.1, 5, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return tuple(int(round(v / f)) for v in (x, y, w, h))


def square_box(box, W, H, scale):
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * scale
    s = min(s, W, H)
    x0 = int(min(max(0, cx - s / 2), W - s))
    y0 = int(min(max(0, cy - s / 2), H - s))
    return x0, y0, int(s)


def extract_video(path: str, out_file: str, cfg: dict) -> dict:
    """Write the clip cache for one video. Returns a status dict (never raises)."""
    K, T, size = cfg["clips_per_video"], cfg["frames_per_clip"], cfg["size"]
    info = dict(n_clips=0, fps=np.nan, n_frames=0, stride=1, status="ok")
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {**info, "status": "unreadable"}
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, int(round(fps / cfg["target_fps"])))
        span = T * stride
        info.update(fps=fps, n_frames=n, stride=stride)
        if n < span:
            cap.release()
            return {**info, "status": "too_short"}
        starts = np.linspace(0, n - span, K + 2)[1:-1].astype(int) if K > 1 else np.array([(n - span) // 2])
        blobs, kept_starts = [], []
        for st in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(st))
            frames = []
            for i in range(span):
                ok, fr = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    frames.append(fr)
            if len(frames) < T:
                continue
            box = None
            for fr in frames[:4]:                      # first frames of the clip
                box = detect_face(fr, cfg["detect_max_side"])
                if box:
                    break
            if box is None:
                continue
            x0, y0, s = square_box(box, W, H, cfg["box_scale"])
            for fr in frames:
                crop = cv2.resize(fr[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]])
                blobs.append(buf.ravel())
            kept_starts.append(int(st))
        cap.release()
        if not kept_starts:
            return {**info, "status": "no_face"}
        offsets = np.concatenate([[0], np.cumsum([b.size for b in blobs])]).astype(np.int64)
        tmp = out_file + ".tmp.npz"
        np.savez(tmp, jpeg=np.concatenate(blobs), offsets=offsets,
                 meta=np.array([len(kept_starts), T, size]), starts=np.array(kept_starts))
        os.replace(tmp, out_file)
        return {**info, "n_clips": len(kept_starts)}
    except Exception as e:  # one broken video must not stop the whole run
        return {**info, "status": f"error: {e}"}


def load_clips(cache_file) -> np.ndarray:
    """Decode a cache file -> uint8 array (n_clips, T, size, size, 3) in RGB."""
    z = np.load(cache_file)
    n_clips, T, size = z["meta"]
    jpeg, off = z["jpeg"], z["offsets"]
    out = np.empty((n_clips * T, size, size, 3), np.uint8)
    for i in range(n_clips * T):
        out[i] = cv2.cvtColor(cv2.imdecode(jpeg[off[i]:off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out.reshape(n_clips, T, size, size, 3)


def cache_name(video_id: str) -> str:
    return video_id.replace("/", "__") + ".npz"


def _job(args):
    cv2.setNumThreads(1)                      # one process per core; avoid oversubscription
    vid, path, out_file, cfg = args
    if os.path.exists(out_file):
        return vid, {"status": "cached"}
    return vid, extract_video(path, out_file, cfg)


def extract_all(man: pd.DataFrame, cache_dir, cfg: dict | None = None, workers: int | None = None,
                log_every: int = 100) -> pd.DataFrame:
    """Extract clips for every video in the manifest; restartable. Writes index.csv + config.json."""
    cfg = {**DEFAULTS, **(cfg or {})}
    cache_dir = Path(cache_dir)
    (cache_dir / "videos").mkdir(parents=True, exist_ok=True)
    cfg_file = cache_dir / "config.json"
    if cfg_file.exists():
        old = json.load(open(cfg_file))
        if {k: old.get(k) for k in DEFAULTS} != {k: cfg[k] for k in DEFAULTS}:
            raise SystemExit(f"{cache_dir} was built with other settings {old}. Use a new cache version.")
    json.dump(cfg, open(cfg_file, "w"), indent=2)
    jobs = [(r.video_id, r.path, str(cache_dir / "videos" / cache_name(r.video_id)), cfg) for r in man.itertuples()]
    status, t0 = {}, time.time()
    workers = workers or os.cpu_count() or 1
    with Pool(workers) as pool:
        for i, (vid, st) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), 1):
            status[vid] = st
            if i % log_every == 0 or i == len(jobs):
                rate = (time.time() - t0) / i
                print(f"{i}/{len(jobs)} videos  ({rate:.2f} s/video, about {(len(jobs) - i) * rate / 60:.0f} min left)", flush=True)
    return write_index(man, cache_dir, status)


def write_index(man: pd.DataFrame, cache_dir, status: dict | None = None) -> pd.DataFrame:
    """index.csv = manifest + cache file + number of clips (read back from the files)."""
    cache_dir = Path(cache_dir)
    idx = man.drop(columns=["path"]).copy()
    idx["cache_file"] = [f"videos/{cache_name(v)}" for v in idx.video_id]
    n_clips, st = [], []
    for v, f in zip(idx.video_id, idx.cache_file):
        p = cache_dir / f
        s = (status or {}).get(v, {}).get("status", "")
        if p.exists():
            n_clips.append(int(np.load(p)["meta"][0]))
            st.append("ok")
        else:
            n_clips.append(0)
            st.append(s or "missing")
    idx["n_clips"], idx["status"] = n_clips, st
    idx.to_csv(cache_dir / "index.csv", index=False)
    return idx
