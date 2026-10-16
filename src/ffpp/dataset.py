"""PyTorch datasets on top of the clip cache.

mode="frame": a sample is ONE frame (for the small CNN and MobileNetV2 per frame).
mode="clip":  a sample is a run of `clip_len` consecutive frames (for MobileNetV2 + GRU).

Training draws a random clip / frame / window and applies ONE random augmentation to all frames
of a sample (otherwise augmentation itself would create flicker between frames).
Evaluation returns all evaluation samples of one video at once, so scores can be averaged
per video.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet statistics (torchvision weights)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_index(cache_dir, split: str | None = None, include_leak_excluded=False) -> pd.DataFrame:
    idx = pd.read_csv(Path(cache_dir) / "index.csv", dtype={"target_id": str, "source_id": str})
    idx = idx[(idx.status == "ok") & (idx.n_clips > 0)]
    if not include_leak_excluded:
        idx = idx[~(idx.leak_excluded & idx.split.isin(["val", "test"]))]
    if split:
        idx = idx[idx.split == split]
    return idx.reset_index(drop=True)


def copy_cache_local(src, dst) -> Path:
    """Copy the cache from Drive to the Colab disk once (reading from Drive is slow).
    Files that are already there with the same size are skipped."""
    src, dst = Path(src), Path(dst)
    (dst / "videos").mkdir(parents=True, exist_ok=True)
    for f in ("index.csv", "config.json"):
        shutil.copy2(src / f, dst / f)
    n = 0
    for f in (src / "videos").iterdir():
        t = dst / "videos" / f.name
        if not t.exists() or t.stat().st_size != f.stat().st_size:
            shutil.copy2(f, t)
            n += 1
    print(f"cache in {dst}: {n} files copied")
    return dst


class FrameCache:
    """Decode only the frames that are needed from a cache file."""

    def __init__(self, file):
        z = np.load(file)
        self.jpeg, self.off = z["jpeg"], z["offsets"]
        self.n_clips, self.T, self.size = (int(v) for v in z["meta"])

    def frame(self, clip: int, t: int) -> np.ndarray:
        i = clip * self.T + t
        return cv2.cvtColor(cv2.imdecode(self.jpeg[self.off[i]:self.off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def frames(self, clip: int, ts) -> np.ndarray:
        return np.stack([self.frame(clip, t) for t in ts])


def augment(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Same random flip / brightness-contrast / JPEG re-compression for every frame of a sample."""
    out = frames
    if rng.random() < 0.5:
        out = out[:, :, ::-1]
    if rng.random() < 0.8:
        a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        q = rng.randint(40, 90)
        out = np.stack([cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_UNCHANGED)
                        for f in np.ascontiguousarray(out)])
    return np.ascontiguousarray(out)


def to_tensor(frames: np.ndarray, img_size: int) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> (N, 3, img_size, img_size) normalised float."""
    if frames.shape[1] != img_size:
        frames = np.stack([cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA) for f in frames])
    x = (frames.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


class TrainSet(Dataset):
    """`samples_per_video` random samples per video and epoch."""

    def __init__(self, idx: pd.DataFrame, cache_dir, mode="frame", img_size=224, clip_len=16,
                 samples_per_video=4, augment_data=True, seed=0):
        self.idx, self.dir, self.mode = idx.reset_index(drop=True), Path(cache_dir), mode
        self.img_size, self.clip_len, self.spv, self.aug = img_size, clip_len, samples_per_video, augment_data
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.idx) * self.spv

    def __getitem__(self, i):
        row = self.idx.iloc[i // self.spv]
        rng = random.Random(hash((self.seed, self.epoch, i)))
        fc = FrameCache(self.dir / row.cache_file)
        clip = rng.randrange(fc.n_clips)
        if self.mode == "frame":
            frames = fc.frames(clip, [rng.randrange(fc.T)])
        else:
            L = min(self.clip_len, fc.T)
            t0 = rng.randrange(fc.T - L + 1)
            frames = fc.frames(clip, range(t0, t0 + L))
        if self.aug:
            frames = augment(frames, rng)
        x = to_tensor(frames, self.img_size)
        if self.mode == "frame":
            x = x[0]
        return x, torch.tensor(float(row.label))


class VideoEvalSet(Dataset):
    """One item = all evaluation samples of one video (frames or clips), no augmentation.
    frame mode: `eval_frames` frames spread over all clips; clip mode: the first clip_len frames of each clip."""

    def __init__(self, idx: pd.DataFrame, cache_dir, mode="frame", img_size=224, clip_len=16, eval_frames=8):
        self.idx, self.dir, self.mode = idx.reset_index(drop=True), Path(cache_dir), mode
        self.img_size, self.clip_len, self.eval_frames = img_size, clip_len, eval_frames

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = self.idx.iloc[i]
        fc = FrameCache(self.dir / row.cache_file)
        if self.mode == "frame":
            total = fc.n_clips * fc.T
            picks = np.linspace(0, total - 1, min(self.eval_frames, total)).astype(int)
            frames = np.stack([fc.frame(p // fc.T, p % fc.T) for p in picks])
            x = to_tensor(frames, self.img_size)                        # (F, 3, H, W)
        else:
            L = min(self.clip_len, fc.T)
            x = torch.stack([to_tensor(fc.frames(c, range(L)), self.img_size) for c in range(fc.n_clips)])  # (C, L, 3, H, W)
        return x, float(row.label), row.video_id
