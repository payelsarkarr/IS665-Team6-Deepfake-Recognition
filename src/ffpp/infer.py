"""Score new video files end to end: video -> face clips -> model -> P(fake).

Uses exactly the same steps as training and evaluation (crops.extract_video, VideoEvalSet,
the trained checkpoint and the threshold chosen on validation), so a score here means the same
as a score in notebook 03. Every step is timed for the latency report.

    det = Detector(f"{RUNS_DIR}/mnv2_v1")          # loads best.pt + threshold from eval/summary.json
    r = det.score("video.mp4")                    # dict: score, verdict, per-clip scores, timings
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .crops import DEFAULTS, _cascade, extract_video
from .dataset import FrameCache, VideoEvalSet, to_tensor
from .models import CLIP_MODELS
from .train import load_trained


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Detector:
    def __init__(self, run_dir, which="best", device=None, crop_cfg=None):
        self.run_dir = Path(run_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg, self.epoch = load_trained(self.run_dir, which, self.device)
        self.mode = "clip" if self.cfg["model"] in CLIP_MODELS else "frame"
        summ = self.run_dir / "eval" / "summary.json"
        self.threshold = json.load(open(summ))["threshold_from_val"] if summ.exists() else 0.5
        self.threshold_source = "validation (notebook 03)" if summ.exists() else "default 0.5 (run notebook 03 first)"
        self.crop_cfg = {**DEFAULTS, **(crop_cfg or {})}      # same settings as the training cache (crops_v1)
        self.warm = False

    def to(self, device):
        """Move the model, e.g. to the CPU for the CPU latency measurement."""
        self.device = torch.device(device)
        self.model.to(self.device)
        self.warm = False
        return self

    @torch.no_grad()
    def _forward(self, x):
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            return torch.sigmoid(self.model(x.to(self.device)).float()).cpu().numpy().ravel()

    def warmup(self):
        """First calls are slow (CUDA/cuDNN set-up, loading the face detector); do them once before timing."""
        shape = (1, self.cfg["clip_len"], 3, self.cfg["img_size"], self.cfg["img_size"]) if self.mode == "clip" \
            else (2, 3, self.cfg["img_size"], self.cfg["img_size"])
        self._forward(torch.zeros(shape))
        _sync(self.device)
        _cascade()                                   # load the face detector once, not inside the timing
        self.warm = True

    def score(self, path, timeline=True) -> dict:
        path = Path(path)
        if not self.warm:
            self.warmup()
        out = dict(video=path.name, device=self.device.type, model=self.cfg["model"], threshold=self.threshold)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "v.npz"
            t0 = time.perf_counter()
            info = extract_video(str(path), str(cache), self.crop_cfg)           # decode + face detection + crops
            t1 = time.perf_counter()
            out.update(status=info["status"], n_clips=info["n_clips"], fps=info["fps"], n_frames=info["n_frames"],
                       duration_s=info["n_frames"] / info["fps"] if info["fps"] else np.nan,
                       t_extract_ms=(t1 - t0) * 1e3)
            if info["status"] != "ok":
                return out
            idx = pd.DataFrame([dict(video_id=path.stem, label=-1, cache_file="v.npz")])
            x, _, _ = VideoEvalSet(idx, tmp, self.mode, self.cfg["img_size"], self.cfg["clip_len"],
                                   self.cfg["eval_frames"])[0]
            t2 = time.perf_counter()
            _sync(self.device)
            t3 = time.perf_counter()
            p = self._forward(x)                                                   # same samples as notebook 03
            _sync(self.device)
            t4 = time.perf_counter()
            out.update(score=float(p.mean()), score_max=float(p.max()),
                       verdict="FAKE" if p.mean() >= self.threshold else "real",
                       t_prepare_ms=(t2 - t1) * 1e3, t_model_ms=(t4 - t3) * 1e3,
                       t_total_ms=(t4 - t0) * 1e3)
            if timeline:                                                           # not timed: for the plot only
                fc = FrameCache(cache)
                if self.mode == "frame":
                    per_frame = [self._forward(to_tensor(fc.frames(c, range(fc.T)), self.cfg["img_size"]))
                                 for c in range(fc.n_clips)]
                    out["frame_scores"] = [f.tolist() for f in per_frame]
                    out["clip_scores"] = [float(f.mean()) for f in per_frame]
                else:
                    out["clip_scores"] = p.tolist()
                out["clip_faces"] = [fc.frame(c, fc.T // 2) for c in range(fc.n_clips)]
                out["clip_start_s"] = (np.load(cache)["starts"] / info["fps"]).round(1).tolist()
        return out
