"""Final evaluation of a run: threshold chosen on validation, applied unchanged to test.

runs/<run_name>/eval/
  summary.json            val + test metrics (at 0.5 and at the chosen threshold)
  per_method_test.csv     recall per manipulation method + FPR on real videos (test)
  predictions_<split>.csv per-video scores
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from . import metrics as M
from .dataset import VideoEvalSet, load_index
from .models import CLIP_MODELS
from .train import load_trained, predict_videos


def per_method(pred: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Share flagged as fake per method: recall for the fake methods, FPR for 'original'."""
    g = pred.assign(flagged=pred.score >= thr).groupby("method")
    return pd.DataFrame({"videos": g.size(), "flagged_share": g.flagged.mean().round(4)}).reset_index()


def evaluate_run(run_dir, cache_dir, which="best", splits=("val", "test"), max_fpr=0.2, device=None,
                 num_workers=2) -> dict:
    run_dir, out = Path(run_dir), Path(run_dir) / "eval"
    out.mkdir(exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch = load_trained(run_dir, which, device)
    mode = "clip" if cfg["model"] in CLIP_MODELS else "frame"
    preds = {}
    for s in splits:
        ds = VideoEvalSet(load_index(cache_dir, s), cache_dir, mode, cfg["img_size"], cfg["clip_len"], cfg["eval_frames"])
        preds[s] = predict_videos(model, ds, device, num_workers)
        preds[s].to_csv(out / f"predictions_{s}.csv", index=False)
    thr = M.threshold_for_fpr(preds["val"].label, preds["val"].score, max_fpr) if "val" in preds else 0.5
    res = dict(run=cfg["run_name"], model=cfg["model"], checkpoint=which, epoch=epoch + 1,
               threshold_from_val=thr, max_fpr=max_fpr)
    for s, p in preds.items():
        res[s] = dict(n=len(p), n_real=int((p.label == 0).sum()),
                      at_0_5=M.summary(p.label, p.score, 0.5, max_fpr),
                      at_val_threshold=M.at_threshold(p.label, p.score, thr))
    json.dump(res, open(out / "summary.json", "w"), indent=2)
    if "test" in preds:
        per_method(preds["test"], thr).to_csv(out / "per_method_test.csv", index=False)
    return res


def compare_runs(runs_dir) -> pd.DataFrame:
    """One row per evaluated run (reads eval/summary.json) -> table for the report."""
    rows = []
    for f in sorted(Path(runs_dir).glob("*/eval/summary.json")):
        r = json.load(open(f))
        t = r.get("test", {})
        rows.append(dict(run=r["run"], model=r["model"], epoch=r["epoch"],
                         test_roc_auc=t.get("at_0_5", {}).get("roc_auc"),
                         test_recall=t.get("at_val_threshold", {}).get("recall"),
                         test_fpr=t.get("at_val_threshold", {}).get("fpr"),
                         threshold=r["threshold_from_val"]))
    return pd.DataFrame(rows)
