"""Final evaluation of a run: threshold chosen on validation, applied unchanged to test.

runs/<run_name>/eval/
  summary.json            val + test metrics (at 0.5 and at the chosen threshold, incl. confusion matrix)
  per_method_test.csv     per method: share flagged (recall; FPR for 'original') + ROC-AUC vs. originals
  predictions_<split>.csv per-video scores

Leave-one-method-out runs (cfg['exclude_methods']): the threshold is chosen on validation WITHOUT the
held-out method, so nothing about it leaks into the decision rule. Test then reports seen vs. held-out
methods separately (summary.json -> 'test_seen', 'test_heldout').
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from . import metrics as M
from .dataset import VideoEvalSet, load_index
from .models import CLIP_MODELS
from .train import drop_methods, load_trained, predict_videos


def per_method(pred: pd.DataFrame, thr: float, heldout=()) -> pd.DataFrame:
    """Share flagged as fake per method (recall for fake methods, FPR for 'original') and the
    ROC-AUC of each method against the original videos."""
    g = pred.assign(flagged=pred.score >= thr).groupby("method")
    out = pd.DataFrame({"videos": g.size(), "flagged_share": g.flagged.mean().round(4)}).reset_index()
    real = pred[pred.label == 0]
    out["roc_auc_vs_original"] = [
        round(M.ranking(pd.concat([real, pred[pred.method == m]]).label,
                        pd.concat([real, pred[pred.method == m]]).score)["roc_auc"], 4) if m != "original" else None
        for m in out.method]
    out["seen_in_training"] = [m == "original" or m not in heldout for m in out.method]
    return out


def _block(p: pd.DataFrame, thr: float, max_fpr: float) -> dict:
    return dict(n=len(p), n_real=int((p.label == 0).sum()), at_0_5=M.summary(p.label, p.score, 0.5, max_fpr),
                at_val_threshold=M.at_threshold(p.label, p.score, thr))


def evaluate_run(run_dir, cache_dir, which="best", splits=("val", "test"), max_fpr=0.2, device=None,
                 num_workers=2) -> dict:
    run_dir, out = Path(run_dir), Path(run_dir) / "eval"
    out.mkdir(exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch = load_trained(run_dir, which, device)
    heldout = sorted(cfg.get("exclude_methods") or [])
    mode = "clip" if cfg["model"] in CLIP_MODELS else "frame"
    preds = {}
    for s in splits:
        ds = VideoEvalSet(load_index(cache_dir, s), cache_dir, mode, cfg["img_size"], cfg["clip_len"], cfg["eval_frames"])
        preds[s] = predict_videos(model, ds, device, num_workers)
        preds[s].to_csv(out / f"predictions_{s}.csv", index=False)
    if "val" in preds:
        v = drop_methods(preds["val"], heldout)          # held-out method must not shape the threshold
        thr = M.threshold_for_fpr(v.label, v.score, max_fpr)
    else:
        thr = 0.5
    res = dict(run=cfg["run_name"], model=cfg["model"], checkpoint=which, epoch=epoch + 1,
               threshold_from_val=thr, max_fpr=max_fpr, heldout_methods=heldout)
    for s, p in preds.items():
        res[s] = _block(p, thr, max_fpr)
    if heldout and "test" in preds:
        t = preds["test"]
        res["test_seen"] = _block(drop_methods(t, heldout), thr, max_fpr)
        res["test_heldout"] = _block(t[(t.label == 0) | t.method.isin(heldout)], thr, max_fpr)
    json.dump(res, open(out / "summary.json", "w"), indent=2)
    if "test" in preds:
        per_method(preds["test"], thr, heldout).to_csv(out / "per_method_test.csv", index=False)
    return res


def compare_runs(runs_dir) -> pd.DataFrame:
    """One row per evaluated run (reads eval/summary.json) -> table for the report."""
    rows = []
    for f in sorted(Path(runs_dir).glob("*/eval/summary.json")):
        r = json.load(open(f))
        t = r.get("test", {})
        h = r.get("test_heldout", {})
        rows.append(dict(run=r["run"], model=r["model"], epoch=r["epoch"],
                         heldout=",".join(r.get("heldout_methods", [])) or "-",
                         test_roc_auc=t.get("at_0_5", {}).get("roc_auc"),
                         test_recall=t.get("at_val_threshold", {}).get("recall"),
                         test_fpr=t.get("at_val_threshold", {}).get("fpr"),
                         heldout_roc_auc=h.get("at_0_5", {}).get("roc_auc"),
                         heldout_recall=h.get("at_val_threshold", {}).get("recall"),
                         threshold=r["threshold_from_val"]))
    return pd.DataFrame(rows)
