"""Robustness, quality slices and external-dataset evaluation (notebook 05).

attack_table      score the test split under each post-processing attack (perturb.ATTACKS) at the
                  run's validation threshold -> recall (evasion) and FPR (side effect on real videos)
quality_table     FPR on real and recall on fake test videos per image-quality tertile
                  (brightness, contrast, sharpness) with Wilson 95% intervals
evaluate_external score an external fake set (e.g. FaceShifter, never used in training) against the
                  FF++ test originals at the run's validation threshold
All thresholds come from eval/summary.json (notebook 03): nothing is tuned on test.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import metrics as M
from .dataset import FrameCache, VideoEvalSet, load_index
from .models import CLIP_MODELS
from .perturb import ATTACKS, make_transform
from .train import load_trained, predict_videos


def _threshold(run_dir) -> float:
    f = Path(run_dir) / "eval" / "summary.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: evaluate the run in notebook 03 (or evaluate_run) first.")
    return json.load(open(f))["threshold_from_val"]


def wilson(k: int, n: int, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rates(pred: pd.DataFrame, thr: float) -> dict:
    flag = pred.score >= thr
    real, fake = pred.label == 0, pred.label == 1
    r = dict(roc_auc=M.ranking(pred.label, pred.score)["roc_auc"],
             recall=float(flag[fake].mean()), fpr=float(flag[real].mean()),
             n_fake=int(fake.sum()), n_real=int(real.sum()))
    r["recall_lo"], r["recall_hi"] = wilson(int(flag[fake].sum()), r["n_fake"])
    r["fpr_lo"], r["fpr_hi"] = wilson(int(flag[real].sum()), r["n_real"])
    return r


# ------------------------------------------------------------------ attacks
def attack_table(run_dir, cache_dir, attacks=None, split="test", device=None, num_workers=2,
                 save=True) -> pd.DataFrame:
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_trained(run_dir, "best", device)
    mode = "clip" if cfg["model"] in CLIP_MODELS else "frame"
    thr = _threshold(run_dir)
    idx = load_index(cache_dir, split)
    rows, preds = [], []
    for a in attacks or list(ATTACKS):
        ds = VideoEvalSet(idx, cache_dir, mode, cfg["img_size"], cfg["clip_len"], cfg["eval_frames"], make_transform(a))
        p = predict_videos(model, ds, device, num_workers)
        kind, strength = ATTACKS[a]
        r = dict(run=cfg["run_name"], attack=a, kind=kind, strength=strength, threshold=thr, **_rates(p, thr))
        for m, g in p[p.label == 1].groupby("method"):
            r[f"recall_{m}"] = float((g.score >= thr).mean())
        rows.append(r)
        preds.append(p.assign(attack=a))
        print(f"{cfg['run_name']:14s} {a:10s} ROC-AUC {r['roc_auc']:.3f}  recall {r['recall']:.3f}  FPR {r['fpr']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    if save:
        out.to_csv(run_dir / "eval" / f"robustness_{split}.csv", index=False)
        pd.concat(preds).to_csv(run_dir / "eval" / f"robustness_predictions_{split}.csv", index=False)
    return out


# ------------------------------------------------------------------ quality slices
def quality_features(idx: pd.DataFrame, cache_dir, n_frames=8) -> pd.DataFrame:
    """Per video, averaged over n_frames evenly spread face frames: brightness (mean luma),
    contrast (luma std) and sharpness (log10 variance of the Laplacian; low = blurry/upscaled face)."""
    rows = []
    for r in idx.itertuples():
        fc = FrameCache(Path(cache_dir) / r.cache_file)
        total = fc.n_clips * fc.T
        picks = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
        b, c, s = [], [], []
        for q in picks:
            g = cv2.cvtColor(fc.frame(q // fc.T, q % fc.T), cv2.COLOR_RGB2GRAY)
            b.append(g.mean()); c.append(g.std()); s.append(np.log10(cv2.Laplacian(g, cv2.CV_64F).var() + 1e-6))
        rows.append(dict(video_id=r.video_id, brightness=np.mean(b), contrast=np.mean(c), sharpness=np.mean(s)))
    return pd.DataFrame(rows)


def quality_table(pred: pd.DataFrame, feats: pd.DataFrame, thr: float, n_bins=3) -> pd.DataFrame:
    """Tertiles of each feature (cut points from the real videos) -> FPR / recall with Wilson CIs."""
    d = pred.merge(feats, on="video_id")
    rows = []
    for f in ("brightness", "contrast", "sharpness"):
        cuts = np.quantile(d.loc[d.label == 0, f], np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.digitize(d[f], cuts)
        names = ["low", "mid", "high"] if n_bins == 3 else [f"q{i + 1}" for i in range(n_bins)]
        for i in range(n_bins):
            g = d[b == i]
            r = _rates(g, thr) if g.label.nunique() == 2 else {}
            rows.append(dict(feature=f, bin=names[i], range=f"{g[f].min():.1f}–{g[f].max():.1f}" if len(g) else "",
                             n_real=int((g.label == 0).sum()), fpr=r.get("fpr"), fpr_lo=r.get("fpr_lo"),
                             fpr_hi=r.get("fpr_hi"), n_fake=int((g.label == 1).sum()), recall=r.get("recall"),
                             recall_lo=r.get("recall_lo"), recall_hi=r.get("recall_hi")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ external dataset
def external_index(ext_cache, base_cache, split="test") -> pd.DataFrame:
    """External fakes of `split` + FF++ originals of the same split. cache_file becomes absolute, so
    one VideoEvalSet can read from both caches (pass cache_dir='/')."""
    ext = load_index(ext_cache, split).assign(cache_file=lambda d: [str(Path(ext_cache) / f) for f in d.cache_file])
    base = load_index(base_cache, split)
    base = base[base.label == 0].assign(cache_file=lambda d: [str(Path(base_cache) / f) for f in d.cache_file])
    return pd.concat([base, ext], ignore_index=True)


def evaluate_external(run_dir, ext_cache, base_cache, name="FaceShifter", split="test", device=None,
                      num_workers=2) -> dict:
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_trained(run_dir, "best", device)
    mode = "clip" if cfg["model"] in CLIP_MODELS else "frame"
    thr = _threshold(run_dir)
    idx = external_index(ext_cache, base_cache, split)
    p = predict_videos(model, VideoEvalSet(idx, "/", mode, cfg["img_size"], cfg["clip_len"], cfg["eval_frames"]),
                       device, num_workers)
    p.to_csv(run_dir / "eval" / f"predictions_{name}_{split}.csv", index=False)
    summ = json.load(open(run_dir / "eval" / "summary.json"))
    res = dict(run=cfg["run_name"], dataset=name, threshold=thr, **_rates(p, thr),
               pr_auc=M.ranking(p.label, p.score)["pr_auc"],
               ffpp_test_roc_auc=summ["test"]["at_0_5"]["roc_auc"],
               ffpp_test_recall=summ["test"]["at_val_threshold"]["recall"])
    json.dump(res, open(run_dir / "eval" / f"external_{name}.json", "w"), indent=2)
    return res
