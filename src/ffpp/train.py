"""Training with a checkpoint after every epoch and automatic resume.

runs/<run_name>/
  config.json        settings of this run (architecture keys must not change on resume)
  metrics.csv        one row per epoch (train loss, val ROC-AUC, recall, FPR, ...)
  last.pt            model + optimizer + epoch -> resume point, overwritten every epoch
  best.pt            weights of the epoch with the best val ROC-AUC
  val_best.csv       per-video val scores of the best epoch

Run train(cfg, ...) again with the same run_name and it continues after the last finished
epoch. Raise cfg["epochs"] to train longer. Change model / img_size / clip_len -> new run_name.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import metrics as M
from .dataset import TrainSet, VideoEvalSet, load_index
from .models import CLIP_MODELS, build_model, count_params, set_backbone_trainable

DEFAULT_CFG = dict(
    run_name="b2_cnn_v1", model="b2_cnn", pretrained=True, img_size=224, clip_len=16,
    epochs=10, batch_size=64, lr=1e-3, weight_decay=1e-4, warmup_epochs=1, freeze_backbone_epochs=0,
    samples_per_video=4, eval_frames=8, num_workers=2, seed=665, amp=True, patience=0,
    max_train_videos=0, max_fpr=0.2, exclude_methods=(),
)
# must not change when a run resumes (architecture + which methods the model may see)
ARCH_KEYS = ("model", "pretrained", "img_size", "clip_len", "exclude_methods")


def drop_methods(idx: pd.DataFrame, methods) -> pd.DataFrame:
    """Leave-one-method-out: remove the held-out manipulation method(s) from an index."""
    return idx[~idx.method.isin(list(methods or []))].reset_index(drop=True)



def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _lr_at(cfg, epoch):
    """Warm-up, then cosine decay; computed from the epoch alone, so resuming or raising
    the number of epochs needs no scheduler state."""
    if epoch < cfg["warmup_epochs"]:
        return cfg["lr"] * (epoch + 1) / (cfg["warmup_epochs"] + 1)
    span = max(1, cfg["epochs"] - cfg["warmup_epochs"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (epoch - cfg["warmup_epochs"]) / span))


@torch.no_grad()
def predict_videos(model, eval_set: VideoEvalSet, device, num_workers=2) -> pd.DataFrame:
    """Mean P(fake) over all evaluation frames / clips of each video."""
    model.eval()
    dl = DataLoader(eval_set, batch_size=None, shuffle=False, num_workers=num_workers)
    rows = []
    for x, y, vid in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device)).float())
        rows.append(dict(video_id=vid, label=int(y), score=float(p.mean()), score_max=float(p.max())))
    out = pd.DataFrame(rows)
    return out.merge(eval_set.idx[["video_id", "method"]], on="video_id")


def _check_config(run_dir: Path, cfg: dict):
    f = run_dir / "config.json"
    if f.exists():
        old = json.load(open(f))
        norm = lambda d, k: sorted(d.get(k) or []) if k == "exclude_methods" else d.get(k)   # old runs: missing = []
        diff = {k: (old.get(k), cfg.get(k)) for k in ARCH_KEYS if norm(old, k) != norm(cfg, k)}
        if diff:
            raise SystemExit(f"Run '{cfg['run_name']}' exists with a different architecture {diff}. "
                             "Use a new run_name (or delete the old run folder).")
    json.dump(cfg, open(f, "w"), indent=2)


def train(cfg: dict, cache_dir, runs_dir, device=None, log_every=100) -> pd.DataFrame:
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["exclude_methods"] = sorted(cfg["exclude_methods"] or [])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(runs_dir) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_config(run_dir, cfg)
    mode = "clip" if cfg["model"] in CLIP_MODELS else "frame"

    # ---- data
    tr_idx, va_idx = load_index(cache_dir, "train"), load_index(cache_dir, "val")
    if cfg["exclude_methods"]:   # LOMO: the held-out method is unseen in training AND in model selection
        tr_idx, va_idx = drop_methods(tr_idx, cfg["exclude_methods"]), drop_methods(va_idx, cfg["exclude_methods"])
        print("held out (never seen in train/val):", ", ".join(cfg["exclude_methods"]))
    if cfg["max_train_videos"]:
        tr_idx = tr_idx.sample(min(cfg["max_train_videos"], len(tr_idx)), random_state=cfg["seed"])
    n_real, n_fake = int((tr_idx.label == 0).sum()), int((tr_idx.label == 1).sum())
    w_real, w_fake = len(tr_idx) / (2 * max(n_real, 1)), len(tr_idx) / (2 * max(n_fake, 1))  # balance 1:4
    tr_set = TrainSet(tr_idx, cache_dir, mode, cfg["img_size"], cfg["clip_len"], cfg["samples_per_video"], True, cfg["seed"])
    va_set = VideoEvalSet(va_idx, cache_dir, mode, cfg["img_size"], cfg["clip_len"], cfg["eval_frames"])
    bs = cfg["batch_size"] if mode == "frame" else max(1, cfg["batch_size"] // cfg["clip_len"])
    print(f"train: {len(tr_idx)} videos ({n_real} real, {n_fake} fake) | val: {len(va_idx)} videos | "
          f"mode={mode}, batch={bs}, device={device}")

    # ---- model + optimizer
    model = build_model(cfg["model"], pretrained=cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    print(f"{cfg['model']}: {count_params(model) / 1e6:.2f} M parameters")

    # ---- resume
    start, best, best_epoch, stale = 0, -1.0, -1, 0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start, best, best_epoch, stale = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck.get("stale", 0)
        print(f"resuming after epoch {ck['epoch'] + 1} (best val ROC-AUC {best:.4f} in epoch {best_epoch + 1})")
    if start >= cfg["epochs"]:
        print(f"already trained {start} epochs; raise cfg['epochs'] to continue")
    hist_file = run_dir / "metrics.csv"

    for epoch in range(start, cfg["epochs"]):
        t0 = time.time()
        lr = _lr_at(cfg, epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        set_backbone_trainable(model, epoch >= cfg["freeze_backbone_epochs"])
        torch.manual_seed(cfg["seed"] + epoch)
        tr_set.set_epoch(epoch)
        dl = DataLoader(tr_set, batch_size=bs, shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=device.type == "cuda", drop_last=len(tr_set) > bs)
        model.train()
        if epoch < cfg["freeze_backbone_epochs"] and hasattr(model, "backbone"):
            model.backbone.eval()          # frozen: keep its BatchNorm statistics from ImageNet
        tot, n = 0.0, 0
        for b, (x, y) in enumerate(dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device)
            w = torch.where(y > 0.5, torch.full_like(y, w_fake), torch.full_like(y, w_real))
            with torch.autocast(device_type=device.type, enabled=cfg["amp"] and device.type == "cuda"):
                logit = model(x)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot, n = tot + loss.item() * len(y), n + len(y)
            if b % log_every == 0:
                print(f"  epoch {epoch + 1} batch {b}/{len(dl)} loss {tot / n:.4f}", flush=True)

        # ---- validation per video
        val = predict_videos(model, va_set, device, cfg["num_workers"])
        s = M.summary(val.label, val.score, 0.5, cfg["max_fpr"])
        score = s["roc_auc"] if not np.isnan(s["roc_auc"]) else -1.0
        row = dict(epoch=epoch + 1, lr=lr, train_loss=tot / max(n, 1), val_roc_auc=s["roc_auc"], val_pr_auc=s["pr_auc"],
                   val_recall=s["recall"], val_fpr=s["fpr"], val_recall_at_fpr=s[f"recall_at_fpr{cfg['max_fpr']}"],
                   minutes=(time.time() - t0) / 60)
        pd.DataFrame([row]).to_csv(hist_file, mode="a", header=not hist_file.exists(), index=False)
        improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            _atomic_save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg, "val": s}, run_dir / "best.pt")
            val.to_csv(run_dir / "val_best.csv", index=False)
        else:
            stale += 1
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                      "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale, "cfg": cfg}, last)
        print(f"epoch {epoch + 1}/{cfg['epochs']}: loss {row['train_loss']:.4f} | val ROC-AUC {s['roc_auc']:.4f} "
              f"recall {s['recall']:.3f} FPR {s['fpr']:.3f} | {row['minutes']:.1f} min"
              + ("  * best, saved" if improved else ""), flush=True)
        if cfg["patience"] and stale >= cfg["patience"]:
            print(f"early stop: no improvement for {stale} epochs")
            break
    return pd.read_csv(hist_file) if hist_file.exists() else pd.DataFrame()


def load_trained(run_dir, which="best", device=None):
    """Model + config of a finished run ('best' or 'last')."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(run_dir) / f"{which}.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"], pretrained=False).to(device)   # weights come from the checkpoint
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["epoch"]
