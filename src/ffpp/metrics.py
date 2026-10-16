"""Video-level metrics. Label 1 = fake (the positive class), score = P(fake)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def at_threshold(y, p, thr=0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(threshold=float(thr), recall=rec, fpr=fp / (fp + tn) if fp + tn else float("nan"),
                precision=prec, f1=2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
                accuracy=(tp + tn) / len(y), tp=tp, fp=fp, fn=fn, tn=tn)


def ranking(y, p) -> dict:
    y = np.asarray(y).astype(int)
    if len(set(y)) < 2:
        return dict(roc_auc=float("nan"), pr_auc=float("nan"))
    return dict(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))


def threshold_for_fpr(y, p, max_fpr=0.2) -> float:
    """Lowest threshold whose false-positive rate stays <= max_fpr (= highest recall under that limit).
    Chosen on validation only, then applied unchanged to test."""
    y = np.asarray(y).astype(int)
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= max_fpr)[0]
    t = thr[ok[-1]]
    return float(min(t, 1.0))


def recall_at_fpr(y, p, max_fpr=0.2) -> float:
    y = np.asarray(y).astype(int)
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= max_fpr].max())


def summary(y, p, thr=0.5, max_fpr=0.2) -> dict:
    return {**ranking(y, p), **at_threshold(y, p, thr), f"recall_at_fpr{max_fpr}": recall_at_fpr(y, p, max_fpr)}
