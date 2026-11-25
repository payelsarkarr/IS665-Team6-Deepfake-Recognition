"""Fast checks of the pipeline logic on synthetic data (no FF++ videos needed, CPU only)."""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from ffpp import metrics as M
from ffpp import perturb
from ffpp.data import classify, cross_split_ids
from ffpp.dataset import TrainSet, VideoEvalSet, to_tensor
from ffpp.download import split_pairs
from ffpp.models import build_model
from ffpp.robustness import quality_table, wilson
from ffpp.train import _lr_at, drop_methods
from ffpp.xai import gradcam, overlay


# ---------------------------------------------------------------- helpers
def write_cache(path: Path, n_clips=2, T=4, size=64, seed=0):
    """A cache file in the crops.py layout with random face-sized images."""
    rng = np.random.default_rng(seed)
    blobs = [cv2.imencode(".jpg", rng.integers(0, 255, (size, size, 3), np.uint8))[1].ravel()
             for _ in range(n_clips * T)]
    offsets = np.concatenate([[0], np.cumsum([b.size for b in blobs])]).astype(np.int64)
    np.savez(path, jpeg=np.concatenate(blobs), offsets=offsets, meta=np.array([n_clips, T, size]),
             starts=np.arange(n_clips) * 10)


@pytest.fixture
def cache(tmp_path):
    (tmp_path / "videos").mkdir()
    rows = []
    for i, (m, lab) in enumerate([("original", 0), ("original", 0), ("Deepfakes", 1), ("FaceSwap", 1)]):
        f = f"videos/v{i}.npz"
        write_cache(tmp_path / f, seed=i)
        rows.append(dict(video_id=f"v{i}", method=m, label=lab, cache_file=f, split="test", leak_excluded=False,
                         status="ok", n_clips=2, target_id=f"{i:03d}", source_id=None))
    pd.DataFrame(rows).to_csv(tmp_path / "index.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------- data
def test_classify_layout(tmp_path):
    root = tmp_path
    assert classify(root / "original_sequences/youtube/c23/videos/000.mp4", root) == ("original", "000", None)
    assert classify(root / "manipulated_sequences/Deepfakes/c23/videos/000_003.mp4", root) == ("Deepfakes", "000", "003")
    assert classify(root / "manipulated_sequences/FaceShifter/c23/videos/000_003.mp4", root)[0] is None
    fs = classify(root / "manipulated_sequences/FaceShifter/c23/videos/000_003.mp4", root,
                  methods=["FaceShifter"], skip=("deepfakedetection", "actors"))
    assert fs == ("FaceShifter", "000", "003")
    assert classify(root / "manipulated_sequences/Deepfakes/c40/videos/000_003.mp4", root)[0] is None


def test_cross_split_ids():
    split_of = {"001": "train", "002": "test", "003": "test", "004": "val"}
    youtube_of = {"001": "A", "002": "A", "003": "B", "004": "C"}
    assert cross_split_ids(split_of, youtube_of) == ["002"]


def test_split_pairs(tmp_path):
    (tmp_path / "splits").mkdir()
    json.dump([["001", "002"], ["003", "004"]], open(tmp_path / "splits" / "test.json", "w"))
    assert split_pairs(tmp_path, "test", exclude_ids={"003"}) == ["001_002", "002_001"]


def test_drop_methods():
    idx = pd.DataFrame(dict(method=["original", "FaceSwap", "Deepfakes"]))
    assert list(drop_methods(idx, ["FaceSwap"]).method) == ["original", "Deepfakes"]


# ---------------------------------------------------------------- metrics
def test_threshold_respects_fpr_limit():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(800)]
    p = np.r_[rng.normal(0.3, 0.15, 200), rng.normal(0.7, 0.15, 800)]
    thr = M.threshold_for_fpr(y, p, 0.2)
    assert M.at_threshold(y, p, thr)["fpr"] <= 0.2
    assert M.at_threshold(y, p, thr)["recall"] == pytest.approx(M.recall_at_fpr(y, p, 0.2))


def test_confusion_counts():
    r = M.at_threshold([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 1, 1, 1)


def test_wilson_interval_contains_estimate():
    lo, hi = wilson(30, 138)
    assert lo < 30 / 138 < hi and 0 <= lo and hi <= 1


def test_lr_schedule_resumable():
    cfg = dict(lr=1e-3, warmup_epochs=1, epochs=10)
    assert _lr_at(cfg, 0) < cfg["lr"] and _lr_at(cfg, 1) == pytest.approx(cfg["lr"]) and _lr_at(cfg, 9) < 1e-4


# ---------------------------------------------------------------- perturbations
@pytest.mark.parametrize("attack", list(perturb.ATTACKS))
def test_attacks_keep_shape(attack):
    f = np.random.default_rng(0).integers(0, 255, (3, 64, 64, 3), np.uint8)
    out = perturb.apply(f, attack)
    assert out.shape == f.shape and out.dtype == np.uint8
    if attack != "none":
        assert not np.array_equal(out, f)


def test_random_perturb_uses_training_kinds_only():
    f = np.full((2, 32, 32, 3), 128, np.uint8)
    for s in range(20):
        out = perturb.random_perturb(f, random.Random(s), kinds=("blur",))
        assert np.array_equal(out, f)                 # blur of a flat image changes nothing


# ---------------------------------------------------------------- datasets + models
def test_eval_set_and_transform(cache):
    idx = pd.read_csv(cache / "index.csv")
    clean = VideoEvalSet(idx, cache, "frame", 64, 4, 4)[0][0]
    noisy = VideoEvalSet(idx, cache, "frame", 64, 4, 4, perturb.make_transform("noise_10"))[0][0]
    assert clean.shape == (4, 3, 64, 64) and not torch.equal(clean, noisy)
    clip = VideoEvalSet(idx, cache, "clip", 64, 4, 4)[0][0]
    assert clip.shape == (2, 4, 3, 64, 64)


def test_train_set_robust_aug(cache):
    idx = pd.read_csv(cache / "index.csv")
    ds = TrainSet(idx, cache, "frame", 64, 4, samples_per_video=2, robust_aug=1.0)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and y.item() in (0.0, 1.0) and len(ds) == 8


@pytest.mark.parametrize("name,shape", [("b2_cnn", (2, 3, 64, 64)), ("mnv2", (2, 3, 64, 64)),
                                        ("mnv2_gru", (1, 3, 3, 64, 64))])
def test_models_and_gradcam(name, shape):
    model = build_model(name, pretrained=False).eval()
    x = torch.randn(*shape)
    assert model(x).shape == (shape[0],)
    cams, p = gradcam(model, x)
    n_frames = shape[0] if len(shape) == 4 else shape[1]
    assert cams.shape == (n_frames, 64, 64) and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert len(p) == shape[0]
    img = overlay(np.zeros((64, 64, 3), np.uint8), cams[0])
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8


def test_to_tensor_normalises():
    x = to_tensor(np.full((1, 32, 32, 3), 255, np.uint8), 16)
    assert x.shape == (1, 3, 16, 16) and x.max() > 2


def test_quality_table():
    rng = np.random.default_rng(1)
    pred = pd.DataFrame(dict(video_id=[f"v{i}" for i in range(60)], label=[0] * 30 + [1] * 30,
                             score=rng.random(60)))
    feats = pd.DataFrame(dict(video_id=pred.video_id, brightness=rng.random(60) * 255,
                              contrast=rng.random(60) * 60, sharpness=rng.random(60) * 3))
    t = quality_table(pred, feats, 0.5)
    assert len(t) == 9 and t.n_real.sum() == 90
