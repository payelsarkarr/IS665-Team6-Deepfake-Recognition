"""Post-processing an attacker (or a sharing platform) can apply to a video without any access to
the model: re-compression, downscaling, blur, noise. Applied to the 224 px face frames.

    apply(frames, "jpeg_30")         one fixed attack, for the robustness test
    random_perturb(frames, rng, ...)  random kind and strength, for perturbation-augmented training

frames: (N, H, W, 3) uint8 RGB. The same perturbation is applied to every frame of a sample.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

# name -> (kind, strength). "none" is the clean reference.
ATTACKS = {
    "none": ("none", 0),
    "jpeg_50": ("jpeg", 50), "jpeg_30": ("jpeg", 30), "jpeg_15": ("jpeg", 15),
    "down_0.5": ("down", 0.5), "down_0.25": ("down", 0.25),
    "blur_1.5": ("blur", 1.5), "blur_3": ("blur", 3.0),
    "noise_5": ("noise", 5.0), "noise_10": ("noise", 10.0),
}
KINDS = ("jpeg", "down", "blur", "noise")


def _jpeg(frames, q):
    return np.stack([cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1], cv2.IMREAD_UNCHANGED)
                     for f in np.ascontiguousarray(frames)])


def _down(frames, f):
    h, w = frames.shape[1:3]
    small = (max(8, int(w * f)), max(8, int(h * f)))
    return np.stack([cv2.resize(cv2.resize(x, small, interpolation=cv2.INTER_AREA), (w, h),
                                interpolation=cv2.INTER_LINEAR) for x in frames])


def _blur(frames, sigma):
    return np.stack([cv2.GaussianBlur(x, (0, 0), sigma) for x in frames])


def _noise(frames, std, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(frames.astype(np.float32) + rng.normal(0, std, frames.shape), 0, 255).astype(np.uint8)


def perturb(frames: np.ndarray, kind: str, strength, seed=0) -> np.ndarray:
    if kind == "none":
        return frames
    if kind == "jpeg":
        return _jpeg(frames, strength)
    if kind == "down":
        return _down(frames, strength)
    if kind == "blur":
        return _blur(frames, strength)
    if kind == "noise":
        return _noise(frames, strength, seed)
    raise ValueError(f"unknown perturbation {kind}")


def apply(frames: np.ndarray, attack: str) -> np.ndarray:
    kind, strength = ATTACKS[attack]
    return perturb(frames, kind, strength)


def make_transform(attack: str):
    """Callable for VideoEvalSet(transform=...), or None for the clean data."""
    return None if attack == "none" else (lambda frames: apply(frames, attack))


# training ranges cover the test levels of the kinds used for training
TRAIN_RANGES = {"jpeg": (15, 60), "down": (0.2, 0.7), "blur": (0.8, 3.5), "noise": (3.0, 12.0)}


def random_perturb(frames: np.ndarray, rng: random.Random, kinds=("jpeg", "down", "blur")) -> np.ndarray:
    """One random kind and strength from `kinds`, the same for all frames of the sample."""
    kind = rng.choice(list(kinds))
    lo, hi = TRAIN_RANGES[kind]
    s = rng.randint(int(lo), int(hi)) if kind == "jpeg" else rng.uniform(lo, hi)
    return perturb(frames, kind, s, seed=rng.randrange(1 << 30))
