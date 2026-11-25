"""Grad-CAM (Selvaraju et al., 2017): which image regions push the score towards "fake".

Works for all three models: the heatmap is taken at the last convolutional block (MobileNetV2:
backbone[-1], 1280 x 7 x 7; small CNN: last conv block). For the GRU model the gradient flows
back through the GRU into every frame, so each frame of the clip gets its own map.

    cams, p = gradcam(model, x)          # x: (N, 3, H, W) frames or (1, T, 3, H, W) one clip
    img = overlay(face_rgb, cams[0])     # uint8 RGB with the heatmap blended in
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def target_layer(model):
    if hasattr(model, "backbone"):
        return model.backbone[-1]
    return model.features[-2]          # SmallCNN: last ReLU before the final max-pool


def gradcam(model, x: torch.Tensor):
    """Returns (cams, p): cams (N_frames, H, W) in [0, 1], p = P(fake) per sample."""
    acts, grads = {}, {}

    def fwd(_m, _i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h = target_layer(model).register_forward_hook(fwd)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            logit = model(x)
            model.zero_grad(set_to_none=True)
            logit.sum().backward()
    finally:
        h.remove()
        model.train(was_training)
    a, g = acts["a"].detach(), grads["g"].detach()                 # (N_frames, C, h, w)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    return cam.cpu().numpy(), torch.sigmoid(logit.detach().float()).cpu().numpy().ravel()


def overlay(face_rgb: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap (any size) over an RGB face crop."""
    h, w = face_rgb.shape[:2]
    cam = cv2.resize(cam.astype(np.float32), (w, h))
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * face_rgb + alpha * heat, 0, 255).astype(np.uint8)


def explain_frames(model, frames: np.ndarray, img_size: int, clip_mode: bool, device):
    """frames: (N, H, W, 3) uint8 RGB faces. Frame model: one map per frame. Clip model: frames are
    one clip, one map per frame. Returns (overlays, cams, p)."""
    from .dataset import to_tensor
    x = to_tensor(frames, img_size).to(device)
    if clip_mode:
        x = x[None]
    cams, p = gradcam(model, x)
    faces = [cv2.resize(f, (img_size, img_size)) if f.shape[0] != img_size else f for f in frames]
    return [overlay(f, c) for f, c in zip(faces, cams)], cams, p
