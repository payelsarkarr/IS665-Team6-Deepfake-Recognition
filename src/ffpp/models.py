"""Models. Every model returns one logit per sample (> 0 means "fake").

b2_cnn    small CNN from scratch, one frame            input (B, 3, H, W)
mnv2      MobileNetV2 (ImageNet weights), one frame    input (B, 3, H, W)
mnv2_gru  MobileNetV2 per frame + GRU over the frames  input (B, T, 3, H, W)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

FRAME_MODELS = {"b2_cnn", "mnv2"}
CLIP_MODELS = {"mnv2_gru"}


class SmallCNN(nn.Module):
    """Baseline B2: four conv blocks (32-64-128-256), global average pooling, one logit."""

    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in (width, width * 2, width * 4, width * 8):
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def _mobilenet_backbone(pretrained: bool):
    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    return tvm.mobilenet_v2(weights=weights).features          # output (B, 1280, h, w)


class MobileNetFrame(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = _mobilenet_backbone(pretrained)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(1280, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class MobileNetGRU(nn.Module):
    """Frame features (1280 per frame) -> bidirectional GRU -> mean over time -> logit.
    The GRU sees how the face features change from frame to frame."""

    def __init__(self, pretrained=True, hidden=256, dropout=0.3):
        super().__init__()
        self.backbone = _mobilenet_backbone(pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gru = nn.GRU(1280, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.pool(self.backbone(x.flatten(0, 1))).flatten(1).view(B, T, -1)   # (B, T, 1280)
        h, _ = self.gru(f)
        return self.head(h.mean(1)).squeeze(1)


def build_model(name: str, pretrained=True, **kw) -> nn.Module:
    if name == "b2_cnn":
        return SmallCNN(**kw)
    if name == "mnv2":
        return MobileNetFrame(pretrained=pretrained, **kw)
    if name == "mnv2_gru":
        return MobileNetGRU(pretrained=pretrained, **kw)
    raise ValueError(f"unknown model {name}")


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Freeze/unfreeze the pretrained part (used for a warm-up phase with only the head training)."""
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = trainable


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
