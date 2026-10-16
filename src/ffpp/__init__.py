"""Deepfake Recognition (IS 665, Team 6) – shared code for the Colab notebooks.

Stages (each writes its result to Drive, the next one builds on it):
  1. data.py    – find videos, official splits, leakage exclusion  -> manifest
  2. crops.py   – face clips per video                              -> cache/crops_<version>/
  3. train.py   – training with per-epoch checkpoints and resume    -> runs/<name>/
  4. evaluate.py– threshold on val, metrics on test, per method     -> runs/<name>/eval/
"""
__version__ = "0.1.0"
