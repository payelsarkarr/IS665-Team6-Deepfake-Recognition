"""Find FF++ videos, attach official splits and flag leakage.

Works with the official download layout
    <root>/original_sequences/youtube/c23/videos/000.mp4
    <root>/manipulated_sequences/<Method>/c23/videos/000_003.mp4
and with mirrors that name folders after the methods (e.g. the Kaggle re-upload).
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pandas as pd

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
ID_RE = re.compile(r"^(\d{3})(?:_(\d{3}))?$")
SKIP_PARTS = ("deepfakedetection", "actors", "faceshifter")


def load_official_meta(meta_dir: str | Path):
    """Official split files + YouTube mapping from the FaceForensics GitHub repo (dataset/)."""
    meta_dir = Path(meta_dir)
    split_of, pairs = {}, []
    for s in ("train", "val", "test"):
        for a, b in json.load(open(meta_dir / "splits" / f"{s}.json")):
            split_of[a] = split_of[b] = s
            pairs.append((a, b))
    conv = json.load(open(meta_dir / "conversion_dict.json"))
    youtube_of = {k: v.split()[0] for k, v in conv.items()}
    return split_of, youtube_of, pairs


def classify(path: Path, root: Path, methods=METHODS, compression="c23"):
    """(method, target_id, source_id) for a video file, or (None, reason, None)."""
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    joined = "/".join(parts)
    m = ID_RE.match(path.stem)
    if not m:
        return None, "file name not FF++ style", None
    if any(k in joined for k in SKIP_PARTS):
        return None, "subset not used (DeepFakeDetection/FaceShifter)", None
    if any(p in ({"c0", "raw", "c23", "c40"} - {compression}) for p in parts):
        return None, "other compression level", None
    target, source = m.group(1), m.group(2)
    method = next((mm for mm in methods if mm.lower() in joined), None)
    if source is None:
        return ("original", target, None) if method is None else (None, "single id in method folder", None)
    if method is None:
        return None, "method not recognised from folder name", None
    return method, target, source


def cross_split_ids(split_of: dict, youtube_of: dict) -> list[str]:
    """Val/test ids whose YouTube source video also has a sequence in another split."""
    groups = collections.defaultdict(list)
    for vid, yt in youtube_of.items():
        groups[yt].append(vid)
    bad = set()
    for ids in groups.values():
        if len({split_of.get(i) for i in ids}) > 1:
            bad |= {i for i in ids if split_of.get(i) in ("val", "test")}
    return sorted(bad)


def build_manifest(video_root, meta_dir, methods=METHODS, compression="c23",
                   max_per_class: int = 0) -> pd.DataFrame:
    """One row per usable video.

    max_per_class > 0 keeps the first N originals (by id) and, per method, only fakes whose
    target is one of them -> a closed subset with the same 1:4 ratio as the full set.
    """
    root = Path(video_root)
    split_of, youtube_of, _ = load_official_meta(meta_dir)
    bad = set(cross_split_ids(split_of, youtube_of))
    rows, skipped = [], collections.Counter()
    for p in sorted(root.rglob("*.mp4")):
        method, a, b = classify(p, root, methods, compression)
        if method is None:
            skipped[a] += 1
            continue
        vid = a if method == "original" else f"{method}/{a}_{b}"
        rows.append(dict(video_id=vid, path=str(p), method=method, label=int(method != "original"),
                         target_id=a, source_id=b, split=split_of.get(a),
                         youtube_id=youtube_of.get(a),
                         leak_excluded=(a in bad) or (b in bad if b else False)))
    man = pd.DataFrame(rows)
    if man.empty:
        raise SystemExit(f"No usable videos under {root}. Skipped: {dict(skipped)}")
    man = man.drop_duplicates("video_id").reset_index(drop=True)
    if max_per_class:
        keep = set(sorted(man.loc[man.method == "original", "target_id"])[:max_per_class])
        man = man[man.target_id.isin(keep)].reset_index(drop=True)
    man.attrs["skipped"] = dict(skipped)
    return man


def summary(man: pd.DataFrame) -> pd.DataFrame:
    """Videos per split and method (leakage-excluded val/test videos not counted)."""
    use = man[~man.leak_excluded | (man.split == "train")]
    return use.groupby(["split", "method"]).size().unstack(fill_value=0)
