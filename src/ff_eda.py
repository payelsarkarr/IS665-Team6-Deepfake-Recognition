#!/usr/bin/env python3
"""
FaceForensics++ data audit, EDA and preliminary baselines (Interim Progress Report).

Layout: found automatically. Works with the official download layout
  <FF_ROOT>/original_sequences/youtube/c23/videos/000.mp4
  <FF_ROOT>/manipulated_sequences/<Method>/c23/videos/000_003.mp4
and with mirrors that use other folder names (method taken from the folder name,
original vs. fake from the file name 000.mp4 / 000_003.mp4).
Official split files + conversion_dict.json come from the FaceForensics GitHub repo
(dataset/splits/{train,val,test}.json, dataset/conversion_dict.json).

Usage:
  python ff_eda.py --ff_root /data/ff++ --meta_dir FaceForensics/dataset --out out \
      [--methods Deepfakes Face2Face FaceSwap NeuralTextures] [--clips 2 --clip_len 16]
      [--limit 50]   # quick test on a few videos per method

Outputs (in --out): manifest.csv, clips.csv, quality_report.json, baseline_results.json,
fig1_class_split.png, fig2_video_properties.png, fig3_temporal_instability.png,
fig4_example_crops.png
"""
import argparse, collections, hashlib, json, os, re, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, accuracy_score)

DEFAULT_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
# fixed categorical order (colorblind-validated palette), colour follows the entity
COLORS = {"original": "#2a78d6", "Deepfakes": "#eb6834", "Face2Face": "#1baf7a",
          "FaceSwap": "#eda100", "NeuralTextures": "#e87ba4", "FaceShifter": "#008300"}
SEED = 665
FEATURES = ["diff_mean", "diff_std", "diff_max", "diff_p95_ratio", "sharp_mean",
            "sharp_std", "face_rate"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150})


# ----------------------------------------------------------------------------- manifest
def load_splits(meta_dir):
    split_of = {}
    for s in ("train", "val", "test"):
        for a, b in json.load(open(Path(meta_dir) / "splits" / f"{s}.json")):
            split_of[a] = s
            split_of[b] = s
    conv = json.load(open(Path(meta_dir) / "conversion_dict.json"))
    yt_of = {k: v.split()[0] for k, v in conv.items()}
    return split_of, yt_of


ID_RE = re.compile(r"^(\d{3})(?:_(\d{3}))?$")
ORIGINAL_ALIASES = ("original", "youtube", "real", "pristine")
SKIP_PARTS = ("deepfakedetection", "actors", "dfd")


def classify(path, root, methods, comp):
    """Return (method, target, source) for a video path, or (None, reason, None).
    Works for the official layout and for re-uploads/mirrors with other folder names."""
    parts = [x.lower() for x in path.relative_to(root).parts[:-1]]   # relative: root folder names must not count
    joined = "/".join(parts)
    m = ID_RE.match(path.stem)
    if not m:
        return None, "filename not FF++ style", None
    if any(k in joined for k in SKIP_PARTS):
        return None, "DeepFakeDetection subset (not used)", None
    other_comp = {"c0", "raw", "c23", "c40"} - {comp}
    if any(p in other_comp for p in parts):
        return None, "other compression level", None
    target, source = m.group(1), m.group(2)
    method = next((mm for mm in methods if mm.lower() in joined), None)
    if source is None:
        if method is None and any(a in joined for a in ORIGINAL_ALIASES):
            return "original", target, None
        if method is None:
            return "original", target, None      # NNN.mp4 outside any method folder
        return None, "single id inside a method folder", None
    if method is None:
        return None, "pair id but method not recognised from folder name", None
    return method, target, source


def build_manifest(ff_root, methods, comp, split_of, yt_of, limit, limit_original=0):
    """Scan ff_root recursively; classify every .mp4. limit = max fakes per method,
    limit_original = max originals (0 = all)."""
    found, skipped = collections.defaultdict(list), collections.Counter()
    for p in sorted(Path(ff_root).rglob("*.mp4")):
        method, a, b = classify(p, Path(ff_root), methods, comp)
        if method is None:
            skipped[a] += 1
        else:
            found[method].append((p, a, b))
    rows = []
    for method, items in found.items():
        cap = limit_original if method == "original" else limit
        seen = set()
        for p, target, source in items:
            key = p.stem
            if key in seen:          # same file present twice in the upload
                skipped["duplicate filename across folders"] += 1
                continue
            seen.add(key)
            if cap and len(seen) > cap:
                break
            rows.append(dict(video_id=key if method == "original" else f"{method}/{key}",
                             path=str(p), method=method, label=int(method != "original"),
                             target_id=target, source_id=source,
                             split=split_of.get(target), source_split=split_of.get(source) if source else None,
                             youtube_id=yt_of.get(target)))
    man = pd.DataFrame(rows)
    man.attrs["skipped"] = dict(skipped)
    return man


def completeness_vs_official(man, meta_dir, methods):
    """Compare the files present with the official FF++ file list (from the split files):
    1,000 originals and 1,000 fakes per method named target_source in both directions."""
    pairs = [p for s in ("train", "val", "test") for p in json.load(open(Path(meta_dir) / "splits" / f"{s}.json"))]
    expected = {"original": {x for p in pairs for x in p}}
    for m in methods:
        expected[m] = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
    out = {}
    for m, exp in expected.items():
        have = set(man.loc[man.method == m, "video_id"].str.split("/").str[-1])
        out[m] = dict(expected=len(exp), present=len(have & exp), not_in_official_list=len(have - exp))
    return out


def flag_cross_split_sources(man, split_of, yt_of):
    """Some FF++ sequences were cut from the same YouTube video (977 sources -> 1,000 ids).
    If such a group spans splits, the same person/scene is in train and val/test.
    We keep the train copy and exclude the val/test id plus every fake that uses it
    as target or source."""
    groups = collections.defaultdict(list)
    for vid, y in yt_of.items():
        groups[y].append(vid)
    bad = set()
    for ids in groups.values():
        if len({split_of.get(i) for i in ids}) > 1:
            bad |= {i for i in ids if split_of.get(i) in ("val", "test")}
    man["leak_excluded"] = man.target_id.isin(bad) | man.source_id.isin(bad)
    return man, sorted(bad)


# ----------------------------------------------------------------------------- per video
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dhash(gray, size=8):
    g = cv2.resize(gray, (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


if not hasattr(cv2, "CascadeClassifier"):
    sys.exit("This OpenCV build has no Haar cascades (OpenCV 5 removed them). "
             "Install OpenCV 4.x:  pip install 'opencv-python<5'")
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face(gray, max_side=640):
    """Haar detection on a downscaled copy (much faster on HD frames); box scaled back."""
    h0, w0 = gray.shape[:2]
    f = min(1.0, max_side / max(h0, w0))
    small = cv2.resize(gray, (int(w0 * f), int(h0 * f)), interpolation=cv2.INTER_AREA) if f < 1 else gray
    faces = _CASCADE.detectMultiScale(small, 1.1, 5, minSize=(24, 24))
    if len(faces) == 0:
        return None, 0
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # largest face = main face
    return tuple(int(round(v / f)) for v in (x, y, w, h)), len(faces)


def expand_box(box, W, H, scale=1.3):
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * scale
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    return x0, y0, x1, y1


def analyse_video(path, n_clips, clip_len, crop=112):
    """Video metadata + clip-level temporal features.
    One face box per clip (detected on the clip's first frame) is held fixed for all
    frames of the clip, so frame-to-frame differences reflect what changes inside
    the face region (manipulation flicker, motion) and not jitter of our own cropper."""
    info = dict(readable=False, n_frames=0, fps=np.nan, width=np.nan, height=np.nan,
                duration_s=np.nan, file_mb=os.path.getsize(path) / 1e6, sha256=sha256(path),
                dhash_mid=None, multi_face_rate=np.nan)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return info, [], None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info.update(readable=n > 0, n_frames=n, fps=fps, width=W, height=H,
                duration_s=n / fps if fps else np.nan)
    if n < clip_len:
        cap.release()
        return info, [], None
    starts = np.linspace(0, n - clip_len, n_clips + 2)[1:-1].astype(int) if n_clips > 1 else [ (n - clip_len) // 2 ]
    clips, example, multi = [], None, []
    for ci, st in enumerate(starts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st))
        frames = []
        for _ in range(clip_len):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if len(frames) < clip_len:
            continue
        if ci == len(starts) // 2:
            info["dhash_mid"] = dhash(frames[0])
        box, nf = detect_face(frames[0])
        multi.append(nf > 1)
        found = sum(detect_face(f)[0] is not None for f in frames[:: max(1, clip_len // 4)])
        face_rate = found / len(frames[:: max(1, clip_len // 4)])
        if box is None:
            clips.append(dict(clip=ci, start=int(st), face_found=0, face_rate=face_rate))
            continue
        x0, y0, x1, y1 = expand_box(box, W, H)
        crops = np.stack([cv2.resize(f[y0:y1, x0:x1], (crop, crop)) for f in frames]).astype(np.float32)
        d = np.abs(np.diff(crops, axis=0)).mean(axis=(1, 2))            # per transition
        sharp = np.array([cv2.Laplacian(c, cv2.CV_32F).var() for c in crops])
        clips.append(dict(clip=ci, start=int(st), face_found=1, face_rate=face_rate,
                          diff_mean=d.mean(), diff_std=d.std(), diff_max=d.max(),
                          diff_p95_ratio=np.percentile(d, 95) / (np.median(d) + 1.0),  # +1 grey level keeps static clips stable
                          sharp_mean=sharp.mean(), sharp_std=sharp.std()))
        if example is None:
            example = crops[0].astype(np.uint8)
    cap.release()
    info["multi_face_rate"] = float(np.mean(multi)) if multi else np.nan
    return info, clips, example


# ----------------------------------------------------------------------------- quality
def quality_report(man, clips):
    q = {}
    q["n_videos"] = int(len(man))
    q["per_method"] = man.method.value_counts().to_dict()
    q["unreadable_videos"] = int((~man.readable).sum())
    q["too_short_for_clip"] = int((man.n_frames < man.attrs.get("clip_len", 16)).sum())
    q["missing_split_assignment"] = int(man.split.isna().sum())
    fake = man[man.label == 1]
    q["fake_source_split_mismatch"] = int((fake.split != fake.source_split).sum())
    q["exact_duplicate_files"] = int(man.duplicated("sha256", keep=False).sum())
    # near-duplicates across *different* target ids (same id is expected by design: real + its fakes)
    nd = man.dropna(subset=["dhash_mid"]).groupby("dhash_mid").target_id.nunique()
    q["near_duplicate_groups_across_targets"] = int((nd > 1).sum())
    # identity / source-video leakage across official splits
    yt = man[man.method == "original"].groupby("youtube_id").split.nunique()
    q["youtube_sources_spanning_splits"] = int((yt > 1).sum())
    q["fps_values"] = man.fps.round(2).value_counts().head(10).to_dict()
    q["resolution_values"] = (man.width.astype("Int64").astype(str) + "x" +
                              man.height.astype("Int64").astype(str)).value_counts().head(10).to_dict()
    q["frames_quartiles"] = man.n_frames.describe().round(1).to_dict()
    q["frames_iqr_outliers"] = int(iqr_outliers(man.n_frames).sum())
    q["clip_face_found_rate"] = float(clips.face_found.mean()) if len(clips) else None
    q["clip_face_found_rate_by_method"] = (clips.merge(man[["video_id", "method"]])
                                           .groupby("method").face_found.mean().round(3).to_dict()
                                           if len(clips) else {})
    q["multi_face_video_rate"] = float(man.multi_face_rate.mean())
    q["class_ratio_real_to_fake"] = f"1:{(man.label == 1).sum() / max(1, (man.label == 0).sum()):.2f}"
    q["split_counts"] = man.groupby(["split", "method"]).size().unstack(fill_value=0).to_dict("index")
    # shortcut check: do trivial file properties differ by class?
    q["file_property_by_class"] = man.groupby("label")[["file_mb", "n_frames", "width"]].median().round(2).to_dict("index")
    return q


def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75])
    return (s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))


# ----------------------------------------------------------------------------- figures
def order(methods):
    return ["original"] + [m for m in methods if m != "original"]


def fig_class_split(man, out):
    ct = man.groupby(["split", "method"]).size().unstack(fill_value=0)
    ct = ct.reindex(["train", "val", "test"]).fillna(0)[[m for m in order(man.method.unique()) if m in ct]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(ct.index)); w = 0.8 / len(ct.columns)
    for i, m in enumerate(ct.columns):
        b = ax.bar(x + i * w - 0.4 + w / 2, ct[m], w * 0.92, color=COLORS.get(m, "#888"), label=m)
    ax.set_xticks(x, ct.index)
    ax.set_ylabel("Number of videos")
    ax.legend(frameon=False, ncol=len(ct.columns), fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.0))
    ax.set_title("Figure 1. Videos per class and method in each official split", pad=28)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "fig1_class_split.png"); plt.close(fig)


def fig_video_properties(man, out):
    meths = order(man.method.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, lab in [(axes[0], "duration_s", "Duration (s)"), (axes[1], "file_mb", "File size (MB)")]:
        data = [man.loc[man.method == m, col].dropna() for m in meths]
        bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=True, medianprops=dict(color="#0b0b0b"),
                        flierprops=dict(markersize=3, alpha=.5))
        for p, m in zip(bp["boxes"], meths):
            p.set_facecolor(COLORS.get(m, "#888")); p.set_alpha(.75)
        ax.set_xticks(range(1, len(meths) + 1), meths, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Figure 2. Video length and file size by class/method (shortcut check)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "fig2_video_properties.png"); plt.close(fig)


def fig_temporal(clips, man, out):
    c = clips[clips.face_found == 1].merge(man[["video_id", "method"]])
    meths = [m for m in order(man.method.unique()) if m in set(c.method)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, lab in [(axes[0], "diff_mean", "Mean |frame t+1 − frame t| in face crop"),
                         (axes[1], "diff_p95_ratio", "Spikiness of transitions (p95 / median)")]:
        data = [c.loc[c.method == m, col].dropna() for m in meths]
        bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=False, medianprops=dict(color="#0b0b0b"))
        for p, m in zip(bp["boxes"], meths):
            p.set_facecolor(COLORS.get(m, "#888")); p.set_alpha(.75)
        ax.set_xticks(range(1, len(meths) + 1), meths, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(lab, fontsize=9); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Figure 3. Frame-to-frame change inside the face region (clip level)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "fig3_temporal_instability.png"); plt.close(fig)


def fig_examples(examples, out, per_method=4):
    meths = [m for m in order(list(examples)) if examples.get(m)]
    if not meths:
        return
    fig, axes = plt.subplots(len(meths), per_method, figsize=(per_method * 1.6, len(meths) * 1.7))
    axes = np.atleast_2d(axes)
    for r, m in enumerate(meths):
        for k in range(per_method):
            ax = axes[r, k]; ax.axis("off")
            if k < len(examples[m]):
                ax.imshow(examples[m][k], cmap="gray")
            if k == 0:
                ax.set_title(m, fontsize=8, loc="left")
    fig.suptitle("Figure 4. Sample face crops after detection and cropping", fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig4_example_crops.png"); plt.close(fig)


# ----------------------------------------------------------------------------- baselines
def metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r = dict(n=int(len(y)), accuracy=accuracy_score(y, pred),
             precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
             fpr=fp / (fp + tn) if fp + tn else np.nan, f1=f1_score(y, pred, zero_division=0),
             tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if score is not None and len(set(y)) == 2:
        r.update(roc_auc=roc_auc_score(y, score), pr_auc=average_precision_score(y, score))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


def baselines(man, clips, eval_split="val"):
    res = {"eval_split": eval_split}
    tr, ev = man[man.split == "train"], man[man.split == eval_split]
    if len(ev) == 0 or len(tr) == 0:
        return {"error": "train or eval split empty"}
    # B0: majority class (learned on train)
    maj = int(tr.label.mean() >= 0.5)
    res["B0_majority_video"] = metrics(ev.label.values, np.full(len(ev), maj), np.full(len(ev), float(maj)))
    res["B0_majority_class"] = "fake" if maj else "real"
    # B1: logistic regression on hand-crafted temporal + sharpness features (clip level)
    c = clips[clips.face_found == 1].merge(man[["video_id", "split", "label", "method"]])
    ctr, cev = c[c.split == "train"].dropna(subset=FEATURES), c[c.split == eval_split].dropna(subset=FEATURES)
    if ctr.label.nunique() < 2 or len(cev) == 0:
        res["B1"] = "not enough clips with faces in both classes"
        return res
    # scaler is fit inside the pipeline on TRAIN only -> no leakage of eval statistics
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
    model.fit(ctr[FEATURES], ctr.label)
    p = model.predict_proba(cev[FEATURES])[:, 1]
    res["B1_logreg_clip"] = metrics(cev.label.values, (p >= .5).astype(int), p)
    cev = cev.assign(p=p)
    v = cev.groupby("video_id").agg(label=("label", "first"), p=("p", "mean"))   # video = mean of clips
    res["B1_logreg_video"] = metrics(v.label.values, (v.p >= .5).astype(int), v.p.values)
    res["B1_flag_rate_by_method_video (recall for fakes, FPR for original)"] = (cev.groupby(["video_id", "method"]).p.mean().reset_index()
                                        .assign(hit=lambda d: d.p >= .5).groupby("method").hit.mean().round(3).to_dict())
    res["B1_coefficients"] = dict(zip(FEATURES, np.round(model[-1].coef_[0], 3).tolist()))
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ff_root", required=True)
    ap.add_argument("--meta_dir", required=True, help="folder containing splits/ and conversion_dict.json")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--clips", type=int, default=2, help="clips sampled per video")
    ap.add_argument("--clip_len", type=int, default=16, help="consecutive frames per clip")
    ap.add_argument("--limit", type=int, default=0, help="max fake videos per method (0 = all)")
    ap.add_argument("--limit_original", type=int, default=-1,
                    help="max original videos (0 = all; default: same as --limit)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    split_of, yt_of = load_splits(a.meta_dir)
    lim_o = a.limit if a.limit_original < 0 else a.limit_original
    man = build_manifest(a.ff_root, a.methods, a.compression, split_of, yt_of, a.limit, lim_o)
    if man.empty:
        sys.exit(f"No usable videos found under {a.ff_root}. Skipped: {man.attrs.get('skipped')}")
    skipped = man.attrs.get("skipped", {})
    completeness = completeness_vs_official(man, a.meta_dir, a.methods)
    print("Skipped files:", skipped)
    print("Completeness vs. official file list:", completeness)
    man, bad_ids = flag_cross_split_sources(man, split_of, yt_of)
    print(f"{len(man)} videos found:", man.method.value_counts().to_dict())
    print(f"val/test ids sharing a YouTube source with another split: {bad_ids} "
          f"-> {int(man.leak_excluded.sum())} videos excluded from evaluation")

    infos, clip_rows, examples = [], [], collections.defaultdict(list)
    for i, r in enumerate(man.itertuples()):
        info, clips, ex = analyse_video(r.path, a.clips, a.clip_len)
        infos.append(info)
        clip_rows += [dict(video_id=r.video_id, **c) for c in clips]
        if ex is not None and len(examples[r.method]) < 4:
            examples[r.method].append(ex)
        if (i + 1) % 100 == 0:
            print(f"  analysed {i + 1}/{len(man)}")
    man = pd.concat([man, pd.DataFrame(infos)], axis=1)
    man.attrs["clip_len"] = a.clip_len
    clips = pd.DataFrame(clip_rows)
    if "face_found" not in clips:
        clips = pd.DataFrame(columns=["video_id", "face_found"] + FEATURES)
    man.drop(columns=["path"]).to_csv(out / "manifest.csv", index=False)
    clips.to_csv(out / "clips.csv", index=False)

    q = quality_report(man, clips)
    q["cross_split_youtube_ids_excluded"] = bad_ids
    q["skipped_files"] = skipped
    q["completeness_vs_official"] = completeness
    q["videos_excluded_for_leakage"] = int(man.leak_excluded.sum())
    q["split_counts_after_exclusion"] = (man[~man.leak_excluded].groupby(["split", "method"]).size()
                                         .unstack(fill_value=0).to_dict("index"))
    json.dump(q, open(out / "quality_report.json", "w"), indent=2, default=str)
    fig_class_split(man, out); fig_video_properties(man, out)
    if (clips.face_found == 1).any():
        fig_temporal(clips, man, out)
    fig_examples(examples, out)
    b = baselines(man[~man.leak_excluded], clips, "val")
    json.dump(b, open(out / "baseline_results.json", "w"), indent=2, default=str)
    print(json.dumps({"quality": q, "baselines": b}, indent=2, default=str))


if __name__ == "__main__":
    main()
