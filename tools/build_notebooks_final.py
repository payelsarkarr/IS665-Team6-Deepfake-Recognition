"""Build notebooks 05 (robustness, XAI, external test) and 06 (demo app) and refresh the setup cell
of notebooks 01–04. Run from the repository root:  python tools/build_notebooks_final.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
MODULES = '["data", "crops", "dataset", "models", "metrics", "train", "evaluate", "download", "infer", ' \
          '"perturb", "xai", "robustness"]'


def setup_source() -> str:
    src = "".join(json.load(open(NB / "04_inference.ipynb"))["cells"][1]["source"])
    start = src.index("for m in [")
    end = src.index("]:", start) + 2
    return src[:start] + f"for m in {MODULES}:" + src[end:]


def cell(kind, text):
    text = text.strip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    lines[-1] = lines[-1].rstrip("\n")
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def write(name, cells):
    nb = {"cells": [cell(k, t) for k, t in cells], "nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    json.dump(nb, open(NB / name, "w"), indent=1, ensure_ascii=False)
    print("wrote", NB / name)


SETUP = setup_source()

# ------------------------------------------------------------------------------------------- 05
nb5 = [
("markdown", """
# 05 · Robustness, defense, quality slices, Grad-CAM, external test (FaceShifter)

Everything the final report still needs, in one notebook. Run the cells **in order** on a **T4 GPU**
(about 60–75 min in total). Each cell saves its result to Drive, so after a disconnect you only rerun
SETUP, cells 1–2 and the cell that was interrupted.

| Cell | What | Output in Drive | Time |
|---|---|---|---|
| 3 | **Defense**: train `mnv2_robust` = `mnv2_v1` + perturbation-augmented training (JPEG, downscaling, blur; **noise is held out**), evaluate on val/test | `runs/mnv2_robust/` | ~15 min |
| 4 | **Attack test**: test split under 9 post-processing attacks, both models, at their validation thresholds | `runs/<run>/eval/robustness_test.csv` | ~20 min |
| 5 | **Quality slices**: false alarms and misses per brightness / contrast / sharpness tertile | `report/quality_slices.csv` | ~2 min |
| 6 | **Grad-CAM**: where the model looks (errors, correct detections, average map per method) | `report/gradcam_*.png` | ~3 min |
| 7 | **External test**: FaceShifter test split (never used in training), downloaded with the official script | `cache/crops_v1_faceshifter/`, `runs/<run>/eval/external_FaceShifter.json` | ~15 min |
| 8 | **Demo videos**: chosen test videos (+ FaceShifter) for notebook 06 | `demo_videos/` | ~2 min |
| 9 | **Pack results** for the report (no weights, no videos) | `results_final.zip` | <1 min |

Requires: notebooks 01–03 done (cache `crops_v1`, runs `mnv2_v1`, `mnv2_gru_v1`, `mnv2_lomo_FS` evaluated),
and the download script in Drive.
"""),
("code", SETUP),
("code", '''
#@title 1 · Settings
CACHE_VERSION = "crops_v1"  #@param {type:"string"}
BASE_RUN = "mnv2_v1"  #@param {type:"string"}
ROBUST_RUN = "mnv2_robust"  #@param {type:"string"}
ROBUST_AUG = 0.5  #@param {type:"number"}
ATTACK_RUNS = "mnv2_v1, mnv2_robust"  #@param {type:"string"}
#@markdown (add mnv2_gru_v1 above if there is time: +25 min)
EXTERNAL_RUNS = "mnv2_v1, mnv2_robust, mnv2_gru_v1, mnv2_lomo_FS"  #@param {type:"string"}
SCRIPT = f"{DRIVE_ROOT}/faceforensics_download_v4.py"
RUNS_DIR = f"{DRIVE_ROOT}/runs"
REPORT_DIR = f"{DRIVE_ROOT}/report"; os.makedirs(REPORT_DIR, exist_ok=True)
EXT_CACHE = f"{DRIVE_ROOT}/cache/{CACHE_VERSION}_faceshifter"
DEMO_DIR = f"{DRIVE_ROOT}/demo_videos"
split_runs = lambda s: [r.strip() for r in s.split(",") if r.strip()]
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to T4 GPU")
'''),
("code", '''
#@title 2 · Copy the face-clip cache to the Colab disk (fast reads)
from ffpp.dataset import copy_cache_local
LOCAL_CACHE = str(copy_cache_local(f"{DRIVE_ROOT}/cache/{CACHE_VERSION}", f"/content/cache/{CACHE_VERSION}"))
'''),
("code", '''
#@title 3 · Defense: train the perturbation-augmented model and evaluate it (resumes after a disconnect)
import json
from ffpp.train import train
from ffpp.evaluate import evaluate_run
base_cfg = json.load(open(f"{RUNS_DIR}/{BASE_RUN}/config.json"))
cfg = {**base_cfg, "run_name": ROBUST_RUN, "robust_aug": ROBUST_AUG, "robust_kinds": ["jpeg", "down", "blur"],
       "num_workers": 2}
print("same settings as", BASE_RUN, "+ robust_aug", ROBUST_AUG, "(JPEG, downscaling, blur; noise held out)")
hist = train(cfg, LOCAL_CACHE, RUNS_DIR)
res = evaluate_run(f"{RUNS_DIR}/{ROBUST_RUN}", LOCAL_CACHE)
t = res["test"]["at_val_threshold"]
print(f"{ROBUST_RUN}: test ROC-AUC {res['test']['at_0_5']['roc_auc']:.3f} | recall {t['recall']:.3f} | FPR {t['fpr']:.3f}")
'''),
("code", '''
#@title 4 · Attack test: recall (fakes caught) and FPR (real videos flagged) under post-processing
import pandas as pd, matplotlib.pyplot as plt
from ffpp.robustness import attack_table
tables = []
for r in split_runs(ATTACK_RUNS):
    f = f"{RUNS_DIR}/{r}/eval/robustness_test.csv"
    tables.append(pd.read_csv(f) if os.path.exists(f) else attack_table(f"{RUNS_DIR}/{r}", LOCAL_CACHE))
att = pd.concat(tables, ignore_index=True)
att.to_csv(f"{REPORT_DIR}/robustness_all.csv", index=False)
display(att.pivot(index="attack", columns="run", values="recall").round(3).loc[att.attack.unique()])
display(att.pivot(index="attack", columns="run", values="fpr").round(3).loc[att.attack.unique()])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for r, g in att.groupby("run", sort=False):
    ax[0].plot(g.attack, g.recall, marker="o", label=r); ax[1].plot(g.attack, g.fpr, marker="o", label=r)
for a, t in zip(ax, ["Recall (fakes caught)", "FPR (real videos flagged)"]):
    a.set_title(t); a.tick_params(axis="x", rotation=45); a.set_ylim(0, 1); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/robustness.png", dpi=150); plt.show()
'''),
("code", '''
#@title 5 · Quality slices: are false alarms concentrated in dark / low-contrast / blurry faces?
from ffpp.dataset import load_index
from ffpp.robustness import quality_features, quality_table
test_idx = load_index(LOCAL_CACHE, "test")
feats = quality_features(test_idx, LOCAL_CACHE)
feats.to_csv(f"{REPORT_DIR}/quality_features_test.csv", index=False)
rows = []
for r in [BASE_RUN, ROBUST_RUN]:
    pred = pd.read_csv(f"{RUNS_DIR}/{r}/eval/predictions_test.csv")
    thr = json.load(open(f"{RUNS_DIR}/{r}/eval/summary.json"))["threshold_from_val"]
    rows.append(quality_table(pred, feats, thr).assign(run=r))
qt = pd.concat(rows, ignore_index=True)
qt.to_csv(f"{REPORT_DIR}/quality_slices.csv", index=False)
display(qt.round(3))
'''),
("code", '''
#@title 6 · Grad-CAM: errors, correct detections, and the average map per method
import numpy as np
from ffpp.train import load_trained
from ffpp.dataset import FrameCache
from ffpp.xai import explain_frames
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mid_face(cache_file):
    fc = FrameCache(f"{LOCAL_CACHE}/{cache_file}"); return fc.frame(0, fc.T // 2)

def gallery(run, groups, fname, title):
    model, cfg, _ = load_trained(f"{RUNS_DIR}/{run}", "best", dev)
    groups = [(l, g) for l, g in groups if len(g)]
    if not groups:
        print("nothing to show for", fname); return
    n = max(len(g) for _, g in groups)
    fig, ax = plt.subplots(2 * len(groups), n, figsize=(2.1 * n, 4.4 * len(groups)))
    ax = np.atleast_2d(ax)
    for gi, (label, g) in enumerate(groups):
        for j in range(n):
            for k in (0, 1): ax[2 * gi + k, j].axis("off")
            if j >= len(g): continue
            row = g.iloc[j]
            face = mid_face(row.cache_file)
            ov, _, p = explain_frames(model, face[None], cfg["img_size"], False, dev)
            ax[2 * gi, j].imshow(face); ax[2 * gi + 1, j].imshow(ov[0])
            ax[2 * gi, j].set_title(f"{label}\\n{row.method} {row.video_id.split('/')[-1]}\\nvideo P(fake)={row.score:.2f}", fontsize=7)
    fig.suptitle(title, fontsize=10); plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/{fname}", dpi=150); plt.show()

idx_c = test_idx[["video_id", "cache_file"]]
base = pd.read_csv(f"{RUNS_DIR}/{BASE_RUN}/eval/predictions_test.csv").merge(idx_c, on="video_id")
thr = json.load(open(f"{RUNS_DIR}/{BASE_RUN}/eval/summary.json"))["threshold_from_val"]
fp = base[(base.label == 0) & (base.score >= thr)].nlargest(4, "score")
fn = base[(base.label == 1) & (base.score < thr)].nsmallest(4, "score")
tp = pd.concat([g.nlargest(1, "score") for _, g in base[base.label == 1].groupby("method")])
tn = base[base.label == 0].nsmallest(4, "score")
gallery(BASE_RUN, [("False alarm", fp), ("Missed fake", fn), ("Detected fake", tp), ("Correct real", tn)],
        "gradcam_errors.png", f"{BASE_RUN}: Grad-CAM (red = pushes towards 'fake')")
lomo = pd.read_csv(f"{RUNS_DIR}/mnv2_lomo_FS/eval/predictions_test.csv").merge(idx_c, on="video_id")
fs_miss = lomo[lomo.method == "FaceSwap"].nsmallest(4, "score")
gallery("mnv2_lomo_FS", [("FaceSwap missed (unseen)", fs_miss)], "gradcam_lomo_faceswap.png",
        "Model trained without FaceSwap: where it looks on FaceSwap fakes it calls real")
gallery(BASE_RUN, [("FaceSwap detected (seen)", base.merge(fs_miss[["video_id"]], on="video_id"))],
        "gradcam_base_faceswap.png", f"Same FaceSwap videos, {BASE_RUN} (FaceSwap seen in training)")

# average Grad-CAM per class/method over all test videos (middle frame of clip 1)
model, cfg, _ = load_trained(f"{RUNS_DIR}/{BASE_RUN}", "best", dev)
avg = {}
for m, g in base.groupby("method"):
    faces = np.stack([mid_face(c) for c in g.cache_file])
    cams = np.concatenate([explain_frames(model, faces[i:i + 32], cfg["img_size"], False, dev)[1]
                           for i in range(0, len(faces), 32)])
    avg[m] = (faces.mean(0).astype(np.uint8), cams.mean(0))
fig, ax = plt.subplots(1, len(avg), figsize=(2.6 * len(avg), 3))
from ffpp.xai import overlay
for a, (m, (face, cam)) in zip(ax, avg.items()):
    a.imshow(overlay(face, cam / cam.max())); a.set_title(m, fontsize=9); a.axis("off")
fig.suptitle(f"{BASE_RUN}: average Grad-CAM per method (test)", fontsize=10); plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/gradcam_average.png", dpi=150); plt.show()
np.savez_compressed(f"{REPORT_DIR}/gradcam_average.npz", **{m: c for m, (f, c) in avg.items()})
'''),
("code", '''
#@title 7 · External test: FaceShifter (official FF++ release, never used for training)
from ffpp import download
from ffpp.data import build_manifest, load_official_meta, cross_split_ids
from ffpp.crops import extract_all
from ffpp.robustness import evaluate_external
split_of, youtube_of, _ = load_official_meta(META)
names = download.split_pairs(META, "test", exclude_ids=set(cross_split_ids(split_of, youtube_of)))
print(len(names), "FaceShifter test videos")
download.fetch(SCRIPT, "/content/fsh", [("FaceShifter", n) for n in names])
man = build_manifest("/content/fsh", META, methods=["FaceShifter"], skip=("deepfakedetection", "actors"))
print(man.groupby(["split", "method"]).size())
extract_all(man, EXT_CACHE)
ext = []
for r in split_runs(EXTERNAL_RUNS):
    if os.path.exists(f"{RUNS_DIR}/{r}/eval/summary.json"):
        ext.append(evaluate_external(f"{RUNS_DIR}/{r}", EXT_CACHE, LOCAL_CACHE))
ext = pd.DataFrame(ext)
ext.to_csv(f"{REPORT_DIR}/external_faceshifter.csv", index=False)
display(ext[["run", "roc_auc", "recall", "recall_lo", "recall_hi", "fpr", "n_fake", "n_real",
             "ffpp_test_roc_auc", "ffpp_test_recall"]].round(3))
'''),
("code", '''
#@title 8 · Demo videos for notebook 06 (test split only: none of them was seen in training)
os.makedirs(DEMO_DIR, exist_ok=True)
fsh = pd.read_csv(f"{RUNS_DIR}/{BASE_RUN}/eval/predictions_FaceShifter_test.csv")
fsh = fsh[fsh.label == 1]
pick = [("correct real", base[base.label == 0].nsmallest(2, "score")),
        ("false alarm", fp.head(2)),
        ("detected fake", tp),
        ("missed fake", fn.head(2)),
        ("FaceShifter detected", fsh.nlargest(1, "score")),
        ("FaceShifter missed", fsh.nsmallest(1, "score"))]
items, cat = [], []
for why, g in pick:
    for r in g.itertuples():
        ds, name = ("original", r.video_id) if r.method == "original" else (r.method, r.video_id.split("/")[-1])
        items.append((ds, name))
        cat.append(dict(file=f"{ds}_{name}.mp4", truth="real" if r.label == 0 else "fake", method=r.method,
                        why=why, base_score=round(r.score, 3)))
paths = download.fetch(SCRIPT, "/content/demo_dl", items)
import shutil
for (ds, name), c in zip(items, cat):
    src = next((p for p in paths if p.stem == name and ("original" if ds == "original" else ds) in str(p)), None)
    if src: shutil.copy2(src, f"{DEMO_DIR}/{c['file']}")
cat = pd.DataFrame(cat).drop_duplicates("file")
cat.to_csv(f"{DEMO_DIR}/catalog.csv", index=False)
display(cat)
'''),
("code", '''
#@title 9 · Pack all results for the report (JSON/CSV/PNG only; no weights, no videos)
import zipfile, glob
out = f"{DRIVE_ROOT}/results_final.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{RUNS_DIR}/*/*.json") + glob.glob(f"{RUNS_DIR}/*/*.csv") + \\
             glob.glob(f"{RUNS_DIR}/*/eval/*.json") + glob.glob(f"{RUNS_DIR}/*/eval/*.csv"):
        z.write(f, "runs/" + os.path.relpath(f, RUNS_DIR))
    for f in glob.glob(f"{REPORT_DIR}/*"):
        z.write(f, "report/" + os.path.basename(f))
    if os.path.exists(f"{DEMO_DIR}/catalog.csv"):
        z.write(f"{DEMO_DIR}/catalog.csv", "demo_videos/catalog.csv")
print(out, f"{os.path.getsize(out) / 1e6:.1f} MB")
from google.colab import files; files.download(out)
'''),
]

# ------------------------------------------------------------------------------------------- 06
nb6 = [
("markdown", """
# 06 · Live demo app (Gradio)

Upload a video (or pick a prepared test video) → face clips → **two models side by side**
(`mnv2_v1` = main model, `mnv2_robust` = with the defense) → verdict at the validation threshold,
**score timeline**, **Grad-CAM** per clip. The attack menu applies the same post-processing as in
notebook 05, so you can show live how re-compression or downscaling changes the verdict.

Run SETUP and cells 1–3; cell 3 prints a public `gradio.live` link (valid 72 h) and also shows the app
inline. Cell 4 writes **backup screenshots** of every prepared demo video to `demo_backup/` in Drive.
Runtime: T4 GPU (CPU also works, ~3 s per video).
"""),
("code", SETUP),
("code", '''
#@title 1 · Settings
MODELS = "mnv2_v1, mnv2_robust"  #@param {type:"string"}
DEMO_DIR = f"{DRIVE_ROOT}/demo_videos"
RUNS_DIR = f"{DRIVE_ROOT}/runs"
BACKUP_DIR = f"{DRIVE_ROOT}/demo_backup"
import importlib.util
if importlib.util.find_spec("gradio") is None:
    !pip -q install gradio
'''),
("code", '''
#@title 2 · Load the models
import torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from ffpp.infer import Detector
from ffpp.perturb import ATTACKS
dets = {m.strip(): Detector(f"{RUNS_DIR}/{m.strip()}") for m in MODELS.split(",") if m.strip()}
for n, d in dets.items():
    d.warmup(); print(f"{n}: threshold {d.threshold:.3f} ({d.threshold_source}), {d.device}")
catalog = pd.read_csv(f"{DEMO_DIR}/catalog.csv") if os.path.exists(f"{DEMO_DIR}/catalog.csv") else pd.DataFrame()
display(catalog)
'''),
("code", '''
#@title 3 · Start the app
import gradio as gr
MAIN = list(dets)[0]

def analyze(video, attack):
    if not video:
        return "Upload or pick a video.", None, []
    res = {n: d.score(video, attack=attack, explain=(n == MAIN)) for n, d in dets.items()}
    r0 = res[MAIN]
    if r0["status"] != "ok":
        return f"No face found ({r0['status']}).", None, []
    truth = ""
    if len(catalog):
        hit = catalog[catalog.file == os.path.basename(video)]
        if len(hit): truth = f"**Ground truth:** {hit.truth.iloc[0]} ({hit.method.iloc[0]})\\n\\n"
    md = truth + f"**Attack:** `{attack}`\\n\\n| Model | P(fake) | Threshold | Verdict |\\n|---|---|---|---|\\n"
    for n, r in res.items():
        md += f"| {n} | {r['score']:.3f} | {r['threshold']:.3f} | **{r['verdict']}** |\\n"
    md += f"\\n{r0['n_clips']} clips, {r0['duration_s']:.1f} s video, analysed in {r0['t_total_ms'] / 1000:.1f} s"
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for n, r in res.items():
        if "frame_scores" in r:
            ys = np.concatenate(r["frame_scores"]); ax.plot(ys, lw=1.5, label=n)
        else:
            ax.plot(np.repeat(r["clip_scores"], 16), lw=1.5, label=n)
    ax.axhline(dets[MAIN].threshold, color="k", ls="--", lw=1, label="threshold")
    ax.set_ylim(0, 1); ax.set_xlabel("frame (4 clips x 16 frames)"); ax.set_ylabel("P(fake)")
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
    gal = []
    for c in range(r0["n_clips"]):
        gal += [(r0["clip_faces"][c], f"clip {c + 1} @ {r0['clip_start_s'][c]} s, P(fake) {r0['clip_scores'][c]:.2f}"),
                (r0["clip_cams"][c], f"clip {c + 1}: Grad-CAM ({MAIN})")]
    return md, fig, gal

examples = [[f"{DEMO_DIR}/{f}", "none"] for f in catalog.file] if len(catalog) else None
with gr.Blocks(title="Deepfake check (IS665 Team 6)") as app:
    gr.Markdown("## Deepfake check · MobileNetV2 trained on FaceForensics++ (research prototype: needs human review)")
    with gr.Row():
        with gr.Column(scale=1):
            vid = gr.Video(label="Video")
            atk = gr.Dropdown(list(ATTACKS), value="none", label="Simulated attack (post-processing)")
            btn = gr.Button("Analyse", variant="primary")
            if examples: gr.Examples(examples, [vid, atk], label="Prepared test videos")
        with gr.Column(scale=2):
            out_md = gr.Markdown(); out_plot = gr.Plot(label="Score timeline")
            out_gal = gr.Gallery(label="Face per clip and Grad-CAM", columns=4, height=360)
    btn.click(analyze, [vid, atk], [out_md, out_plot, out_gal])
app.launch(share=True, debug=False)
'''),
("code", '''
#@title 4 · Backup screenshots of every demo video (clean + jpeg_15 + down_0.25) -> demo_backup/
os.makedirs(BACKUP_DIR, exist_ok=True)
for f in catalog.file:
    for attack in ["none", "jpeg_15", "down_0.25"]:
        md, fig, gal = analyze(f"{DEMO_DIR}/{f}", attack)
        n = len(gal)
        sheet = plt.figure(figsize=(2.3 * max(n // 2, 1), 7.5))
        ax0 = sheet.add_subplot(3, 1, 1); ax0.axis("off")
        ax0.text(0, 1, md.replace("**", "").replace("`", ""), va="top", family="monospace", fontsize=8)
        for i, (img, cap) in enumerate(gal):
            a = sheet.add_subplot(3, max(n // 2, 1), max(n // 2, 1) + 1 + (i // 2) + (i % 2) * max(n // 2, 1))
            a.imshow(img); a.axis("off"); a.set_title(cap, fontsize=6)
        sheet.savefig(f"{BACKUP_DIR}/{os.path.splitext(f)[0]}_{attack}.png", dpi=130, bbox_inches="tight")
        fig.savefig(f"{BACKUP_DIR}/{os.path.splitext(f)[0]}_{attack}_timeline.png", dpi=130, bbox_inches="tight")
        plt.close("all")
    print("saved", f)
'''),
]

# ------------------------------------------------------------------ refresh setup in 01-04, write 05-06
for f in sorted(NB.glob("0[1-4]_*.ipynb")):
    nb = json.load(open(f))
    for c in nb["cells"]:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and s.startswith("#@title 0 · Setup"):
            c["source"] = cell("code", SETUP)["source"]
    json.dump(nb, open(f, "w"), indent=1, ensure_ascii=False)
write("05_robustness_xai_external.ipynb", nb5)
write("06_demo_app.ipynb", nb6)
