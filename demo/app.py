"""Laptop demo: deepfake triage console (no Colab, no internet needed).

    pip install -r demo/requirements.txt
    python demo/app.py                      # opens http://127.0.0.1:7860

Expects the bundle exported by notebook 06 (cell 5), unzipped into demo/:
    demo/models/<run>/best.pt, demo/models/<run>/eval/summary.json   (mnv2_v1, mnv2_robust)
    demo/videos/*.mp4 + demo/videos/catalog.csv                       (prepared test videos)
Weights and videos stay out of git (see .gitignore).
"""
from __future__ import annotations

import argparse
import base64
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from ffpp.infer import Detector  # noqa: E402
from ffpp.perturb import ATTACKS, make_transform  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

MODELS = {"mnv2_v1": "MobileNetV2", "mnv2_robust": "MobileNetV2 + defense"}
ATTACK_LABELS = {"none": "None", "jpeg_50": "JPEG q50", "jpeg_30": "JPEG q30", "jpeg_15": "JPEG q15",
                 "down_0.5": "Downscale ×0.5", "down_0.25": "Downscale ×0.25", "blur_1.5": "Blur σ1.5",
                 "blur_3": "Blur σ3", "noise_5": "Noise 5", "noise_10": "Noise 10"}


def jpg(img_rgb: np.ndarray, width: int | None = None, q=85) -> str:
    if width and img_rgb.shape[1] > width:
        h = int(img_rgb.shape[0] * width / img_rgb.shape[1])
        img_rgb = cv2.resize(img_rgb, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def preview_frames(path: Path, n=14, width=480, attack="none") -> list[str]:
    """Evenly spaced full frames for the flipbook (browsers cannot play FF++'s 4:4:4 H.264)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    tf = make_transform(attack)
    out = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h = int(rgb.shape[0] * width / rgb.shape[1])
        rgb = cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA)
        if tf:
            rgb = tf(rgb[None])[0]
        out.append(jpg(rgb, q=80))
    cap.release()
    return out


def build_app(models_dir: Path, videos_dir: Path) -> FastAPI:
    dets = {}
    for run in MODELS:
        if (models_dir / run / "best.pt").exists():
            d = Detector(models_dir / run, device=None)
            d.warmup()
            dets[run] = d
    if not dets:
        raise SystemExit(f"No models in {models_dir}. Unzip the demo bundle from notebook 06 into demo/.")
    catalog = pd.read_csv(videos_dir / "catalog.csv") if (videos_dir / "catalog.csv").exists() else pd.DataFrame()
    thumbs = {}
    for f in sorted(videos_dir.glob("*.mp4")) if videos_dir.exists() else []:
        fr = preview_frames(f, n=1, width=240)
        thumbs[f.name] = fr[0] if fr else ""

    app = FastAPI(title="Deepfake triage console")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/api/setup")
    def setup():
        ex = []
        for name, thumb in thumbs.items():
            row = catalog[catalog.file == name] if len(catalog) else catalog
            r = row.iloc[0].to_dict() if len(row) else {}
            ex.append(dict(file=name, thumb=thumb, truth=r.get("truth", "unknown"), method=r.get("method", ""),
                           why=r.get("why", "")))
        return dict(models=[dict(run=k, name=MODELS[k], threshold=round(d.threshold, 3), model=d.cfg["model"],
                                 device=d.device.type) for k, d in dets.items()],
                    attacks=[dict(key=k, label=ATTACK_LABELS[k]) for k in ATTACKS], examples=ex)

    def analyze_path(path: Path, attack: str, truth: dict | None):
        if attack not in ATTACKS:
            raise HTTPException(400, f"unknown attack {attack}")
        results, raw = [], None
        for run, d in dets.items():
            r = d.score(path, attack=attack, explain=True)
            raw = raw or r
            if r["status"] != "ok":
                raise HTTPException(422, "No face found in this video. Try a video with one clearly visible, frontal face."
                                    if r["status"] == "no_face" else f"Could not read the video ({r['status']}).")
            results.append(dict(run=run, name=MODELS[run], score=round(r["score"], 4), threshold=round(r["threshold"], 4),
                                verdict=r["verdict"], frame_scores=r.get("frame_scores"), clip_scores=r["clip_scores"],
                                cams=[jpg(c, q=82) for c in r["clip_cams"]], t_total_ms=round(r["t_total_ms"]),
                                t_model_ms=round(r["t_model_ms"], 1)))
        clips = raw
        return JSONResponse(dict(
            attack=attack, truth=truth, results=results,
            faces=[jpg(f, q=85) for f in clips["clip_faces"]], clip_start_s=clips["clip_start_s"],
            duration_s=round(clips["duration_s"], 1), fps=round(clips["fps"], 1),
            preview=preview_frames(path, attack=attack)))

    @app.post("/api/analyze")
    async def analyze(attack: str = Form("none"), example: str = Form(""), file: UploadFile | None = File(None)):
        if example:
            path = videos_dir / Path(example).name
            if not path.exists():
                raise HTTPException(404, "Example video not found.")
            row = catalog[catalog.file == path.name] if len(catalog) else catalog
            truth = row.iloc[0][["truth", "method", "why"]].to_dict() if len(row) else None
            return analyze_path(path, attack, truth)
        if file is None:
            raise HTTPException(400, "Choose an example or upload a video.")
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
        try:
            return analyze_path(Path(tmp.name), attack, None)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--videos", default=str(HERE / "videos"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    import uvicorn
    app = build_app(Path(a.models), Path(a.videos))
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Deepfake triage console: {url}")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
