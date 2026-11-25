"""Webcam upload checks with synthetic video and no trained model weights."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from demo import app as demo


@pytest.fixture
def videos(tmp_path):
    source = tmp_path / "video.webm"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"VP80"), 25, (64, 48))
    assert writer.isOpened()
    for i in range(75):
        writer.write(np.full((48, 64, 3), i * 3, dtype=np.uint8))
    writer.release()
    # MediaRecorder commonly writes WebM without a Duration element. Replace it
    # with an equal-sized EBML Void, preserving the rest of the real container.
    data = source.read_bytes()
    offset = data.index(b"\x44\x89\x88")
    recording = tmp_path / "webcam.webm"
    recording.write_bytes(data[:offset] + b"\xec\x89" + bytes(9) + data[offset + 11:])
    return source, recording


def test_normalize_durationless_recording_is_seekable_and_lossless(videos, tmp_path):
    _, source = videos
    target = tmp_path / "recording.avi"
    demo.normalize_recording(source, target)
    original, normalized = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(target))
    try:
        assert normalized.get(cv2.CAP_PROP_FRAME_COUNT) == 75
        assert normalized.get(cv2.CAP_PROP_FPS) == pytest.approx(25)
        for _ in range(75):
            ok1, expected = original.read()
            ok2, actual = normalized.read()
            assert ok1 and ok2
            np.testing.assert_array_equal(actual, expected)
        assert not normalized.read()[0]
        normalized.set(cv2.CAP_PROP_POS_FRAMES, 50)
        assert normalized.read()[0]
    finally:
        original.release()
        normalized.release()


@pytest.fixture
def client(monkeypatch, tmp_path):
    class Detector:
        threshold = 0.5
        cfg = {"model": "mnv2"}
        device = SimpleNamespace(type="cpu")
        status = "ok"
        paths = []

        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            pass

        def score(self, path, **kwargs):
            self.paths.append(Path(path))
            if self.status != "ok":
                return {"status": self.status}
            cap = cv2.VideoCapture(str(path))
            try:
                assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 75
                assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25)
                ok, frame = cap.read()
                assert ok
            finally:
                cap.release()
            return dict(status="ok", score=0.1, threshold=0.5, verdict="real", clip_scores=[0.1] * 4,
                        clip_faces=[frame] * 4, clip_cams=[frame] * 4, clip_start_s=[0, 0.5, 1, 1.5],
                        duration_s=3, fps=25, t_total_ms=1, t_model_ms=1)

    models = tmp_path / "models"
    (models / "mnv2_v1").mkdir(parents=True)
    (models / "mnv2_v1" / "best.pt").touch()
    monkeypatch.setattr(demo, "Detector", Detector)
    with TestClient(demo.build_app(models, tmp_path / "examples")) as test_client:
        yield test_client, Detector


@pytest.mark.parametrize("webcam", [False, True])
def test_video_analysis_upload_and_temporary_cleanup(client, videos, webcam):
    test_client, detector = client
    source = videos[1 if webcam else 0]
    response = test_client.post("/api/analyze", data={"attack": "none", "webcam": str(webcam).lower()},
                                files={"file": (source.name, source.read_bytes(), "video/webm")})
    assert response.status_code == 200
    result = response.json()
    assert result["duration_s"] == 3
    assert len(result["preview"]) == 14
    assert len(result["faces"]) == 4
    assert result["truth"] is None
    path, = detector.paths
    assert path.suffix == (".avi" if webcam else ".webm")
    assert not path.parent.exists()


@pytest.mark.parametrize("status, message", [("too_short", "at least 3 seconds"), ("no_face", "No face found")])
def test_webcam_analysis_error_cleans_temporary_files(client, videos, status, message):
    test_client, detector = client
    detector.status = status
    recording = videos[1]
    response = test_client.post("/api/analyze", data={"webcam": "true"},
                                files={"file": (recording.name, recording.read_bytes(), "video/webm")})
    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert not detector.paths[0].parent.exists()


def test_invalid_recording_is_rejected(client):
    test_client, detector = client
    response = test_client.post("/api/analyze", data={"webcam": "true"},
                                files={"file": ("webcam.webm", b"not a video", "video/webm")})
    assert response.status_code == 422
    assert "Could not read the webcam recording" in response.json()["detail"]
    assert not detector.paths
