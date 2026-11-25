# Laptop demo: deepfake triage console

Runs entirely on your laptop (CPU is enough, about 3 s per video). No Colab and no internet needed after setup.

1. **Get the bundle.** In Colab, run notebook 06: setup cell, cells 1–2 and cell 5. Cell 5 downloads
   `demo_bundle.zip` (both models + the prepared test videos).
2. **Unzip it into this folder**, so that you have `demo/models/mnv2_v1/best.pt`, `demo/models/mnv2_robust/best.pt`
   and `demo/videos/*.mp4`. These files are git-ignored: never commit them.
3. **Install and start** from the repository root (Python 3.10 or newer):

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r demo/requirements.txt
   python demo/app.py
   ```

   The browser opens http://127.0.0.1:7860.

**Demo script (about 2 minutes):**
1. *Real · Correct real* → no alert.
2. *Fake · Deepfakes* → alert; drag across the film strip to show the heatmap on nose and mouth.
3. The same fake with post-processing *JPEG q15* → MobileNetV2 lets it pass, the defended model still alerts.
4. *Real · False alarm* → an honest false alarm on a real person.
5. *Fake · FaceShifter (external) · Missed* → the new-tool problem.
