# Project Notes Since Interim Report 1 (as of Oct 27, 2026)

Team 6, IS665: Flemming Grabowski, Payel Sarkar. Topic: deepfake detection on FaceForensics++ (FF++).
Purpose of this file: all facts, numbers and decisions since Interim Report 1 was submitted (Oct 11, 2026),
so that the status report (Nov 8), the final report and the presentation (Dec 6) can build on them.

---

## 1. Deliverables and Rubrics

| Due | Deliverable | Points | Length |
|---|---|---|---|
| Nov 8, 2026 | Interim Status Report (template `03_Interim_Status_Report_Template.docx`) | 10 | 3–5 page PDF |
| Dec 6, 2026 | Final Project Report (template `04_Final_Project_Report_Template.docx`) | 40 | 8–12 page PDF (excluding title page, references, appendices) |
| Dec 6, 2026 | In-Class Presentation (template `05_In_Class_Presentation_Template.pptx`) | 20 | 10–12 min, live demo + backup |

**Status report (10):** progress 1 · models 2 · metrics 2 · error analysis + generalization 2 · adversarial 1 · RAI + monitoring 1 · plan/repo/contributions 1.
Contents: updated status; model development (baseline + ≥ 2 approaches, hyperparameters/tuning, training/validation,
imbalance, reproducibility); preliminary evaluation (justified primary metric, confusion matrix, FPR/FNR,
ROC-AUC, PR-AUC, latency/cost); error analysis (representative FP/FN, causes, security consequences);
generalization (class sample data + ≥ 1 larger/external dataset, schema issues); adversarial threat
model (asset, attacker, attack, planned defense experiment); responsible AI (unequal performance across
groups, otherwise another dimension such as attack type); monitoring and response plan; team contribution.
Accuracy alone is not enough.

**Final report (40):** problem 2 · data/EDA 4 · model development 8 · evaluation 8 · cross-dataset 2 · monitoring 4 ·
adversarial 4 · RAI 4 · code/reproducibility 2 · writing 2.
Sections: executive summary; problem/use case; data + quality (provenance, labels, imbalance, splits,
leakage); EDA/features; model development; evaluation + error analysis (results table, confusion matrix,
vs. baseline, FP/FN, trade-offs between detection, false alarms, complexity, speed, interpretability); deep analysis +
generalization (class sample data + external dataset, document schema/drift); interpretation/XAI;
adversarial robustness (threat model, controlled experiment, defense, before/after); RAI/privacy/governance;
monitoring/feedback/response (diagram recommended); limitations; conclusion; reproducibility (README, environment
file, configs, data access, tests, model card, AI use, contributions). No restricted data in the public repository.

**Presentation (20):** problem 3 · pipeline 4 · results 5 · demo 4 · limitations/RAI 2 · delivery 2.
Contents: problem, pipeline, baseline + models, results, FP/FN examples, generalization, adversarial/
privacy/RAI, monitoring, limitations, live demo + backup (screenshots/recording).

---

## 2. What Happened Since Interim Report 1 (Timeline)

| Date | Step |
|---|---|
| Oct 11 | Interim Report 1 submitted (ran on a 1,250-video subset of the Kaggle mirror because the official servers were down). |
| Oct 13–16 | New modular pipeline built (PyTorch instead of Keras): `src/ffpp/` + notebooks 01/02/03. The old notebook `FFpp_EDA_Colab.ipynb` stays frozen. |
| Oct 16 | Official download possible again (server **EU2**). New Google account, empty Drive. |
| Oct 16/17 | All 5,000 c23 videos downloaded into the Colab runtime; partial download backed up to `DRIVE/ffpp_partial` with `rsync` after a connection drop. |
| Oct 16/17 (overnight) | Face extraction for all videos → `DRIVE/cache/crops_v1` (~3 GB). |
| Oct 17 | Trained b2_cnn_v1, mnv2_v1, mnv2_gru_v1 on a T4; test evaluation with the validation threshold; LOMO runs; error analysis. |
| Oct 19 | GRU LOMO runs (FaceSwap, NeuralTextures held out). |
| Oct 27 | Inference notebook 04 with latency measurement; status report finalized and submitted to the repository. |

---

## 3. Data

- **FF++ c23**, official release (access requested; script `faceforensics_download_v4.py`; the script and the personal
  link live only in Drive, **never in the repository**). 1,000 originals + 4 methods with 1,000 each: Deepfakes (DF),
  Face2Face (F2F), FaceSwap (FS), NeuralTextures (NT). Real:fake = 1:4.
- **Servers** (addresses are in the private download script): EU (refused on Oct 16), **EU2 (worked)**, CA. Kaggle mirror
  `xdxd003/ff-c23` used only for Interim Report 1; the instructor wants the official data, Kaggle only by agreement.
- **Video format:** H.264 High 4:4:4 (yuv444p). QuickTime shows a black picture, OpenCV reads the files correctly. No conversion.
- **Splits:** official `train/val/test.json` (github ondyari/FaceForensics, `dataset/splits`): 360/70/70 pairs
  = 720/140/140 original IDs. Fakes `target_source` inherit the split.
- **Leakage exclusion:** 22 YouTube sources yield 45 sequences; 7 groups span splits
  (605/606, 639/640, 755/756, 837/838, 875/876, 954/955, 982/983). Excluded from val/test: originals
  606, 640, 755, 837, 875, 955, 982 + all fakes that use one of these IDs (63 videos). Visual check:
  different people, same studio/setting (837/838 are sisters) → the risk is background/studio leakage,
  not identity leakage. They stay in training.
- **Effective sizes after exclusion:** val 135 real + 130 per method (= 655), test 138 real + 136 per method (= 682).
- **Extraction (notebook 01):** 4,999 ok, 1 `no_face`. Clips per video: {0:1, 1:2, 2:8, 3:43, 4:4946}.

### Face crops (`crops.py`, `crops_v1`)
- Haar cascade (OpenCV 4.x; OpenCV 5 has no cascades → checked in setup) on frames downscaled to at most 640 px
  (437 → 216 ms per frame).
- **Fixed box per clip** (no flicker), enlarged ×1.3, square, 224 px, JPEG quality 90.
- 4 clips × 16 frames per video, resampled to ~25 fps (`stride = round(fps/25)`).
- One `.npz` per video (jpeg, offsets, meta, starts), written atomically; a restart skips finished files.
  `index.csv`: manifest + cache_file, n_clips, status. `config.json` stores the extraction settings.

---

## 4. Pipeline / Code

```
src/ffpp/
  data.py      manifest: find videos, official splits, leakage flag (leak_excluded)
  download.py  official script (server fallback EU→EU2→CA, live progress), Kaggle fallback
  crops.py     face clips per video (.npz), parallel, restartable
  dataset.py   TrainSet (frame or clip samples, same augmentation for all frames), VideoEvalSet
  models.py    b2_cnn, mnv2, mnv2_gru
  train.py     training loop, checkpoint every epoch, automatic resume
  metrics.py   ROC-AUC, PR-AUC, threshold at FPR ≤ x, recall@FPR
  evaluate.py  threshold on val, test metrics, per method, run comparison
  infer.py     end-to-end scoring of a new video file (see Section 10)
notebooks/ 01_prepare_data · 02_train · 03_evaluate · 04_inference
```

**Drive layout** (`DRIVE_ROOT=/content/drive/MyDrive/IS665_Deepfake`):
```
faceforensics_download_v4.py     (private)
cache/crops_v1/                  index.csv, config.json, videos/*.npz   (~3 GB)
runs/<run>/                      config.json, metrics.csv, last.pt, best.pt, val_best.csv
runs/<run>/eval/                 summary.json, per_method_test.csv, predictions_{val,test}.csv,
                                 error_examples.png/.csv, inference_examples.csv, latency.json
ffpp_partial/                    old backup of the partial download → DELETE and empty the trash
```

**Models**
| Name | Architecture | Parameters |
|---|---|---|
| b2_cnn | 4 conv blocks trained from scratch (baseline) | 1.17 M |
| mnv2 | MobileNetV2, ImageNet (IMAGENET1K_V2), frame classifier | 2.23 M |
| mnv2_gru | MobileNetV2 per frame + bidirectional GRU (hidden 256), mean over time | 4.59 M |

**Training:** AdamW, warm-up + cosine (computed from the epoch → safe to resume), BCE with class weights
(imbalance 1:4), AMP, backbone frozen in the first `freeze_backbone_epochs` (BatchNorm in eval mode), augmentation
(flip, brightness, JPEG) identical for all frames of a sample, seed 665. `last.pt` every epoch,
`best.pt` by validation ROC-AUC. A guard on the run's key settings (model, pretrained, img_size, clip_len,
exclude_methods) prevents resuming with a different setup. Clip mode: batch = batch_size // clip_len.

**Evaluation:** video score = mean P(fake) over 8 evaluation frames or over the clips. Threshold chosen on **validation**
(highest recall with FPR ≤ 0.20) and applied unchanged to **test**. Per method: share flagged as fake
(= recall; for `original` = FPR).

**Primary metric (rationale for the report):** recall at FPR ≤ 0.20 (operationally: a fixed false-alarm budget, catch
as many fakes as possible) + ROC-AUC/PR-AUC as threshold-free comparisons. Accuracy is misleading at 1:4
(always "fake" = 80%).

---

## 5. Results (Oct 17, 2026, Colab T4)

### Run settings and validation
| Run | Image | LR | Epochs | Frozen | Other | min/epoch | Val AUC (best) | Val recall@FPR0.2 |
|---|---|---|---|---|---|---|---|---|
| b2_cnn_v1 | 128 | 1e-3 | 10 | – | batch 64, samples 4 | ~0.8 | 0.60 → **0.699** | 0.46 |
| mnv2_v1 | 224 | 3e-4 | 8 | 1 | batch 64, samples 4 | ~1.1 | **0.975** | 0.967 |
| mnv2_gru_v1 | 224 | 1e-4 | 6 | 1 | clip 16, samples 2 | ~7 | **0.977** | 0.967 |

- b2: loss still falling → underfit (candidate b2_cnn_v2: 30 epochs, samples 8).
- mnv2: loss 0.69 → 0.165. The GRU was still improving slightly at the end.

### Test (threshold from validation)
| Model | Best epoch | Test ROC-AUC | Recall | FPR | Threshold |
|---|---|---|---|---|---|
| b2_cnn | 9 | 0.684 | 0.467 | 0.283 | 0.606 |
| mnv2 | 8 | 0.967 | 0.965 | 0.217 (30/138) | 0.311 |
| mnv2_gru | 6 | **0.976** | **0.976** | 0.246 (34/138) | 0.249 |

More metrics (test, validation threshold, bootstrap 1,000×, seed 665):
b2 P 0.867 R 0.467 F1 0.607 PR-AUC 0.884 ROC 0.684 [0.630, 0.733] ·
mnv2 P 0.946 R 0.965 F1 0.955 PR-AUC 0.992 ROC 0.967 [0.952, 0.980] FPR 0.217 [0.155, 0.294] ·
GRU P 0.940 R 0.976 F1 0.958 PR-AUC 0.994 ROC 0.976 [0.965, 0.986] FPR 0.246 [0.177, 0.315].
Validation FPR at the threshold: mnv2 0.17, GRU 0.20. GRU vs. mnv2 not significant (CIs overlap).

### Per method (test: 136 fakes per method, 138 originals)
| Model | DF | F2F | FS | NT | original (FPR) |
|---|---|---|---|---|---|
| b2_cnn | 0.662 | 0.419 | 0.404 | 0.382 | 0.283 |
| mnv2 | 1.000 | 0.963 | 0.985 | 0.912 | 0.217 |
| mnv2_gru | 1.000 | 0.971 | 0.993 | 0.941 | 0.246 |

ROC-AUC per method vs. originals (test): b2 DF 0.80 / F2F 0.67 / FS 0.64 / NT 0.62 · mnv2 0.987 / 0.966 / 0.978 / 0.938 ·
GRU 0.990 / 0.980 / 0.986 / 0.948.

### Confusion matrices (test, validation threshold; 138 real, 544 fake)
| Model | TN | FP | FN | TP | FPR | FNR |
|---|---|---|---|---|---|---|
| b2_cnn | 99 | 39 | 290 | 254 | 0.283 | 0.533 |
| mnv2 | 108 | 30 | 19 | 525 | 0.217 | 0.035 |
| mnv2_gru | 104 | 34 | 13 | 531 | 0.246 | 0.024 |

### Interpretation
- Transfer learning is the big step (AUC 0.68 → 0.97). The baseline trained from scratch learns almost only DF
  (the coarsest method) in 10 epochs.
- **Recall goal met, FPR goal narrowly missed on test** (0.217 / 0.246 instead of ≤ 0.20). With 138 originals,
  one video ≈ 0.7 pp; sampling noise ±~3.5 pp. Do **not** tune on test afterwards. Lesson: choose the threshold
  with a safety margin (e.g., validation FPR ≤ 0.15 for a 0.20 target) and report confidence intervals.
- GRU barely better on known methods (+0.009 AUC), more a shift along the ROC curve, at ~6× the training time.
- **NeuralTextures is the hardest** (subtle texture change in the mouth region only), DF the easiest.
- False alarms (FPR) are the real weakness.
- **Base-rate effect (important for the report):** the test set is 80% fake. At a realistic rate of, e.g., 1% fakes,
  mnv2's precision would be 0.965·0.01 / (0.965·0.01 + 0.217·0.99) ≈ **4%**, i.e., ~24 false alarms per true hit.
  → The model works as a pre-filter with human review, not as an automatic block; the FPR is the lever.

## 5b. Leave-one-method-out (Oct 17, 2026, mnv2, same settings as mnv2_v1, patience 3)
The held-out method is missing from training **and** validation (so also from the threshold choice). Test: 138 real + 136 per method.

| Run | Best ep. | Threshold | AUC seen | AUC **unseen** | Recall seen | Recall **unseen** | FPR |
|---|---|---|---|---|---|---|---|
| mnv2_lomo_DF | 7 | 0.325 | 0.966 | **0.910** | 0.958 | **0.897** | 0.225 |
| mnv2_lomo_F2F | 8 | 0.386 | 0.972 | **0.782** | 0.961 | **0.544** | 0.167 |
| mnv2_lomo_NT | 5 | 0.295 | 0.988 | **0.723** | 0.971 | **0.368** | 0.058 |
| mnv2_lomo_FS | 8 | 0.302 | 0.971 | **0.414** | 0.975 | **0.132** | 0.217 |

Recall per method in the LOMO runs (unseen in bold): DF run DF **0.897**, F2F 0.956, FS 1.000, NT 0.919 ·
F2F run DF 0.978, F2F **0.544**, FS 0.978, NT 0.927 · FS run DF 1.000, F2F 0.978, FS **0.132**, NT 0.949 ·
NT run DF 0.978, F2F 0.949, FS 0.985, NT **0.368**.

**Interpretation**
- Key finding: ~0.97 AUC on known methods strongly overstates performance against new manipulation methods.
  Unseen method: AUC 0.41–0.91, recall 13–90%. → central for generalization, threat model and limitations.
- **FaceSwap unseen: AUC 0.41 < 0.5** – the model scores FS fakes as *more real* than originals. FS is graphics-based
  (landmarks + 3D model + color matching), DF is learning-based (autoencoder). Asymmetry: with FS seen, unseen DF
  is detected well (0.91); with DF seen, FS is not. Hypothesis (to check with Grad-CAM/examples): the model learns
  autoencoder/rendering traces (blur, texture) that FS does not have.
- Expression methods (F2F, NT) transfer only partly to each other (0.78 and 0.72).
- Known methods stay equally good (0.97–0.99) → holding one out does not hurt the others.
- FPR varies between runs (0.058–0.225) because each threshold is chosen on a different validation set/model;
  the low FPR of the NT run (8/138) is an artifact of the threshold choice, not an improvement.
- Security consequence: an attacker with a new generator very likely evades the detector →
  monitor score drift, retrain regularly with new methods, test on an external dataset.

## 5c. GRU LOMO (Oct 19, 2026; same settings as mnv2_gru_v1, ~5.7 min/epoch)
| Run | Best ep. | Threshold | AUC seen | AUC unseen [95% CI] | Recall unseen | FPR |
|---|---|---|---|---|---|---|
| mnv2_lomo_FS | 8 | 0.302 | 0.971 | 0.414 [0.351, 0.480] | 0.132 | 0.217 |
| **mnv2_gru_lomo_FS** | 5 | 0.178 | 0.976 | **0.339** [0.277, 0.400] | 0.059 | 0.203 |
| mnv2_lomo_NT | 5 | 0.295 | 0.988 | 0.723 [0.660, 0.780] | 0.368 | 0.058 |
| **mnv2_gru_lomo_NT** | 6 | 0.108 | 0.990 | **0.748** [0.688, 0.806] | 0.559 | 0.196 |

Paired bootstrap (same videos, 2,000×), AUC GRU − mnv2 on the unseen method:
FS −0.075 [−0.133, −0.019] (GRU significantly **worse**), NT +0.025 [−0.024, 0.074] (no difference).
Recall at equal test FPR ≤ 0.2 (descriptive only): FS mnv2 0.118 / GRU 0.051; NT 0.537 / 0.603.
**Conclusion:** temporal modeling does not help with unseen methods. The GRU learns the temporal traces of the
methods it has seen, not general flicker. The GRU's higher NT recall comes mostly from a looser
operating point (FPR 0.20 vs. 0.06). Reported in the status report (Section 5, Table 4). DF/F2F GRU LOMO not run.

## 5d. Error analysis (Oct 17, 2026, error-example cell in notebook 03)
Images: `runs/<run>/eval/error_examples.png` + `.csv` in Drive. FF++ frames do **not** go into the
public repository; in the report PDF they are fine as figures with a source citation. Do not name people in the report.

**mnv2_v1 (threshold 0.311): 30 false alarms, 19 missed fakes**
- Top false alarms (P(fake) 0.79–0.96, single frames up to 0.999 → very confidently wrong): originals 851, 314, 725, 691,
  035, 257. All news/studio footage. Visible patterns: head covering (851), glasses (314, 691), open
  mouth with teeth (725), lowered eyes/closed eyes (035). Hypothesis: unusual features in the
  face region or low source quality look like manipulation traces to the model.
- **851 and 314 are also the top false alarms in the LOMO-FS run** → systematic, not random.
- All 6 worst missed fakes are expression methods (F2F, NT), scores 0.001–0.07.
  **Pairs 073_024 and 233_995 are missed in both F2F AND NT** → the error depends on the target video, not the method:
  calm, frontal speakers, uniform background, little motion → small manipulation, few traces.
- Security consequence (FN): exactly the highest-risk scenario (calm, frontal talking-head video, e.g., a fake CEO
  in a video call) is the hardest to detect.
- Security consequence (FP) / RAI: real people with head coverings or glasses flagged as fake → possible bias against
  accessories/cultural clothing; not provable with 6 examples → frame it as a test to run (slice analysis).
  A false alarm can damage the credibility of a real person → human review is mandatory.

**mnv2_lomo_FS (FaceSwap unseen; threshold 0.302): 30 false alarms, 128 missed fakes**
- Missed FaceSwap fakes with P(fake) ≈ 0.000 (maximally confident "real"), although some show errors **visible to humans**:
  color patches/blending edges (527_454), a dark, mismatched face (485_425), distorted features (376_381).
- Interpretation: the model does not use the cues humans see but method-specific traces of the
  training methods ("shortcut"/fingerprint learning). Strong figure for generalization + presentation.

---

## 6. Pitfalls and Fixes (for limitations/reproducibility)

| Problem | Fix |
|---|---|
| Official servers refused connections (Oct 10–11) | Kaggle mirror for Interim Report 1; EU2 from Oct 16; `download.py` tries the servers in turn and keeps the one that works |
| Download progress invisible | `_run_live` streams the output (tqdm via `\r`) and confirms the terms-of-use prompt via stdin |
| Drive "credential propagation was unsuccessful" | Account mismatch / not all permissions confirmed → delete the runtime, pick the right account, tick all boxes |
| Connection lost during download | The Colab runtime is temporary → interim state copied to Drive with `rsync` and restored later |
| QuickTime shows black videos | yuv444p; files are fine, OpenCV reads them correctly |
| OpenCV 5 without Haar cascades | Check in setup, OpenCV < 5 |
| Haar detection slow (437 ms/frame) | Detection on ≤ 640 px (216 ms), built into `crops.py` |
| Patch overwritten by `%%writefile` | Code now lives in the repository module, not in notebook cells |
| BatchNorm collapse on tiny test data (constant eval outputs) | Artifact of too few training steps; did not occur with real data |
| Leakage assumption "same person" was wrong | Visual check: same studios, different people; the exclusion stays as a precaution |

---

## 7. Status Report: How Open Items Were Resolved

- Feedback on Interim Report 1 (−0.25 status, −0.25 unofficial data/subset, −0.25 model not started/small validation set;
  requested: official data, more data, the planned model, uncertainty on a source/method-disjoint test) answered
  point by point in Section 1 of the status report (validation 101 → 655 videos; bootstrap CIs; LOMO = method-disjoint).
- Class sample data and external dataset (FaceShifter) removed from the status report at the team's request;
  checklist says "LOMO only".
- Owners in Section 9 of the status report: Flemming = inference/latency, Grad-CAM, demo; Payel = robustness + defense,
  slice analysis of false alarms, repository (requirements, tests, model card, README); both = final report + slides.
- Status report finalized in the shared Google Doc; PDF in the repository as
  `Deepfake_Recognition_Interim_Report_2.pdf` (10 pages incl. references/checklist/appendix).

## 8. Inference and Latency (Oct 27, 2026, notebook 04, mnv2_v1, Colab T4, 2 CPU cores)

- `src/ffpp/infer.py`: `Detector(run_dir)` loads best.pt + the validation threshold from eval/summary.json;
  `score(video)` = extract_video (same crops as training) → VideoEvalSet (same 8 frames or clips as notebook 03)
  → P(fake), verdict, score per clip/frame, timings: t_extract (decoding + Haar + crops, CPU), t_prepare,
  t_model, t_total. Warm-up (CUDA + loading Haar) before timing.
- Notebook 04: video source `ffpp_sample` (official script, first N per class; the split column shows train/val/test),
  `upload` or `drive_folder`; cell 4 score timeline; cell 5 latency on GPU + CPU → runs/<RUN>/eval/latency.json.
- 15 FF++ videos (`ffpp_sample`, N = 3 per class: IDs 469/585/599 → train/val only, **no test** → not evidence
  of accuracy). All 15 correct (fakes 0.79–1.00; originals 0.002–0.115; threshold 0.311).
- Median per video (~13 s of video): t_extract 2.5 s (CPU: decoding + Haar + crops), t_prepare 14 ms,
  t_model **10 ms GPU / 291 ms CPU** (8 frames), total **2.55 s GPU / 2.79 s CPU** → about 5× faster than real time.
  The bottleneck is face detection, not the model. First GPU call 110 ms (warm-up outlier, median unaffected).
- Reported in Section 3 of the status report. Optional: the same for mnv2_gru_v1.

---

## 9. Next Deliverable (Dec 6): Final Report + Presentation

Plan = Table 8 of the status report:

| Due | Task | Owner | Rubric |
|---|---|---|---|
| done | Inference notebook and latency | Flemming | evaluation |
| Nov 22 | Robustness experiment (JPEG, downscaling, blur, noise on test fakes) + perturbation-augmented fine-tuning as defense, before/after | Payel | adversarial 4 |
| Nov 22 | Slice analysis of false alarms (annotate the 138 real test videos: head covering, glasses, face size, lighting; FPR per attribute with intervals) | Payel | RAI 4 |
| Nov 22 | Grad-CAM + score timeline (false alarms, FaceSwap misses) | Flemming | RAI/XAI |
| Nov 29 | Demo (upload video → score, timeline, heatmap) with selected **test** videos + backup recording | Flemming | demo 4 |
| Nov 29 | Repository: requirements file, tests, model card, setup instructions in README | Payel | code 2 |
| Dec 4 | Final report (template 04) + slides; monitoring workflow diagram | both | monitoring 4, writing 2 |

**Open decision:** the final-report rubric's cross-dataset criterion (2 points) explicitly asks for the class sample
data and an external dataset; LOMO alone does not cover it. Proposal: FaceShifter from the FF++ release (same access,
same pipeline, about 1–2 h in Colab).

**Cleanup:** delete `DRIVE/ffpp_partial` and empty the Drive trash. Optional: b2_cnn_v2 (30 epochs, samples 8).
