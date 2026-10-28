# Deepfake Recognition Using CNN

This repository contains Team 6’s cybersecurity analytics project. The goal is to develop a deepfake-recognition system using a convolutional neural network (CNN).

The model will examine facial images or selected video frames and classify them as likely real or manipulated.

## Project Documents

* [Project proposal](Deepfake_Recognition_Project_Proposal.pdf)
* [Interim progress report 1](Deepfake_Recognition_Interim_Report_1.pdf)
* [Interim status report 2](Deepfake_Recognition_Interim_Report_2.pdf)

## Repository Structure

```
notebooks/
  FFpp_EDA_Colab.ipynb     Interim Report 1: data audit, EDA, baselines B0/B1 (executed, frozen)
  01_prepare_data.ipynb    videos -> face clips, stored once in Drive (restartable)
  02_train.ipynb           training with a checkpoint every epoch and automatic resume
  03_evaluate.ipynb        threshold on validation, metrics on test, per method
  04_inference.ipynb       score any video file (upload / Drive / FF++ sample), timeline, GPU/CPU latency
src/
  ff_eda.py                script of the interim analysis (used by FFpp_EDA_Colab.ipynb)
  ffpp/                    shared code of notebooks 01-03
    data.py                find videos, official splits, leakage exclusion
    download.py            official download script / Kaggle fallback
    crops.py               face clips per video (.npz cache)
    dataset.py             PyTorch datasets (frame or clip samples)
    models.py              b2_cnn, mnv2, mnv2_gru
    train.py               training loop, checkpoints, resume
    metrics.py, evaluate.py
    infer.py               end-to-end scoring of a new video file (same preprocessing as training)
```

## Training Pipeline (PyTorch, Google Colab)

Each stage writes its result to Google Drive and the next stage builds on it, so a change in one
stage never forces a rerun of the earlier ones:

```
01_prepare_data  ->  DRIVE/cache/crops_v1/   4 clips x 16 frames per video, 224 px JPEG (~3 GB for 5,000 videos)
02_train         ->  DRIVE/runs/<run>/       config.json, metrics.csv, last.pt (every epoch), best.pt
03_evaluate      ->  DRIVE/runs/<run>/eval/  summary.json, per_method_test.csv, predictions
04_inference     ->  DRIVE/runs/<run>/eval/  inference_examples.csv, latency.json
```

* Videos are downloaded into the Colab runtime only; Drive keeps the face clips.
* A run with the same name resumes after its last finished epoch; raising `epochs` trains longer.
  Changing the model, image size or clip length requires a new run name.
* Face boxes are fixed per clip, clips are resampled to about 25 fps, and augmentation is identical
  for all frames of a clip, so the pipeline itself adds no flicker between frames.
* Validation picks the decision threshold (highest recall with FPR <= 0.2); test is used only at the end.
* Leave-one-method-out: `EXCLUDE_METHOD` in notebook 02 hides one manipulation method from training and
  validation (incl. threshold choice); notebook 03 reports seen vs. unseen methods.

Working notes with all settings and results so far: [docs/PROJECT_NOTES_since_interim1.md](docs/PROJECT_NOTES_since_interim1.md).

## Why This Project Matters

Deepfakes can be used to:

* Impersonate family members or company leaders
* Trick people into sending money
* Steal or misuse someone’s identity
* Create harmful images for blackmail
* Spread false or misleading information

Our system is intended to provide an early warning before someone trusts or shares suspicious media. The result will support human review and should not be treated as legal proof.

## Project Objectives

* Prepare facial images from real and manipulated videos
* Train a basic CNN as the baseline model
* Use MobileNetV2 with transfer learning as the main model
* Classify images as real or manipulated
* Measure precision, recall, F1-score, ROC-AUC, false positives, and false negatives
* Create a simple Streamlit demonstration
* Display a confidence score and face heatmap

## Dataset

We use the FaceForensics++ research dataset (c23, all 5,000 videos). Official access has been granted; because the official servers were unreachable on Oct 10-11, 2026, the interim analysis ran on a 1,250-video subset of a Kaggle re-upload and will be re-run on the official download.

The dataset files will not be uploaded to this repository. Anyone using the dataset must request access and follow its terms of use.

## Planned Technologies

* Python
* PyTorch / torchvision (switched from Keras after Interim Report 1)
* MobileNetV2 (+ GRU over frame sequences)
* OpenCV
* Scikit-learn
* Streamlit
* Grad-CAM

## Team Members

### Payel Sarkar

* Dataset access and preparation
* Frame extraction and face detection
* Data-quality checks
* Robustness testing

### Flemming Grabowski

* CNN training
* Model evaluation
* Streamlit demonstration
* Heatmap and project documentation

Both team members will review the code, results, ethical concerns, final report, and presentation.

## Responsible AI

Deepfake-detection systems can make mistakes. A false positive may incorrectly label a real image as fake, while a false negative may fail to identify manipulated content.

To reduce these risks, we will:

* Report both false positives and false negatives
* Test compressed and resized images
* Clearly explain the model’s limitations
* Avoid making accusations based only on the model
* Require human review for important results
* Use only approved research data
* Avoid uploading private or harmful victim images

## Project Status

Interim Report 1 (Oct 11, 2026): the data pipeline, quality and leakage checks, EDA and two baselines are done on a 1,250-video subset. Simple frame-difference features barely beat chance (ROC-AUC 0.53), so temporal evidence has to be learned (MobileNetV2 + GRU). Next milestone: face crops for all 5,000 videos and the simple CNN baseline by Oct 18, 2026.

Interim Report 2 (Nov 8, 2026): all 5,000 official videos processed; CNN baseline, MobileNetV2 and MobileNetV2 + GRU trained and tested on the official test split (MobileNetV2: ROC-AUC 0.967, recall 0.965 at FPR 0.217; the GRU is not significantly better). Leave-one-method-out shows the main weakness: unseen manipulation methods are caught far less often (FaceSwap held out: ROC-AUC 0.41). Scoring one video takes about 2.5 s, mostly face detection. Next: robustness experiment with a defense, slice analysis of false alarms, Grad-CAM, demo, and repository hardening for the final report (Dec 6, 2026).

## AI Use Disclosure

ChatGPT and Claude were used for topic brainstorming, proposal organization, review, and wording support. Since Interim Report 1, Claude (Anthropic) also drafted the pipeline code (src/ffpp, notebooks 01–04) and report drafts. The team ran all experiments on the official data, read and tested the code before running it, checked every reported number against the saved run outputs, and remains responsible for the submission.

## References

* Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., and Nießner, M. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.*
* [FaceForensics++ official repository](https://github.com/ondyari/FaceForensics)
