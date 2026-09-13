# Deepfake Recognition Using CNN

This repository contains Team 6’s cybersecurity analytics project. The goal is to develop a simple deepfake-recognition system using a convolutional neural network (CNN).

The model will examine facial images or selected video frames and classify them as likely real or manipulated.

## Why This Project Matters

Deepfakes can be used to:

* Impersonate family members or company leaders
* Trick people into sending money
* Steal or misuse someone’s identity
* Create harmful images for blackmail
* Spread false or misleading information

Our system is intended to provide an early warning before a person trusts or shares suspicious media. Its result should support human review and should not be treated as legal proof.

## Project Objectives

* Prepare facial images from real and manipulated videos
* Train a basic CNN as the baseline model
* Use MobileNetV2 with transfer learning as the main model
* Classify images as real or manipulated
* Measure precision, recall, F1-score, ROC-AUC, false positives, and false negatives
* Create a simple Streamlit demonstration
* Display a confidence score and face heatmap

## Dataset

We plan to use the FaceForensics++ research dataset. The project will use an approved subset because the complete dataset is too large for a two-person class project.

The dataset files will not be uploaded to this repository. Anyone using the dataset must request access and follow its terms of use.

## Planned Technologies

* Python
* TensorFlow and Keras
* MobileNetV2
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

This project is currently in the proposal and planning stage. Code, notebooks, results, and demonstration instructions will be added as the project develops.

## AI Use Disclosure

ChatGPT and Claude were used for topic brainstorming, proposal organization, review, and wording support. The team will verify all technical claims and citations, revise the work in its own voice, and remain responsible for the final submission.

## References

* Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., and Nießner, M. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.*
* [FaceForensics++ official repository](https://github.com/ondyari/FaceForensics)
