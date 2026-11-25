# Model Card: mnv2_robust (Deepfake Recognition, IS665 Team 6)

| Field | Summary |
|---|---|
| Model | MobileNetV2 (ImageNet weights) per 224 px face frame, one logit; video score = mean P(fake) over 8 frames. 2.23 M parameters. PyTorch. |
| Version / run | `runs/mnv2_robust` (Nov 25, 2026): settings of `mnv2_v1` + perturbation-augmented training (50% of samples: random JPEG q15–60, downscale 0.2–0.7, blur σ0.8–3.5). Seed 665. |
| Intended use | Triage of videos attached to high-risk requests (payments, identity checks). Output: score, per-clip timeline and Grad-CAM for a human analyst, who verifies through a second channel. |
| Out of scope | Automatic blocking or approval; legal proof; accusing individuals; surveillance or identification; audio deepfakes; very low-quality input (route to manual review). |
| Training data | FaceForensics++ c23 official train split (720 originals + their Deepfakes, Face2Face, FaceSwap, NeuralTextures fakes; about 3,600 videos). |
| Evaluation data | FF++ official test split, leakage-filtered (682 videos, 138 real); FaceShifter test split (136 fakes, external generator); leave-one-method-out runs; 9 post-processing attacks. |
| Decision threshold | 0.304 = highest recall with FPR ≤ 0.20 on validation (FPR 0.19 there). Re-choose on validation after every retraining. |
| Metrics (test) | Recall 0.961 at FPR 0.232; ROC-AUC 0.965; PR-AUC 0.991. Reference model `mnv2_v1`: 0.965 / 0.217 / 0.967. |
| Robustness | JPEG q15: recall 0.956 (mnv2_v1: 0.506). Downscale ×0.5: FPR 0.29 (mnv2_v1: 0.99). Gaussian noise (held out of training): FPR 0.96–0.99, not fixed. |
| Generalization | FaceShifter ROC-AUC 0.692 (mnv2_v1: 0.750), recall about 0.48–0.56. Held-out FF++ methods: recall 0.13–0.90. |
| Fairness / slices | No demographic labels. Real videos with blurry faces are flagged 33–35% vs. 13% for sharp faces (Wilson CIs in `report/quality_slices.csv`). Accessory effect (head covering, glasses) suspected from error gallery, untested. |
| Latency | ~2.5 s per 13 s video (Colab T4), ~2.8 s CPU-only; face detection dominates (model: 10 ms GPU / 291 ms CPU). |
| Limitations | Unseen generators caught about half the time; degraded real videos raise false alarms; one dataset family (studio news footage); one run per configuration; at a 1% base rate only ~4% of alerts are true. |
| Ethics / privacy | Wrongful flags can discredit real people. Only face crops are stored; no identities; FF++ used under research terms; no data in this repository. |
| Monitoring | Input-quality drift, share flagged (PSI > 0.2), FPR on a verified-real reference set (> 0.25), confirmed misses from new tools → review/retrain (see final report, Section 10). |
