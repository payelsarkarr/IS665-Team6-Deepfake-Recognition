# Projekt-Notizen seit Interim Report 1 (Stand 17.10.2026)

Team 6, IS665: Flemming Grabowski, Payel Sarkar. Thema: Deepfake-Erkennung auf FaceForensics++ (FF++).
Zweck dieser Datei: alle Fakten, Zahlen und Entscheidungen seit der Abgabe von Interim Report 1 (11.10.2026),
damit Status Report (8.11.), Final Report und Präsentation (6.12.) darauf aufbauen können.

---

## 1. Abgaben und Rubriken

| Termin | Abgabe | Punkte | Umfang |
|---|---|---|---|
| 8. Nov 2026 | Interim Status Report (Template `03_Interim_Status_Report_Template.docx`) | 10 | 3–5 Seiten PDF |
| 6. Dez 2026 | Final Project Report (Template `04_Final_Project_Report_Template.docx`) | 40 | 8–12 Seiten PDF (ohne Titel, Refs, Anhang) |
| 6. Dez 2026 | In-Class Presentation (Template `05_In_Class_Presentation_Template.pptx`) | 20 | 10–12 min, Live-Demo + Backup |

**Status Report (10):** Fortschritt 1 · Modelle 2 · Metriken 2 · Fehleranalyse + Generalisierung 2 · Adversarial 1 · RAI + Monitoring 1 · Plan/Repo/Beiträge 1.
Inhalte: Updated Status; Model Development (Baseline + ≥2 Ansätze, Hyperparameter/Tuning, Training/Validierung,
Imbalance, Reproduzierbarkeit); Preliminary Evaluation (Primärmetrik begründet, Konfusionsmatrix, FPR/FNR,
ROC-AUC, PR-AUC, Latenz/Kosten); Error Analysis (repräsentative FP/FN, Ursachen, Sicherheitsfolgen);
Generalization (Klassen-Beispieldaten + ≥1 größerer/externer Datensatz, Schema-Probleme); Adversarial Threat
Model (Asset, Angreifer, Angriff, geplantes Abwehr-Experiment); Responsible AI (ungleiche Performance über
Gruppen, sonst andere Dimension z. B. Angriffstyp); Monitoring and Response Plan; Team Contribution.
Accuracy allein reicht nicht.

**Final Report (40):** Problem 2 · Data/EDA 4 · Model Dev 8 · Evaluation 8 · Cross-dataset 2 · Monitoring 4 ·
Adversarial 4 · RAI 4 · Code/Repro 2 · Writing 2.
Abschnitte: Executive Summary; Problem/Use Case; Data + Quality (Provenienz, Labels, Imbalance, Splits,
Leakage); EDA/Features; Model Development; Evaluation + Fehleranalyse (Ergebnistabelle, Konfusionsmatrix,
vs. Baseline, FP/FN, Trade-offs Erkennung/Fehlalarm/Komplexität/Speed/Interpretierbarkeit); Deep Analysis +
Generalization (Klassen-Beispieldaten + externer Datensatz, Schema/Drift dokumentieren); Interpretation/XAI;
Adversarial Robustness (Threat Model, kontrolliertes Experiment, Abwehr, vorher/nachher); RAI/Privacy/Governance;
Monitoring/Feedback/Response (Diagramm empfohlen); Limitations; Conclusion; Reproducibility (README, Environment-
Datei, Configs, Daten-Zugang, Tests, Model Card, AI-Use, Beiträge). Keine restricted Daten ins öffentliche Repo.

**Präsentation (20):** Problem 3 · Pipeline 4 · Ergebnisse 5 · Demo 4 · Limitationen/RAI 2 · Delivery 2.
Inhalte: Problem, Pipeline, Baseline + Modelle, Ergebnisse, FP/FN-Beispiele, Generalisierung, Adversarial/
Privacy/RAI, Monitoring, Limitationen, Live-Demo + Backup (Screenshots/Aufnahme).

---

## 2. Was seit Interim Report 1 passiert ist (Chronologie)

| Datum | Schritt |
|---|---|
| 11.10. | Interim Report 1 abgegeben (lief auf 1.250-Video-Teilmenge des Kaggle-Mirrors, weil die offiziellen Server down waren). |
| 13.–16.10. | Neue, modulare Pipeline gebaut (PyTorch statt Keras): `src/ffpp/` + Notebooks 01/02/03. Altes Notebook `FFpp_EDA_Colab.ipynb` bleibt eingefroren. |
| 16.10. | Offizieller Download wieder möglich (Server **EU2**). Neuer Google-Account, Drive leer. |
| 16./17.10. | Download aller 5.000 c23-Videos in die Colab-Laufzeit; Teil-Download zwischendurch per `rsync` nach `DRIVE/ffpp_partial` gesichert (Café-Verbindungsabbruch). |
| 16./17.10. Nacht | Gesichtsextraktion für alle Videos → `DRIVE/cache/crops_v1` (~3 GB). |
| 17.10. | Training b2_cnn_v1, mnv2_v1, mnv2_gru_v1 auf T4; Evaluation auf Test mit Val-Schwelle. |

---

## 3. Daten

- **FF++ c23**, offiziell (Zugang beantragt, Skript `faceforensics_download_v4.py`; Skript und persönlicher
  Link liegen nur in Drive, **nie im Repo**). 1.000 Originale + 4 Methoden je 1.000: Deepfakes (DF),
  Face2Face (F2F), FaceSwap (FS), NeuralTextures (NT). Real:Fake = 1:4.
- **Server** (Adressen stehen im privaten Download-Skript): EU (refused am 16.10.), **EU2 (funktionierte)**, CA. Kaggle-Mirror `xdxd003/ff-c23` nur für Interim 1; Professorin
  will offizielle Daten, Kaggle nur nach Absprache.
- **Videoformat:** H.264 High 4:4:4 (yuv444p). QuickTime zeigt schwarz, OpenCV liest korrekt. Keine Konvertierung.
- **Splits:** offizielle `train/val/test.json` (github ondyari/FaceForensics, `dataset/splits`): 360/70/70 Paare
  = 720/140/140 Original-IDs. Fakes `target_source` erben den Split.
- **Leakage-Ausschluss:** 22 YouTube-Quellen liefern 45 Sequenzen; 7 Gruppen liegen über Splits verteilt
  (605/606, 639/640, 755/756, 837/838, 875/876, 954/955, 982/983). Ausgeschlossen aus Val/Test: Originale
  606, 640, 755, 837, 875, 955, 982 + alle Fakes, die eine dieser IDs nutzen (63 Videos). Sichtprüfung:
  verschiedene Personen, gleiches Studio/Setting (837/838 Schwestern) → Risiko ist Hintergrund/Studio-Leakage,
  nicht Identitäts-Leakage. Im Train bleiben sie drin.
- **Effektive Größen nach Ausschluss:** Val 135 real + 130 je Methode (= 655), Test 138 real + 136 je Methode (= 682).
- **Extraktion (Notebook 01):** 4.999 ok, 1 `no_face`. Clips pro Video: {0:1, 1:2, 2:8, 3:43, 4:4946}.

### Gesichtsausschnitte (`crops.py`, `crops_v1`)
- Haar-Cascade (OpenCV 4.x; OpenCV 5 hat keine Cascades → Check im Setup) auf Frames, die auf max. 640 px
  verkleinert sind (437 → 216 ms/Frame).
- **Feste Box pro Clip** (kein Flackern), ×1,3 vergrößert, quadratisch, 224 px, JPEG q90.
- 4 Clips × 16 Frames pro Video, auf ~25 fps resampelt (`stride = round(fps/25)`).
- Ein `.npz` pro Video (jpeg, offsets, meta, starts), atomar geschrieben, Neustart überspringt fertige Dateien.
  `index.csv`: Manifest + cache_file, n_clips, status. `config.json` sichert die Extraktions-Settings.

---

## 4. Pipeline / Code (Repo-Arbeitskopie, Zip `IS665-Team6-training-pipeline.zip`)

```
src/ffpp/
  data.py      Manifest: Videos finden, offizielle Splits, Leakage-Flag (leak_excluded)
  download.py  offizielles Skript (Server-Fallback EU→EU2→CA, Live-Fortschritt), Kaggle-Fallback
  crops.py     Gesichts-Clips pro Video (.npz), parallel, restartable
  dataset.py   TrainSet (Frame- oder Clip-Samples, gleiche Augmentierung für alle Frames), VideoEvalSet
  models.py    b2_cnn, mnv2, mnv2_gru
  train.py     Trainingsloop, Checkpoint jede Epoche, Auto-Resume
  metrics.py   ROC-AUC, PR-AUC, Schwelle bei FPR ≤ x, Recall@FPR
  evaluate.py  Schwelle auf Val, Test-Metriken, pro Methode, Run-Vergleich
notebooks/ 01_prepare_data · 02_train · 03_evaluate   (generiert aus build_notebooks.py)
```

**Drive-Struktur** (`DRIVE_ROOT=/content/drive/MyDrive/IS665_Deepfake`):
```
faceforensics_download_v4.py     (privat)
cache/crops_v1/                  index.csv, config.json, videos/*.npz   (~3 GB)
runs/<run>/                      config.json, metrics.csv, last.pt, best.pt, val_best.csv
runs/<run>/eval/                 summary.json, per_method_test.csv, predictions_{val,test}.csv
ffpp_partial/                    alte Sicherung des Teil-Downloads → LÖSCHEN + Papierkorb leeren
```

**Modelle**
| Name | Architektur | Parameter |
|---|---|---|
| b2_cnn | 4 Conv-Blöcke von Grund auf (Baseline) | 1,17 M |
| mnv2 | MobileNetV2, ImageNet (IMAGENET1K_V2), Frame-Klassifikator | 2,23 M |
| mnv2_gru | MobileNetV2 pro Frame + bidirektionales GRU (hidden 256), Mittel über Zeit | 4,59 M |

**Training:** AdamW, Warmup + Cosine (aus Epoche berechnet → resume-sicher), BCE mit Klassengewichten
(Imbalance 1:4), AMP, Backbone in den ersten `freeze_backbone_epochs` eingefroren (BN in eval), Augmentierung
(Flip, Helligkeit, JPEG) identisch für alle Frames eines Samples, Seed 665. `last.pt` jede Epoche,
`best.pt` nach Val-ROC-AUC. Architektur-Guard (model, pretrained, img_size, clip_len) verhindert Resume mit
falscher Architektur. Clip-Modus: Batch = batch_size // clip_len.

**Evaluation:** Video-Score = Mittel P(fake) über 8 Eval-Frames bzw. die Clips. Schwelle auf **Val** gewählt
(höchster Recall bei FPR ≤ 0,20), unverändert auf **Test** angewandt. Pro Methode: Anteil als fake markiert
(= Recall, bei `original` = FPR).

**Primärmetrik (Begründung für den Report):** Recall bei FPR ≤ 0,20 (operativ: Fehlalarmbudget fix, möglichst
viele Fakes fangen) + ROC-AUC/PR-AUC als schwellenunabhängige Vergleichsmaße. Accuracy ist bei 1:4 irreführend
(immer „fake“ = 80 %).

---

## 5. Ergebnisse (17.10.2026, Colab T4)

### Lauf-Settings und Validierung
| Run | Bild | LR | Epochen | Frozen | Sonstiges | min/Epoche | Val-AUC (best) | Val Recall@FPR0.2 |
|---|---|---|---|---|---|---|---|---|
| b2_cnn_v1 | 128 | 1e-3 | 10 | – | batch 64, samples 4 | ~0,8 | 0,60 → **0,699** | 0,46 |
| mnv2_v1 | 224 | 3e-4 | 8 | 1 | batch 64, samples 4 | ~1,1 | **0,975** | 0,967 |
| mnv2_gru_v1 | 224 | 1e-4 | 6 | 1 | clip 16, samples 2 | ~7 | **0,977** | 0,967 |

- b2: Loss fällt noch → unterfittet (Kandidat b2_cnn_v2: 30 Epochen, samples 8).
- mnv2: Loss 0,69 → 0,165. GRU stieg am Ende noch leicht.

### Test (Schwelle aus Val)
| Modell | best Epoche | Test ROC-AUC | Recall | FPR | Schwelle |
|---|---|---|---|---|---|
| b2_cnn | 9 | 0,684 | 0,467 | 0,283 | 0,606 |
| mnv2 | 8 | 0,967 | 0,965 | 0,217 (30/138) | 0,311 |
| mnv2_gru | 6 | **0,976** | **0,976** | 0,246 (34/138) | 0,249 |

### Pro Methode (Test: 136 Fakes je Methode, 138 Originale)
| Modell | DF | F2F | FS | NT | original (FPR) |
|---|---|---|---|---|---|
| b2_cnn | 0,662 | 0,419 | 0,404 | 0,382 | 0,283 |
| mnv2 | 1,000 | 0,963 | 0,985 | 0,912 | 0,217 |
| mnv2_gru | 1,000 | 0,971 | 0,993 | 0,941 | 0,246 |

### Interpretation
- Transfer Learning ist der große Sprung (AUC 0,68 → 0,97). Die von Grund auf trainierte Baseline lernt in
  10 Epochen fast nur DF (die gröbste Methode).
- **Recall-Ziel erreicht, FPR-Ziel auf Test knapp verfehlt** (0,217 / 0,246 statt ≤ 0,20). Bei 138 Originalen
  ist 1 Video ≈ 0,7 pp, Stichprobenrauschen ±~3,5 pp. **Nicht** nachträglich auf Test tunen. Lehre: Schwelle
  mit Sicherheitsabstand wählen (z. B. Val-FPR ≤ 0,15 für Ziel 0,20) bzw. Konfidenzintervall berichten.
- GRU auf bekannten Methoden kaum besser (+0,009 AUC); eher Verschiebung entlang der ROC-Kurve. Kosten ~6×
  Trainingszeit. Ob die zeitliche Modellierung bei **unbekannten** Methoden hilft → LOMO.
- **NeuralTextures am schwierigsten** (subtile Textur-Änderung nur im Mundbereich), DF am leichtesten.
- Fehlalarme (FPR) sind der eigentliche Schwachpunkt → Fehleranalyse auf FP-Originale konzentrieren
  (Kompression, Licht, Kopfdrehung, kleine Gesichter?).


### Konfusionsmatrizen (Test, Val-Schwelle; 138 real, 544 fake)
| Modell | TN | FP | FN | TP | FPR | FNR |
|---|---|---|---|---|---|---|
| b2_cnn | 99 | 39 | 290 | 254 | 0,283 | 0,533 |
| mnv2 | 108 | 30 | 19 | 525 | 0,217 | 0,035 |
| mnv2_gru | 104 | 34 | 13 | 531 | 0,246 | 0,024 |

ROC-AUC pro Methode vs. Originale (Test): b2 DF 0,80 / F2F 0,67 / FS 0,64 / NT 0,62 · mnv2 0,987 / 0,966 / 0,978 / 0,938 ·
GRU 0,990 / 0,980 / 0,986 / 0,948.

**Basisraten-Effekt (wichtig für Report):** Test hat 80 % Fakes. Bei realistischer Rate von z. B. 1 % Fakes wäre die
Precision von mnv2 = 0,965·0,01 / (0,965·0,01 + 0,217·0,99) ≈ **4 %**, d. h. ~24 Fehlalarme pro echtem Treffer.
→ Modell taugt als Vorfilter mit menschlicher Prüfung, nicht als automatische Blockade; FPR ist der Hebel.

## 5b. Leave-one-method-out (17.10.2026, mnv2, gleiche Settings wie mnv2_v1, PATIENCE 3)
Ausgelassene Methode fehlt in Training **und** Val (also auch bei der Schwellenwahl). Test: 138 real + 136 je Methode.

| Run | best Ep. | Schwelle | AUC bekannte | AUC **unbekannte** | Recall bekannte | Recall **unbekannte** | FPR |
|---|---|---|---|---|---|---|---|
| mnv2_lomo_DF | 7 | 0,325 | 0,966 | **0,910** | 0,958 | **0,897** | 0,225 |
| mnv2_lomo_F2F | 8 | 0,386 | 0,972 | **0,782** | 0,961 | **0,544** | 0,167 |
| mnv2_lomo_NT | 5 | 0,295 | 0,988 | **0,723** | 0,971 | **0,368** | 0,058 |
| mnv2_lomo_FS | 8 | 0,302 | 0,971 | **0,414** | 0,975 | **0,132** | 0,217 |

Recall pro Methode in den LOMO-Runs (unbekannte fett): DF-Run DF **0,897**, F2F 0,956, FS 1,000, NT 0,919 ·
F2F-Run DF 0,978, F2F **0,544**, FS 0,978, NT 0,927 · FS-Run DF 1,000, F2F 0,978, FS **0,132**, NT 0,949 ·
NT-Run DF 0,978, F2F 0,949, FS 0,985, NT **0,368**.

**Interpretation**
- Kernbefund: ~0,97 AUC auf bekannten Methoden überschätzt die Leistung gegen neue Fälschungsverfahren stark.
  Unbekannte Methode: AUC 0,41–0,91, Recall 13–90 %. → zentral für Generalisierung, Threat Model und Limitations.
- **FaceSwap unbekannt: AUC 0,41 < 0,5** – das Modell hält FS-Fakes für *echter* als Originale. FS ist grafikbasiert
  (Landmarks + 3D-Modell + Farbangleich), DF ist lernbasiert (Autoencoder). Asymmetrie: FS gesehen → DF unbekannt
  wird gut erkannt (0,91); DF gesehen → FS nicht. Hypothese (mit Grad-CAM/Beispielen prüfen): Das Modell lernt
  Autoencoder-/Rendering-Spuren (Unschärfe, Textur), die FS nicht hat.
- Expression-Methoden (F2F, NT) übertragen nur teilweise aufeinander (0,78 bzw. 0,72).
- Bekannte Methoden bleiben gleich gut (0,97–0,99) → Ausschluss schadet dem Rest nicht.
- FPR schwankt zwischen Runs (0,058–0,225), weil jede Schwelle auf einem anderen Val-Set/Modell gewählt wird;
  die niedrige FPR des NT-Runs (8/138) ist Zufall der Schwellenwahl, nicht als Verbesserung verkaufen.
- Sicherheitsfolge: Ein Angreifer mit einem neuen Generator umgeht den Detektor mit hoher Wahrscheinlichkeit →
  Monitoring auf Score-Drift, regelmäßiges Nachtrainieren mit neuen Methoden, externer Datensatz als Pflicht-Test.
- Offen: GRU-LOMO für FS/NT (hilft Zeitinformation bei unbekannten Methoden?), externer Datensatz (FaceShifter ist
  ebenfalls Identity-Swap, aber lernbasiert → guter nächster Test).


## 5c. Fehleranalyse (17.10.2026, Zelle 7 in Notebook 03)
Bilder: `runs/<run>/eval/error_examples.png` + `.csv` in Drive (lokal: outputs/figures/). FF++-Frames **nicht** ins
öffentliche Repo; im Report-PDF als Abbildung mit Quellenangabe ok. Personen im Report nicht namentlich nennen.

**mnv2_v1 (Schwelle 0,311): 30 Fehlalarme, 19 verpasste Fakes**
- Top-Fehlalarme (P(fake) 0,79–0,96, Einzelframes bis 0,999 → sehr sicher falsch): Originale 851, 314, 725, 691,
  035, 257. Alle Nachrichten-/Studio-Material. Auffällig: Kopfbedeckung (851, Kufiya), Brille (314, 691), offener
  Mund mit Zähnen (725), gesenkter Blick/geschlossene Augen (035). Hypothese: ungewöhnliche Merkmale im
  Gesichtsbereich bzw. schwache Quellqualität sehen für das Modell wie Manipulationsspuren aus.
- **851 und 314 sind auch im LOMO-FS-Run die Top-Fehlalarme** → systematisch, nicht zufällig.
- Alle 6 schlimmsten verpassten Fakes sind Expression-Methoden (F2F, NT), Scores 0,001–0,07.
  **Paare 073_024 und 233_995 werden in F2F UND NT verpasst** → Fehler hängt am Zielvideo, nicht an der Methode:
  ruhige, frontale Sprecher, gleichmäßiger Hintergrund, wenig Bewegung → Manipulation klein, kaum Spuren.
- Sicherheitsfolge FN: Genau das Szenario mit größtem Risiko (ruhiges, frontales Talking-Head-Video, z. B. CEO-
  Fake im Videocall) ist am schwersten zu erkennen.
- Sicherheitsfolge FP / RAI: echte Personen mit Kopfbedeckung oder Brille als Fake markiert → möglicher Bias gegen
  Accessoires/kulturelle Kleidung; mit 6 Beispielen nicht belegbar → als Prüfauftrag (Slice-Analyse) formulieren.
  Ein Fehlalarm kann Glaubwürdigkeit einer echten Person beschädigen → menschliche Prüfung Pflicht.

**mnv2_lomo_FS (FaceSwap unbekannt; Schwelle 0,302): 30 Fehlalarme, 128 verpasste Fakes**
- Verpasste FaceSwap-Fakes mit P(fake) ≈ 0,000 (maximal sicher „echt“), obwohl teils **für Menschen sichtbare**
  Fehler: Farbflecken/Blending-Kante (527_454), dunkles, nicht passendes Gesicht (485_425), verzerrte Züge (376_381).
- Interpretation: Das Modell nutzt nicht die für Menschen sichtbaren Hinweise, sondern methodenspezifische Spuren der
  Trainingsmethoden („Shortcut“/Fingerprint-Lernen). Starkes Bild für Generalisierung + Präsentation.

---

## 6. Stolperfallen und Lösungen (für Limitations/Repro)

| Problem | Lösung |
|---|---|
| Offizielle Server refused (10./11.10.) | Kaggle-Mirror für Interim 1; ab 16.10. EU2; `download.py` probiert Server der Reihe nach und bleibt beim funktionierenden |
| Download-Fortschritt unsichtbar | `_run_live` streamt Ausgabe (tqdm über `\r`), bestätigt ToS-Prompt per stdin |
| Drive „credential propagation was unsuccessful“ | Konto-Mismatch / nicht alle Berechtigungen bestätigt → Laufzeit löschen, richtiges Konto, alle Häkchen |
| Verbindungsabbruch während Download | Colab-Laufzeit lokal ist flüchtig → Zwischenstand per `rsync` nach Drive, später zurückkopiert |
| QuickTime zeigt Videos schwarz | yuv444p, Dateien ok, OpenCV liest korrekt |
| OpenCV 5 ohne Haar-Cascades | Check im Setup, OpenCV < 5 |
| Haar langsam (437 ms/Frame) | Detektion auf ≤ 640 px (216 ms), fest in `crops.py` |
| Patch durch `%%writefile` überschrieben | Code liegt jetzt im Repo-Modul, nicht in Notebook-Zellen |
| BN-Kollaps bei Mini-Testdaten (eval-Outputs konstant) | Artefakt zu weniger Schritte; mit echten Daten nicht aufgetreten |
| Leakage-Annahme „gleiche Person“ falsch | Sichtprüfung: gleiche Studios, andere Personen; Ausschluss bleibt als Vorsicht |

---

## 7. Offene Punkte (Stand 17.10.)

**Für den Status Report (8.11.):**
1. ~~LOMO~~ erledigt 17.10. (Abschnitt 5b). Optional GRU-LOMO für FS und NT.
2. ~~Konfusionsmatrizen~~ erledigt (Abschnitt 5). Offen: ROC-/PR-Kurven als Abbildung.
3. ~~FP/FN-Beispiele~~ erledigt (Abschnitt 5c).
4. Latenz/Kosten: ms pro Video (Extraktion + Inferenz), Parameter, Trainingszeit.
5. Externer Datensatz: FaceShifter und/oder DeepFakeDetection (DFD) aus dem FF++-Skript (gleicher Zugang),
   evtl. Celeb-DF (eigener Zugangsantrag nötig). Klassen-Beispieldaten (prüfen, was die Professorin bereitstellt).
6. Robustheit/Adversarial: c40-Kompression, JPEG-Rekompression, Blur, Rauschen, Resize; Abwehr = Augmentierung
   bzw. Fine-Tuning mit Störungen, vorher/nachher.
7. RAI: Performance pro Methode und pro Qualitätsdimension (Gesichtsgröße, Helligkeit, Kompression); FF++ hat
   keine Demografie-Labels → offen benennen.
8. Monitoring- und Response-Plan (Score-Verteilung, Drift, Schwelle, menschliche Prüfung, Retraining-Trigger).
9. README + Code ins Repo (Payel ist Owner), Team-Beiträge ehrlich dokumentieren.

**Später (6.12.):** Grad-CAM + Score-Zeitleiste (XAI), Streamlit-Demo + Backup-Aufnahme, Model Card,
environment/requirements-Datei, Tests, Workflow-Diagramm, optional b2_cnn_v2.

**Aufräumen:** `DRIVE/ffpp_partial` löschen und Papierkorb leeren.

## 8. Status Report Entwurf (17.10.2026)
- Datei: `03_Interim_Status_Report_F-Grabowski_P-Sarkar.docx/.pdf` (9 Seiten inkl. Refs, Checkliste, Anhang;
  Abschnitte 1–9 ≈ 6,5 Seiten). Builder: `/home/claude/work/status/fill_status.py` (+ helpers.py, Abbildungen in fig/).
- Zusätzliche Kennzahlen (Test, Val-Schwelle, Bootstrap 1.000×, Seed 665):
  b2 P 0,867 R 0,467 F1 0,607 PR-AUC 0,884 ROC 0,684 [0,630; 0,733] ·
  mnv2 P 0,946 R 0,965 F1 0,955 PR-AUC 0,992 ROC 0,967 [0,952; 0,980] FPR 0,217 [0,155; 0,294] ·
  GRU P 0,940 R 0,976 F1 0,958 PR-AUC 0,994 ROC 0,976 [0,965; 0,986] FPR 0,246 [0,177; 0,315].
  Val-FPR an der Schwelle: mnv2 0,17, GRU 0,20. GRU vs. mnv2 nicht signifikant (CIs überlappen).
- Offene [TODO]s im Entwurf: Feedback IR1, Klassen-Beispieldaten, FaceShifter-Ergebnis, Inferenz-Latenz, Owner im
  Plan, Team-Beiträge (Payel), AI-Use-Ergänzungen.

## 5d. GRU-LOMO (19.10.2026; Settings wie mnv2_gru_v1, ~5,7 min/Epoche)
| Run | best Ep. | Schwelle | AUC bekannte | AUC unbekannte [95 %-CI] | Recall unbekannte | FPR |
|---|---|---|---|---|---|---|
| mnv2_lomo_FS | 8 | 0,302 | 0,971 | 0,414 [0,351; 0,480] | 0,132 | 0,217 |
| **mnv2_gru_lomo_FS** | 5 | 0,178 | 0,976 | **0,339** [0,277; 0,400] | 0,059 | 0,203 |
| mnv2_lomo_NT | 5 | 0,295 | 0,988 | 0,723 [0,660; 0,780] | 0,368 | 0,058 |
| **mnv2_gru_lomo_NT** | 6 | 0,108 | 0,990 | **0,748** [0,688; 0,806] | 0,559 | 0,196 |

Gepaarter Bootstrap (gleiche Videos, 2.000×), AUC GRU − mnv2 auf der unbekannten Methode:
FS −0,075 [−0,133; −0,019] (GRU signifikant **schlechter**), NT +0,025 [−0,024; 0,074] (kein Unterschied).
Recall bei gleicher Test-FPR ≤ 0,2 (nur beschreibend): FS mnv2 0,118 / GRU 0,051; NT 0,537 / 0,603.
**Fazit:** Zeitliche Modellierung hilft nicht bei unbekannten Methoden. Das GRU lernt die zeitlichen Spuren der
gesehenen Methoden, kein allgemeines Flackern. Höherer NT-Recall des GRU kommt vor allem vom lockereren
Arbeitspunkt (FPR 0,20 vs. 0,06). Im Status Report (Abschnitt 5, Tabelle 4) eingetragen. DF/F2F-GRU-LOMO nicht gelaufen.

## 9. Status Report: TODOs bereinigt (27.10.2026)
- Feedback IR1 (−0,25 Status, −0,25 unoffizielle Daten/Subset, −0,25 Modell nicht gestartet/kleine Val;
  Wunsch: offizielle Daten, mehr Daten, geplantes Modell, Unsicherheit auf quell-/methodendisjunktem Test) in
  Abschnitt 1 Punkt für Punkt beantwortet (Val 101 → 655 Videos; Bootstrap-CIs; LOMO = methodendisjunkt).
- Klassen-Beispieldaten und externer Datensatz (FaceShifter) auf Wunsch überall entfernt; Checkliste „LOMO only“.
- Owner Abschnitt 9: Flemming = Inferenz/Latenz, Grad-CAM, Demo; Payel = Robustheit + Abwehr, Slice-Analyse
  Fehlalarme, Repo (requirements, Tests, Model Card, README); beide = Final Report + Slides.
- Offene TODOs: Latenzwert (Notebook 04), Team-Beiträge/Koordination, AI-Use-Ergänzungen, Commit-Link.

## 10. Notebook 04_inference + src/ffpp/infer.py (27.10.2026)
- `Detector(run_dir)`: lädt best.pt + Val-Schwelle aus eval/summary.json; `score(video)` = extract_video
  (gleiche Crops wie Training) → VideoEvalSet (gleiche 8 Frames bzw. Clips wie Notebook 03) → P(fake), Urteil,
  Score pro Clip/Frame, Zeiten: t_extract (Decodieren + Haar + Crops, CPU), t_prepare, t_model, t_total.
  Warm-up (CUDA + Haar laden) vor dem Messen.
- Notebook: Videoquelle `ffpp_sample` (offizielles Skript, erste N pro Klasse; Split-Spalte zeigt Train/Val/Test),
  `upload` oder `drive_folder`; Zelle 4 Zeitleiste; Zelle 5 Latenz GPU+CPU → runs/<RUN>/eval/latency.json.
- Lokal mit synthetischen Videos getestet (Frame- und Clip-Modell).

## 12. Status Report abgabefertig (27.10.2026)
- Finale Fassung im Google Doc (1bZuGvKIn0bgKZL7JpWjJXbzJWRzCEodfCHaEnaJzmpA), PDF im Repo als
  `Deepfake_Recognition_Interim_Report_2.pdf` (10 Seiten inkl. Refs/Checkliste/Anhang). Keine TODOs mehr.
- Nächste Abgabe 6.12.: Final Report (Template 04) + Präsentation mit Live-Demo. Plan = Tabelle 8 im Report.
  Offene Entscheidung: Cross-dataset (2 P.) verlangt Klassen-Beispieldaten + externen Datensatz; Vorschlag FaceShifter.
