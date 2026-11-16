# Real-Time ASL Sign Recognition Using Computer Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IykeOdum/Real-Time-ASL-Recognition/blob/main/ASL_Recognition_Baseline.ipynb)

**Team:** Iyke Odum, Ethan Frazier, Quinn Bankhead

This project explores recognizing American Sign Language (ASL) signs from hand landmarks. The long-term goal is real-time recognition from a webcam. The current stage is an offline baseline: classify a small set of signs from landmarks extracted from [WLASL](https://dlib.cs.umbc.edu/) video clips using MediaPipe.

## Status: Milestone 3 (baseline)

The notebook `ASL_Recognition_Baseline.ipynb` is an experimentation notebook, not a finished application. It currently:

- classifies **3 signs** (`cousin`, `thin`, `before`) from **one representative frame per video**;
- uses **MediaPipe hand landmarks** as features and a **K-nearest-neighbors** classifier;
- does **not** use a live webcam yet.

## Results

| Evaluation | Result |
|---|---|
| Held-out test set (12 videos, 4 per sign) | 12/12 correct (100%), macro F1 = 1.00 |
| Training-only 4-fold cross-validation, best K = 1 | 77.8% mean accuracy (std 13.6%) |

Cross-validation accuracy by K (training set only):

| K | Mean accuracy | Std |
|---|---|---|
| 1 | 0.778 | 0.136 |
| 2 | 0.583 | 0.144 |
| 3 | 0.639 | 0.164 |
| 4 | 0.611 | 0.124 |
| 5 | 0.611 | 0.124 |
| 6 | 0.556 | 0.079 |
| 7 | 0.528 | 0.092 |
| 8 | 0.444 | 0.111 |
| 9 | 0.444 | 0.079 |
| 10 | 0.389 | 0.056 |

**How to read these numbers.** The test set has only 12 videos, so a single mistake would move accuracy by about 8 points, and a perfect score is still consistent with a much lower true accuracy. The cross-validation figure is the more conservative estimate. Accuracy falls as K increases; with only 9 training videos per sign in each CV fold, any K above 9 must include neighbors from other signs.

## Pipeline

1. **Load data.** Download the WLASL video dataset from Kaggle (`waseemnagahhenes/sign-language-dataset-wlasl-videos`) with `kagglehub`. This copy has 12,505 videos across 2,000 signs, organized as one folder per sign, with no JSON annotation files and therefore no signer metadata.
2. **Select a balanced subset.** Take the 3 signs with the most videos (each with at least 12) and sample the same number of videos per sign (16 each, 48 total). With the current Kaggle copy these are `cousin`, `thin`, and `before`; if the source changes, the selection can change.
3. **Extract landmarks.** For each video, inspect frames at 40%, 50%, and 60% of its length with the MediaPipe Hand Landmarker (up to 2 hands, 21 landmarks each, `(x, y, z)` per landmark) and keep the frame with the most detected hands. Subtract the wrist position and divide by the largest wrist-to-landmark distance in the image plane. A missing second hand is zero-filled. This gives **126 features** per video (2 × 21 × 3). All 48 videos produced usable landmarks.
4. **Split.** One video is one row, so frames from the same video never land in both train and test. Stratified 75/25 split: 36 training and 12 test videos.
5. **Train and evaluate.** A `StandardScaler` + `KNeighborsClassifier` pipeline. K is chosen from 1–10 by stratified 4-fold cross-validation on the training set only (ties go to the smaller K), then the final model is fit on all training videos and evaluated once on the test videos. Metrics: accuracy, per-sign precision/recall/F1, and a confusion matrix.

## Known limitations

- **Very small sample.** 48 videos total, 12 in the test set.
- **No signer information.** The Kaggle copy has no annotation files, so splits are by video, not by signer. The same signer or recording setup may appear in both train and test, which can inflate accuracy. A signer-independent test set is needed for an unbiased final estimate.
- **No motion.** One frame per video discards hand movement, which matters for many ASL signs, as well as facial and body information.
- **Hand ordering.** The two hand slots follow image left-to-right position, not a consistent left/right-hand identity, and one-handed signs are zero-padded.
- **Video vs. webcam.** The model is trained and evaluated on dataset videos, so performance on live webcam input is untested.

## Running the notebook

Use Google Colab (the badge above opens the notebook). The first cell installs dependencies:

```
%pip -q install mediapipe opencv-python-headless scikit-learn pandas matplotlib seaborn kagglehub tqdm joblib
```

Then run the cells top to bottom. The dataset download is about 4.9 GB. Settings you can change at the top of the notebook:

| Setting | Default | Meaning |
|---|---|---|
| `NUM_SIGNS` | 3 | Number of signs to classify |
| `MIN_VIDEOS_PER_SIGN` / `MAX_VIDEOS_PER_SIGN` | 12 / 25 | Eligibility threshold and per-sign cap |
| `TEST_SIZE` | 0.25 | Fraction of videos held out for testing |
| `K_VALUES` / `CV_FOLDS` | 1–10 / 4 | KNN neighbor counts compared and CV folds |
| `SAMPLE_POSITIONS` | 0.4, 0.5, 0.6 | Relative video positions inspected for landmarks |
| `RANDOM_SEED` | 42 | Seed for sampling and splits |

## Generated outputs

The notebook writes these to `/content/asl_outputs` in Colab, which is erased when the session ends. Run the final cell to download them as a ZIP.

| File | Contents |
|---|---|
| `asl_landmarks.csv` | Extracted landmark features, one row per video (reusable without reprocessing videos) |
| `asl_knn_cross_validation.csv` / `.png` | Cross-validation results and plot |
| `asl_test_predictions.csv` | Per-video test predictions |
| `asl_evaluation_results.csv` | Per-sign precision, recall, F1 |
| `asl_confusion_matrix.png` | Confusion matrix on held-out videos |
| `asl_knn_model.pkl` | Fitted scaler + KNN model, for later webcam integration |
| `experiment_settings.json` | Settings used for the run |

## Next steps

- Increase the number of signs beyond 3 and measure how accuracy changes.
- Compare KNN against other models (SVM, a small neural network).
- Report more stable estimates (for example, repeated cross-validation) given the small dataset.
- Evaluate on unseen signers or a separate test set.
- Move from single-frame landmarks toward temporal features and a live webcam pipeline.