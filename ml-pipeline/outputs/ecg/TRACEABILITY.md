# Traceability report — ecg tier

Run: 2026-10-27T09:34:38 → 2026-10-27T09:51:16 · seed 42 · Python 3.14.7 · scikit-learn 1.9.1

## Inputs

| File | Description | SHA-256 |
|---|---|---|
| `data/raw/ecg/mitdb.zip` | PhysioNet mitdb v1.0.0 (ODC-By 1.0) | `47b26926927c11bd…` |
| `data/raw/ecg/svdb.zip` | PhysioNet svdb v1.0.0 (ODC-By 1.0) | `ef0cf2aae4523410…` |

## Data lineage

| # | Step | Rows before | Rows after | Removed | Detail |
|---|---|---|---|---|---|
| 1 | Segment beats (mitdb) | 0 | 100,630 | -100,630 | 44 records read, 4 paced records excluded (102, 104, 107, 217) |
| 2 | Segment beats (svdb) | 0 | 184,347 | -184,347 | 78 records read, 0 paced records excluded (none) |
| 3 | Remove non-finite values | 284,977 | 284,977 | 0 | NaN or infinite values in any numeric column |
| 4 | Filter implausible RR intervals (20-300 bpm) | 284,977 | 284,887 | 90 | pre_rr outside [0.2, 3.0]: 44; post_rr outside [0.2, 3.0]: 46 |
| 5 | Balance classes (random undersampling) | 284,887 | 65,414 | 219,473 | class counts {0: 252180, 1: 32707} -> {0: 32707, 1: 32707} |
| 6 | Stratified train/test split | 65,414 | 52,331 | 13,083 | train 52,331 / test 13,083 (20% held out, seed 42) |

## Parameters

```json
{
  "mode": "full",
  "ecg_extraction": {
    "fs_target_hz": 128,
    "window_s": [
      0.25,
      0.45
    ],
    "samples_per_beat": 90,
    "bandpass_hz": [
      0.5,
      40.0
    ],
    "filter": "Butterworth order 3, zero-phase (sosfiltfilt)",
    "local_rr_beats": 10,
    "lead": "channel 0 (MLII in MIT-BIH)",
    "peak_alignment": "largest |band-passed| deflection within \u00b175 ms",
    "peak_detection_in_app": "XQRS (wfdb.processing) + the same alignment",
    "normalisation": "z-score per beat",
    "label": "AAMI EC57: N = normal, S/V/F = abnormal"
  },
  "patients": 121,
  "model features": "100 columns (['pre_rr', 'post_rr', 'local_rr', 'rr_ratio', 'post_pre_ratio', 'amplitude_mv'] ...)",
  "Logistic Regression best hyperparameters": {
    "C": 56.69849511478853
  },
  "K-Nearest Neighbors best hyperparameters": {
    "n_neighbors": 40,
    "p": 2,
    "weights": "distance"
  },
  "Random Forest best hyperparameters": {
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "max_depth": 16
  },
  "Support Vector Machine best hyperparameters": {
    "C": 0.3511356313970407,
    "gamma": 0.0035498788321965025
  },
  "stacking": {
    "cv_folds": 5,
    "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
    "meta_features": [
      "LR",
      "KNN",
      "RF",
      "SVM"
    ],
    "referral_band": [
      0.35,
      0.65
    ],
    "grouped_cv": true
  },
  "meta-model weights": {
    "intercept": -3.398022981408816,
    "LR": 1.6190051217806056,
    "KNN": 0.0,
    "RF": 6.265795050512607,
    "SVM": 0.0
  },
  "patient-CV fold 1 stacking": {
    "cv_folds": 5,
    "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
    "meta_features": [
      "LR",
      "KNN",
      "RF",
      "SVM"
    ],
    "referral_band": [
      0.35,
      0.65
    ],
    "grouped_cv": true
  },
  "patient-CV fold 1 meta-model weights": {
    "intercept": -3.500749331503083,
    "LR": 1.585791419693598,
    "KNN": 0.049088762984424036,
    "RF": 6.442538303369531,
    "SVM": 0.0
  },
  "patient-CV fold 2 stacking": {
    "cv_folds": 5,
    "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
    "meta_features": [
      "LR",
      "KNN",
      "RF",
      "SVM"
    ],
    "referral_band": [
      0.35,
      0.65
    ],
    "grouped_cv": true
  },
  "patient-CV fold 2 meta-model weights": {
    "intercept": -3.415372482039555,
    "LR": 0.6179503484516568,
    "KNN": 0.0,
    "RF": 7.219484777176931,
    "SVM": 0.0
  },
  "patient-CV fold 3 stacking": {
    "cv_folds": 5,
    "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
    "meta_features": [
      "LR",
      "KNN",
      "RF",
      "SVM"
    ],
    "referral_band": [
      0.35,
      0.65
    ],
    "grouped_cv": true
  },
  "patient-CV fold 3 meta-model weights": {
    "intercept": -3.2873335768146656,
    "LR": 0.3819281106374475,
    "KNN": 0.0,
    "RF": 7.468598632943624,
    "SVM": 0.0
  },
  "patient-CV fold 4 stacking": {
    "cv_folds": 5,
    "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
    "meta_features": [
      "LR",
      "KNN",
      "RF",
      "SVM"
    ],
    "referral_band": [
      0.35,
      0.65
    ],
    "grouped_cv": true
  },
  "patient-CV fold 4 meta-model weights": {
    "intercept": -3.1531902097397286,
    "LR": 1.6498655546818675,
    "KNN": 0.0,
    "RF": 5.928240643535361,
    "SVM": 0.0
  },
  "patient-CV fold 5 stacking": {
    "cv_folds": 5,
    "meta_model": "LogisticRegression(C=1.0), weights beta_k >= 0",
    "meta_features": [
      "LR",
      "KNN",
      "RF",
      "SVM"
    ],
    "referral_band": [
      0.35,
      0.65
    ],
    "grouped_cv": true
  },
  "patient-CV fold 5 meta-model weights": {
    "intercept": -3.5816607409738976,
    "LR": 1.942473095898521,
    "KNN": 0.23499019217731523,
    "RF": 5.225524899791791,
    "SVM": 0.808734999380981
  }
}
```

## Metrics (held-out test set)

|                                        |   cv_accuracy |   tuning_rows |   oof_seconds |   final_fit_seconds |   rows_used_final_fit |   accuracy |   balanced_accuracy |   precision |   recall_sensitivity |   specificity |         f1 |    roc_auc |     pr_auc |        mcc |       brier |   tp |   fp |   tn |   fn |   stacking_accuracy |   accuracy_95ci_low |   accuracy_95ci_high | best_base_model   |   mcnemar_vs_best_base_p |   referral_rate |   accuracy_on_non_referred |   meets_95_target |   test_beats_from_training_patients |   nearest_training_beat_same_patient |   patients_in_train |   patients_in_test |   accuracy_mean |   accuracy_std |   recall_sensitivity_mean |   recall_sensitivity_std |   specificity_mean |   specificity_std |   roc_auc_mean |   roc_auc_std |   natural_accuracy_mean |   natural_accuracy_std |   natural_balanced_accuracy_mean |   natural_balanced_accuracy_std |   detection_sensitivity |   detection_ppv |   classified_beats | record    |   beats |   true_abnormal_burden |   predicted_abnormal_burden |   beat_accuracy |
|:---------------------------------------|--------------:|--------------:|--------------:|--------------------:|----------------------:|-----------:|--------------------:|------------:|---------------------:|--------------:|-----------:|-----------:|-----------:|-----------:|------------:|-----:|-----:|-----:|-----:|--------------------:|--------------------:|---------------------:|:------------------|-------------------------:|----------------:|---------------------------:|------------------:|------------------------------------:|-------------------------------------:|--------------------:|-------------------:|----------------:|---------------:|--------------------------:|-------------------------:|-------------------:|------------------:|---------------:|--------------:|------------------------:|-----------------------:|---------------------------------:|--------------------------------:|------------------------:|----------------:|-------------------:|:----------|--------:|-----------------------:|----------------------------:|----------------:|
| tuning Logistic Regression             |      0.900038 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| tuning K-Nearest Neighbors             |      0.801394 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| tuning Random Forest                   |      0.898338 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| tuning Support Vector Machine          |      0.882369 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| training time LR                       |    nan        |           nan |       2.93797 |           0.352732  |                 52331 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| training time KNN                      |    nan        |           nan |       1.83508 |           0.036057  |                 52331 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| training time RF                       |    nan        |           nan |      53.8225  |          11.105     |                 52331 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| training time SVM                      |    nan        |           nan |      57.2265  |           5.8534    |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| Logistic Regression                    |    nan        |           nan |     nan       |         nan         |                   nan |   0.93289  |            0.932889 |    0.942353 |             0.922183 |      0.943595 |   0.932159 |   0.978871 |   0.97834  |   0.865978 |   0.050619  | 6032 |  369 | 6173 |  509 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| K-Nearest Neighbors                    |    nan        |           nan |     nan       |         nan         |                   nan |   0.958419 |            0.958418 |    0.970058 |             0.946033 |      0.970804 |   0.957895 |   0.991672 |   0.990681 |   0.91712  |   0.0323508 | 6188 |  191 | 6351 |  353 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| Random Forest                          |    nan        |           nan |     nan       |         nan         |                   nan |   0.974012 |            0.974012 |    0.969133 |             0.979208 |      0.968817 |   0.974144 |   0.995872 |   0.99571  |   0.948075 |   0.0229193 | 6405 |  204 | 6338 |  136 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| Support Vector Machine                 |    nan        |           nan |     nan       |         nan         |                   nan |   0.943132 |            0.943131 |    0.955238 |             0.929827 |      0.956435 |   0.942361 |   0.983151 |   0.983916 |   0.886578 |   0.0423344 | 6082 |  285 | 6257 |  459 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| Hard voting                            |    nan        |           nan |     nan       |         nan         |                   nan |   0.964993 |            0.964993 |    0.966416 |             0.963461 |      0.966524 |   0.964936 | nan        | nan        |   0.92999  | nan         | 6302 |  219 | 6323 |  239 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| Soft voting                            |    nan        |           nan |     nan       |         nan         |                   nan |   0.964916 |            0.964916 |    0.966554 |             0.963155 |      0.966677 |   0.964852 |   0.993803 |   0.993676 |   0.929838 |   0.0295468 | 6300 |  218 | 6324 |  241 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| Stacking ensemble                      |    nan        |           nan |     nan       |         nan         |                   nan |   0.968662 |            0.968663 |    0.957742 |             0.980584 |      0.956741 |   0.969029 |   0.995027 |   0.99487  |   0.93759  |   0.0246693 | 6414 |  283 | 6259 |  127 |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| system summary                         |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |            0.968662 |            0.965604 |             0.971795 | Random Forest     |              4.82542e-13 |       0.0398227 |                   0.978029 |                 1 |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| leakage audit (beat-level split)       |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                   1 |                                0.944 |                 121 |                121 |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 1 training time LR     |    nan        |           nan |       2.27381 |           0.279771  |                 52658 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 1 training time KNN    |    nan        |           nan |       1.59801 |           0.0187277 |                 52658 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 1 training time RF     |    nan        |           nan |      51.1833  |          10.7425    |                 52658 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 1 training time SVM    |    nan        |           nan |      50.5324  |           5.05553   |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 2 training time LR     |    nan        |           nan |       2.09147 |           0.297685  |                 53072 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 2 training time KNN    |    nan        |           nan |       1.66766 |           0.0263476 |                 53072 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 2 training time RF     |    nan        |           nan |      51.7408  |          10.8748    |                 53072 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 2 training time SVM    |    nan        |           nan |      57.3061  |           5.83928   |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 3 training time LR     |    nan        |           nan |       2.83537 |           0.517125  |                 52512 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 3 training time KNN    |    nan        |           nan |       1.56553 |           0.0180601 |                 52512 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 3 training time RF     |    nan        |           nan |      51.0542  |          10.7148    |                 52512 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 3 training time SVM    |    nan        |           nan |      58.2376  |           5.92507   |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 4 training time LR     |    nan        |           nan |       2.58786 |           0.505522  |                 52556 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 4 training time KNN    |    nan        |           nan |       1.92344 |           0.0161974 |                 52556 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 4 training time RF     |    nan        |           nan |      44.2553  |           9.08831   |                 52556 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 4 training time SVM    |    nan        |           nan |      52.226   |           5.32289   |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 5 training time LR     |    nan        |           nan |       1.34265 |           0.241427  |                 50858 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 5 training time KNN    |    nan        |           nan |       2.36448 |           0.0175819 |                 50858 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 5 training time RF     |    nan        |           nan |      42.3757  |           8.79501   |                 50858 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-CV fold 5 training time SVM    |    nan        |           nan |      48.4566  |           4.98408   |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-wise CV K-Nearest Neighbors    |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |        0.813086 |      0.0514564 |                  0.73813  |                0.103061  |           0.888042 |        0.0685594  |       0.887837 |     0.0419121 |                0.923016 |              0.0359846 |                         0.770198 |                       0.0656388 |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-wise CV Logistic Regression    |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |        0.913954 |      0.0451312 |                  0.888913 |                0.095076  |           0.938995 |        0.0112278  |       0.96094  |     0.0278366 |                0.945205 |              0.0190906 |                         0.842074 |                       0.0693885 |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-wise CV Random Forest          |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |        0.894536 |      0.063594  |                  0.841306 |                0.119582  |           0.947766 |        0.00990036 |       0.9688   |     0.0203703 |                0.945758 |              0.0239902 |                         0.791438 |                       0.0773031 |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-wise CV Stacking ensemble      |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |        0.904626 |      0.0442109 |                  0.878936 |                0.0853611 |           0.930316 |        0.0167411  |       0.969455 |     0.0201954 |                0.952463 |              0.0202804 |                         0.879415 |                       0.0671897 |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| patient-wise CV Support Vector Machine |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |        0.890271 |      0.0523998 |                  0.836813 |                0.101654  |           0.943728 |        0.0113564  |       0.959846 |     0.023608  |                0.94975  |              0.0231626 |                         0.845378 |                       0.0718361 |              nan        |      nan        |                nan | nan       |     nan |             nan        |                  nan        |      nan        |
| automatic detection (unseen patients)  |    nan        |           nan |     nan       |         nan         |                   nan |   0.94304  |            0.880535 |  nan        |             0.800032 |      0.961038 | nan        |   0.940765 | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |                0.995349 |        0.993614 |              55741 | nan       |     nan |             nan        |                  nan        |      nan        |
| monitoring session (unseen patient)    |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan         |  nan |  nan |  nan |  nan |          nan        |          nan        |           nan        | nan               |            nan           |     nan         |                 nan        |               nan |                                 nan |                              nan     |                 nan |                nan |      nan        |    nan         |                nan        |              nan         |         nan        |      nan          |     nan        |   nan         |              nan        |            nan         |                       nan        |                     nan         |              nan        |      nan        |                nan | mitdb_201 |    1881 |               0.134503 |                    0.239766 |        0.835194 |

## Figures

Each figure with its reading guide (what it shows · how to read it).

1. `fig_01_ecg_etl_example.png` — **ETL of an unstructured ECG signal: raw vs. filtered/resampled (MIT-BIH 208)**
   - *What it shows:* Top: the original signal with noise and wandering baseline; bottom: the filtered (0.5–40 Hz) and resampled 128 Hz signal, with the cardiologist's label on each beat.
   - *How to read it:* Red letters are abnormal beats (V ventricular, A/S supraventricular, F fusion). This is how an unstructured signal becomes table rows.
2. `fig_02_cleaning_funnel.png` — **Rows (beats) remaining after each ETL step**
   - *What it shows:* Each bar is the dataset after one cleaning step; in brackets, how many rows that step removed.
   - *How to read it:* Read it top to bottom. No row disappears without a reason: each step is explained in TRACEABILITY.md.
3. `fig_03_beat_counts.png` — **Heartbeats per database and AAMI class before balancing**
   - *What it shows:* Number of beats of each AAMI class (N normal, S supraventricular, V ventricular, F fusion), on a log scale.
   - *How to read it:* Each axis tick is ×10. Normal beats are the vast majority: that is why balancing is needed.
4. `fig_04_class_balance.png` — **Class balance before and after undersampling**
   - *What it shows:* How many cases of each class there are. All cases of the rarer class are kept and the same number is drawn at random from the common one (undersampling).
   - *How to read it:* On the right both bars must be equal (50/50). Accuracy cannot then be inflated by always answering the common class.
5. `fig_05_mean_beats.png` — **Mean beat shape per AAMI class**
   - *What it shows:* Mean ± 1 standard deviation of the beat shape; time 0 is the R peak.
   - *How to read it:* The ventricular beat (V) is wide and deformed. The supraventricular (S) looks normal: that is why rhythm (when it arrives) is also used.
6. `fig_06_rr_features.png` — **RR and amplitude features by class**
   - *What it shows:* Violins of the rhythm (RR interval), amplitude and patient-relative features.
   - *How to read it:* Example: abnormal beats have rr_ratio < 1 (they arrive early) and lower similarity to the usual beat (template_corr).
7. `fig_07_distributions.png` — **Histograms and KDE of rhythm features by class**
   - *What it shows:* Histogram + KDE curve of the rhythm and similarity-to-usual-beat features.
   - *How to read it:* Curves that sit apart mark features that tell normal from abnormal beats.
8. `fig_08_correlations.png` — **Pearson vs. Spearman correlation of rhythm features**
   - *What it shows:* Pearson measures linear association; Spearman, ordered association (uses ranks, robust to outliers). From −1 to 1.
   - *How to read it:* Red: they rise together. Blue: one rises as the other falls. White: no relation. Look at the target's row.
9. `fig_09_vif_rhythm_features.png` — **Multicollinearity of the rhythm features (VIF)**
   - *What it shows:* VIF = 1 / (1 − R²) of the rhythm and amplitude features.
   - *How to read it:* Green < 5 fine, red > 10 severe. Tree models and regularisation tolerate moderate redundancy.
10. `fig_10_pca_projection.png` — **2-D PCA projection of all beat features by class**
   - *What it shows:* PCA summarises all variables in two axes that keep as much variation as possible.
   - *How to read it:* If the colours form separate clouds, the classes are easy to separate; if they mix, the problem is hard.
11. `fig_11_metric_comparison.png` — **Test-set metrics of the four base models, voting schemes and the stacking ensemble**
   - *What it shows:* Accuracy, precision, sensitivity, specificity, F1 and AUC on test beats not used for training, but from patients who did give other beats to training (split by beat).
   - *How to read it:* Higher is better; the red line is the 95 % target. This is the official evaluation (standard MIT-BIH protocol). The additional analysis on completely new patients is in the 'Leakage check' figure.
12. `fig_12_confusion_matrices.png` — **Confusion matrices on the held-out test set**
   - *What it shows:* 2×2 table per model: rows = truth, columns = prediction. Each row adds up to 100 %.
   - *How to read it:* The diagonal holds the hits. The bottom-left cell is the sick cases the model missed: the most dangerous error.
13. `fig_13_roc_pr_curves.png` — **ROC and precision-recall curves**
   - *What it shows:* How hits and false alarms change as the decision threshold moves. The area under the curve (AUC) sums it up.
   - *How to read it:* A curve hugging the top-left (ROC) or top-right (PR) corner is better. The dotted diagonal is chance (AUC 0.5).
14. `fig_14_calibration.png` — **Reliability of the predicted probabilities**
   - *What it shows:* Cases are grouped by predicted probability and compared with the real share of positives.
   - *How to read it:* On the diagonal, '70 %' really means 7 out of 10 such cases are positive. Below it, the model over-estimates.
15. `fig_15_meta_weights.png` — **Coefficients learned by the meta-model for each base model**
   - *What it shows:* Weights β learned by the meta-model: P = σ(β₀ + Σ βₖ·pₖ). They are ≥ 0 by design.
   - *How to read it:* Taller bar = the final answer listens more to that model. A weight of 0 means the others already carry its information.
16. `fig_16_model_diversity.png` — **Diversity between base models (OOF correlation and disagreement)**
   - *What it shows:* Left: correlation between the probabilities of each pair of models. Right: % of cases where they disagree.
   - *How to read it:* Combining models only helps if they sometimes disagree: correlations below 1 and non-zero disagreement mean complementary models.
17. `fig_17_probability_distribution.png` — **Distribution of the final probability per true class and referral band**
   - *What it shows:* Histogram of the system's probability, one colour per true class. The grey band is the doubt zone.
   - *How to read it:* Ideal: one colour near 0 and the other near 1. Errors live where colours overlap; cases in the grey band go to a clinician.
18. `fig_18_coverage_curve.png` — **Accuracy when the system answers only its most confident cases**
   - *What it shows:* Cases are sorted from most to least confident; for each share answered (coverage) the accuracy is computed.
   - *How to read it:* Towards the left the system answers fewer cases and gets more right. Where the curve crosses the 95 % line tells which share of cases it can solve on its own at that accuracy.
19. `fig_19_permutation_importance.png` — **Permutation importance of each feature group for the ensemble**
   - *What it shows:* Permutation importance: how much accuracy drops when one input is randomly shuffled.
   - *How to read it:* Longer bar = the system depends more on that input. Bars near 0 = it could be removed without losing accuracy.
20. `fig_20_simulation_test_beats.png` — **Per-model and final probability for 12 held-out beats of every AAMI class**
   - *What it shows:* Beats of each type (N, S, V, F) not used for training (split by beat), with each model's probability and the final answer.
   - *How to read it:* Blue = normal, red = abnormal. '(WRONG)' marks the errors; S beats are the hardest.
21. `fig_21_leakage_check.png` — **Accuracy with a beat-level split (known patients) vs. patient-wise CV (new patients)**
   - *What it shows:* Accuracy of each model with two ways of splitting the data: at random by beat (the same patient can be in training and test) and by patient (test patients were never seen).
   - *How to read it:* The official figure is the grey bar (split by beat, the standard MIT-BIH protocol). The coloured bar is an additional analysis: it shows how much accuracy drops when the patient is completely new.
22. `fig_22_patient_cv.png` — **Patient-wise 5-fold CV: accuracy per fold and recall per AAMI class**
   - *What it shows:* Patients are split into 5 groups; each time the system trains on 4 and is tested on the remaining one, never sharing patients.
   - *How to read it:* Each dot is one fold. If they all sit close together, the result does not depend on which patients ended up in the test.
23. `fig_23_automatic_detection.png` — **Annotated vs. automatically detected beats on unseen patients**
   - *What it shows:* Compares classifying the beats marked by the cardiologist with the ones the automatic detector (XQRS) finds, as the app does.
   - *How to read it:* If both bars are almost equal, the app works on unannotated ECGs without losing accuracy.
24. `fig_24_monitoring_session.png` — **Simulated monitoring session on unseen record mitdb_201**
   - *What it shows:* A stretch of an unseen patient's ECG with the system's prediction for every beat.
   - *How to read it:* Blue = predicted normal, red = predicted abnormal, X = error. Under each beat: its true class and the probability.

## Tables

- `beats_per_class.csv` — Beats per database and AAMI class
- `vif_rhythm_features.csv` — VIF of the rhythm features
- `descriptive_statistics.csv` — Descriptive statistics of rhythm features
- `split_assignment.csv` — Row-level train/test assignment
- `tuning_LR.csv` — Randomized search results for Logistic Regression
- `tuning_KNN.csv` — Randomized search results for K-Nearest Neighbors
- `tuning_RF.csv` — Randomized search results for Random Forest
- `tuning_SVM.csv` — Randomized search results for Support Vector Machine
- `oof_metrics.csv` — Cross-validated (out-of-fold) metrics
- `oof_predictions.csv` — Out-of-fold probabilities used to train the meta-model
- `test_metrics.csv` — Held-out test metrics
- `permutation_importance.csv` — Permutation importance of the stacking ensemble
- `simulation_test_beats.csv` — 12 test beats through the system
- `patient_cv_folds.csv` — Patient-wise CV: metrics per fold and model (new patients)
- `patient_cv_per_class.csv` — Patient-wise CV: recall per AAMI class
- `automatic_detection_per_record.csv` — XQRS beat detection quality per unseen record (150 ms tolerance)
- `automatic_detection_metrics.csv` — Classification of unseen patients: annotated vs. automatically detected beats
- `monitoring_session_beats.csv` — Beat-by-beat predictions for record mitdb_201

## Exported models

| File | Description | SHA-256 |
|---|---|---|
| `stacking_ensemble.joblib` | StackingEnsemble: 4 tuned base models + logistic meta-model | `bd425d60a84680fa…` |

## Notes

- Leakage audit: 100% of the beat-level test beats come from patients also seen in training, and for 94% of them the nearest training beat is from the same patient. The beat-level accuracy therefore measures performance on KNOWN patients; patient_cv() measures it on NEW patients.
