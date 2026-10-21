# Traceability report — clinical tier

Run: 2026-10-27T09:30:16 → 2026-10-27T09:34:34 · seed 42 · Python 3.14.7 · scikit-learn 1.9.1

## Inputs

| File | Description | SHA-256 |
|---|---|---|
| `data/raw/cardio_train.csv` | Kaggle Cardiovascular Disease dataset (S. Ulianova, 2019) | `21a705d23381b0df…` |

## Data lineage

| # | Step | Rows before | Rows after | Removed | Detail |
|---|---|---|---|---|---|
| 1 | Load raw CSV | 70,000 | 70,000 | 0 | 13 columns, separator ';' |
| 2 | Drop identifier columns | 70,000 | 70,000 | 0 | dropped ['id'] (no predictive meaning) |
| 3 | Remove duplicated records | 70,000 | 69,976 | 24 | identical rows (all features and label) kept once |
| 4 | Remove contradictory records | 69,976 | 69,942 | 34 | same feature values with different diagnosis |
| 5 | Filter impossible blood pressure | 69,942 | 68,688 | 1,254 | ap_hi outside [80, 250]: 247; ap_lo outside [40, 160]: 1007 |
| 6 | Check systolic > diastolic | 68,688 | 68,606 | 82 | systolic pressure must exceed diastolic pressure |
| 7 | Filter impossible height/weight | 68,606 | 68,501 | 105 | height outside [130, 210]: 89; weight outside [35, 200]: 16 |
| 8 | Convert age from days to years | 68,501 | 68,501 | 0 | age_years = age_days / 365.25 |
| 9 | Engineer clinical features | 68,501 | 68,501 | 0 | added bmi, map, pulse_pressure, bp_stage (AHA 2017) |
| 10 | Filter implausible BMI | 68,501 | 68,480 | 21 | bmi outside [13, 60]: 21 |
| 11 | Balance classes (random undersampling) | 68,480 | 67,772 | 708 | class counts {0: 34594, 1: 33886} -> {0: 33886, 1: 33886} |
| 12 | Stratified train/test split | 67,772 | 54,217 | 13,555 | train 54,217 / test 13,555 (20% held out, seed 42) |

## Parameters

```json
{
  "mode": "full",
  "model features": [
    "age_years",
    "gender",
    "height",
    "bmi",
    "ap_hi",
    "ap_lo",
    "bp_stage",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
  ],
  "Logistic Regression best hyperparameters": {
    "C": 0.0019517224641449498
  },
  "K-Nearest Neighbors best hyperparameters": {
    "weights": "uniform",
    "p": 2,
    "n_neighbors": 2500
  },
  "Random Forest best hyperparameters": {
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "max_depth": null
  },
  "Support Vector Machine best hyperparameters": {
    "C": 0.13066739238053282,
    "gamma": 0.029621516588303493
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
    "grouped_cv": false
  },
  "meta-model weights": {
    "intercept": -2.7160381860881513,
    "LR": 1.2528686567635454,
    "KNN": 0.7796103349404183,
    "RF": 1.970432764623543,
    "SVM": 1.549723611382434
  },
  "clinical monotonic constraints": {
    "age_years": 1,
    "bmi": 1,
    "ap_hi": 1,
    "ap_lo": 1,
    "bp_stage": 1,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 1,
    "alco": 1,
    "active": -1
  },
  "inputs not given to KNN and SVM": [
    "gluc",
    "smoke",
    "alco"
  ],
  "risk segmentation": {
    "features": [
      "age_years",
      "bmi",
      "ap_hi",
      "ap_lo",
      "pulse_pressure",
      "cholesterol",
      "gluc"
    ],
    "k": 3,
    "pca_components": 5,
    "cluster_to_level": {
      "0": "Low",
      "2": "Medium",
      "1": "High"
    }
  }
}
```

## Metrics (held-out test set)

|                               |   cv_accuracy |   tuning_rows |   oof_seconds |   final_fit_seconds |   rows_used_final_fit |   accuracy |   balanced_accuracy |   precision |   recall_sensitivity |   specificity |         f1 |    roc_auc |     pr_auc |        mcc |      brier |   tp |   fp |   tn |   fn |   stacking_accuracy |   accuracy_95ci_low |   accuracy_95ci_high | best_base_model     |   mcnemar_vs_best_base_p |   referral_rate |   accuracy_on_non_referred |   meets_95_target |   max_share_violating |   mean_share_violating |   constraints |
|:------------------------------|--------------:|--------------:|--------------:|--------------------:|----------------------:|-----------:|--------------------:|------------:|---------------------:|--------------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----:|-----:|-----:|-----:|--------------------:|--------------------:|---------------------:|:--------------------|-------------------------:|----------------:|---------------------------:|------------------:|----------------------:|-----------------------:|--------------:|
| tuning Logistic Regression    |      0.726833 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| tuning K-Nearest Neighbors    |      0.7105   |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| tuning Random Forest          |      0.718083 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| tuning Support Vector Machine |      0.728417 |         12000 |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| training time LR              |    nan        |           nan |       0.11731 |           0.0217433 |                 54217 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| training time KNN             |    nan        |           nan |      11.1128  |           0.0138552 |                 54217 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| training time RF              |    nan        |           nan |       5.38693 |           1.03137   |                 54217 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| training time SVM             |    nan        |           nan |      68.4094  |           7.2255    |                 20000 | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| Logistic Regression           |    nan        |           nan |     nan       |         nan         |                   nan |   0.723202 |            0.723198 |    0.748317 |             0.672569 |      0.773827 |   0.708424 |   0.78746  |   0.765955 |   0.448704 |   0.188919 | 4558 | 1533 | 5245 | 2219 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| K-Nearest Neighbors           |    nan        |           nan |     nan       |         nan         |                   nan |   0.720915 |            0.720907 |    0.776915 |             0.619743 |      0.822071 |   0.689485 |   0.788298 |   0.768127 |   0.451148 |   0.189891 | 4200 | 1206 | 5572 | 2577 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| Random Forest                 |    nan        |           nan |     nan       |         nan         |                   nan |   0.717669 |            0.717663 |    0.761339 |             0.634056 |      0.801269 |   0.691893 |   0.787959 |   0.765777 |   0.441544 |   0.18977  | 4297 | 1347 | 5431 | 2480 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| Support Vector Machine        |    nan        |           nan |     nan       |         nan         |                   nan |   0.721874 |            0.721869 |    0.758644 |             0.65073  |      0.793007 |   0.700556 |   0.788146 |   0.774111 |   0.4483   |   0.187437 | 4410 | 1403 | 5375 | 2367 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| Hard voting                   |    nan        |           nan |     nan       |         nan         |                   nan |   0.72394  |            0.723934 |    0.763775 |             0.648369 |      0.799498 |   0.701357 | nan        | nan        |   0.453074 | nan        | 4394 | 1359 | 5419 | 2383 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| Soft voting                   |    nan        |           nan |     nan       |         nan         |                   nan |   0.724677 |            0.724672 |    0.764277 |             0.649698 |      0.799646 |   0.702345 |   0.793058 |   0.770138 |   0.454484 |   0.186462 | 4403 | 1358 | 5420 | 2374 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| Stacking ensemble             |    nan        |           nan |     nan       |         nan         |                   nan |   0.72512  |            0.725115 |    0.757642 |             0.661945 |      0.788286 |   0.706568 |   0.793072 |   0.769661 |   0.453869 |   0.185199 | 4486 | 1435 | 5343 | 2291 |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |              nan      |            nan         |           nan |
| system summary                |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |             0.72512 |            0.718036 |             0.732647 | Logistic Regression |                 0.329025 |        0.248986 |                   0.783006 |                 0 |              nan      |            nan         |           nan |
| plausibility (final answer)   |    nan        |           nan |     nan       |         nan         |                   nan | nan        |          nan        |  nan        |           nan        |    nan        | nan        | nan        | nan        | nan        | nan        |  nan |  nan |  nan |  nan |           nan       |          nan        |           nan        | nan                 |               nan        |      nan        |                 nan        |               nan |                0.0125 |              0.0015625 |             1 |

## Figures

Each figure with its reading guide (what it shows · how to read it).

1. `fig_01_raw_outliers.png` — **Raw measurements: impossible blood pressures, heights and weights**
   - *What it shows:* Boxplots of blood pressure, height and weight exactly as they come in the source file, before cleaning.
   - *How to read it:* Points far from the box are impossible values (e.g. pressures of thousands of mmHg or negative). They justify the physiological range filters.
2. `fig_02_cleaning_funnel.png` — **Rows remaining after each cleaning step**
   - *What it shows:* Each bar is the dataset after one cleaning step; in brackets, how many rows that step removed.
   - *How to read it:* Read it top to bottom. No row disappears without a reason: each step is explained in TRACEABILITY.md.
3. `fig_03_class_balance.png` — **Class balance before and after undersampling**
   - *What it shows:* How many cases of each class there are. All cases of the rarer class are kept and the same number is drawn at random from the common one (undersampling).
   - *How to read it:* On the right both bars must be equal (50/50). Accuracy cannot then be inflated by always answering the common class.
4. `fig_04_distributions.png` — **Histograms and KDE of the numeric variables by diagnosis**
   - *What it shows:* Histogram with its smoothed curve (KDE) for each variable, one colour per class. Above, the skewness.
   - *How to read it:* If the two curves are apart, the variable helps to tell the classes apart; if they overlap, it adds little.
5. `fig_05_iqr_boxplots.png` — **Boxplots with the IQR outlier rule by diagnosis**
   - *What it shows:* Boxes per class. Anything outside Q1 − 1.5·IQR or Q3 + 1.5·IQR is flagged as an outlier.
   - *How to read it:* The line is the median. A higher box in the sick group points to a risk factor. Outliers are reviewed, not deleted automatically.
6. `fig_06_split_violins.png` — **Split violin plots: categorical factors vs. numeric variables**
   - *What it shows:* Each violin is the shape of the distribution; the left half is one class and the right half the other.
   - *How to read it:* When the right half (CVD) sits higher, the variable is higher in sick people.
7. `fig_07_categorical_prevalence.png` — **CVD prevalence by categorical factor**
   - *What it shows:* Prevalence = sick people in the group ÷ people in the group, for every level of every factor.
   - *How to read it:* Bars above the dashed line (overall average) mark risk factors. In this dataset smoking and glucose go the opposite way to medicine: a known bias of the data, which is why the model has clinical constraints.
8. `fig_08_scatter.png` — **Scatter plots of blood pressure and age vs. BMI**
   - *What it shows:* Each dot is a patient, coloured by diagnosis.
   - *How to read it:* Areas where one colour dominates show value combinations linked to the disease.
9. `fig_09_pairplot.png` — **Pairplot of the main numeric variables**
   - *What it shows:* Grid of scatter plots between pairs of variables; on the diagonal, each variable's distribution.
   - *How to read it:* Shows at a glance how variables relate and whether any of them separates the classes.
10. `fig_10_correlations.png` — **Pearson vs. Spearman correlation matrices (triangular)**
   - *What it shows:* Pearson measures linear association; Spearman, ordered association (uses ranks, robust to outliers). From −1 to 1.
   - *How to read it:* Red: they rise together. Blue: one rises as the other falls. White: no relation. Look at the target's row.
11. `fig_11_vif_candidates.png` — **Multicollinearity of the candidate features (VIF)**
   - *What it shows:* VIF = 1 / (1 − R²): how well one variable can be predicted from the others.
   - *How to read it:* Green < 5 fine, yellow 5–10 moderate, red > 10 severe. Variables with infinite VIF are copies of others and are removed from the model.
12. `fig_12_vif_model_features.png` — **VIF after dropping ['map', 'pulse_pressure', 'weight'] (exact linear combinations)**
   - *What it shows:* The same VIF computation with the variables the model really uses.
   - *How to read it:* All bars should now be green or yellow: no duplicated information is left.
13. `fig_13_pca_projection.png` — **2-D PCA projection of the clinical features by diagnosis**
   - *What it shows:* PCA summarises all variables in two axes that keep as much variation as possible.
   - *How to read it:* If the colours form separate clouds, the classes are easy to separate; if they mix, the problem is hard.
14. `fig_14_metric_comparison.png` — **Test-set metrics of the four base models, voting schemes and the stacking ensemble**
   - *What it shows:* Accuracy, precision, sensitivity, specificity, F1 and AUC on data the models never saw.
   - *How to read it:* Higher is better. Compare the combined system (stacking) with the 4 models; the red line is the 95 % target.
15. `fig_15_confusion_matrices.png` — **Confusion matrices on the held-out test set**
   - *What it shows:* 2×2 table per model: rows = truth, columns = prediction. Each row adds up to 100 %.
   - *How to read it:* The diagonal holds the hits. The bottom-left cell is the sick cases the model missed: the most dangerous error.
16. `fig_16_roc_pr_curves.png` — **ROC and precision-recall curves**
   - *What it shows:* How hits and false alarms change as the decision threshold moves. The area under the curve (AUC) sums it up.
   - *How to read it:* A curve hugging the top-left (ROC) or top-right (PR) corner is better. The dotted diagonal is chance (AUC 0.5).
17. `fig_17_calibration.png` — **Reliability of the predicted probabilities**
   - *What it shows:* Cases are grouped by predicted probability and compared with the real share of positives.
   - *How to read it:* On the diagonal, '70 %' really means 7 out of 10 such cases are positive. Below it, the model over-estimates.
18. `fig_18_meta_weights.png` — **Coefficients learned by the meta-model for each base model**
   - *What it shows:* Weights β learned by the meta-model: P = σ(β₀ + Σ βₖ·pₖ). They are ≥ 0 by design.
   - *How to read it:* Taller bar = the final answer listens more to that model. A weight of 0 means the others already carry its information.
19. `fig_19_model_diversity.png` — **Diversity between base models (OOF correlation and disagreement)**
   - *What it shows:* Left: correlation between the probabilities of each pair of models. Right: % of cases where they disagree.
   - *How to read it:* Combining models only helps if they sometimes disagree: correlations below 1 and non-zero disagreement mean complementary models.
20. `fig_20_probability_distribution.png` — **Distribution of the final probability per true class and referral band**
   - *What it shows:* Histogram of the system's probability, one colour per true class. The grey band is the doubt zone.
   - *How to read it:* Ideal: one colour near 0 and the other near 1. Errors live where colours overlap; cases in the grey band go to a clinician.
21. `fig_21_coverage_curve.png` — **Accuracy when the system answers only its most confident cases**
   - *What it shows:* Cases are sorted from most to least confident; for each share answered (coverage) the accuracy is computed.
   - *How to read it:* Towards the left the system answers fewer cases and gets more right. Where the curve crosses the 95 % line tells which share of cases it can solve on its own at that accuracy.
22. `fig_22_permutation_importance.png` — **Permutation importance of each feature group for the ensemble**
   - *What it shows:* Permutation importance: how much accuracy drops when one input is randomly shuffled.
   - *How to read it:* Longer bar = the system depends more on that input. Bars near 0 = it could be removed without losing accuracy.
23. `fig_23_logistic_coefficients.png` — **Standardised logistic-regression coefficients**
   - *What it shows:* Logistic-regression coefficients per standard deviation of each input (constrained to the clinical sign).
   - *How to read it:* Orange = raises the risk, blue = lowers it. The longer the bar, the stronger the effect.
24. `fig_24_plausibility_audit.png` — **Clinical plausibility audit: wrong-way changes per model and factor**
   - *What it shows:* For real patients one factor (pressure, weight, cholesterol…) is raised with everything else fixed, counting for how many the risk falls by more than 1 point.
   - *How to read it:* 0 % (white) = always consistent with medicine. Red = the model contradicts clinical knowledge. The 'Stacking' column is the final answer.
25. `fig_25_kmeans_k_scan.png` — **Choice of k for K-means (k = 3 chosen)**
   - *What it shows:* Elbow method (inertia J) and silhouette coefficient for k = 2…8 groups.
   - *How to read it:* Look for the 'elbow' where J stops falling fast and a high silhouette. k = 3 was chosen (low, medium, high risk).
26. `fig_26_risk_segments_pca.png` — **Risk segments in PCA space (test set)**
   - *What it shows:* Test patients coloured by the group K-means gave them (without using the diagnosis).
   - *How to read it:* Each colour is a similar patient profile; the groups occupy different areas of the map.
27. `fig_27_risk_segments_profile.png` — **Mean profile and observed CVD prevalence per segment (test set)**
   - *What it shows:* Left: mean of each variable per group. Right: real share with CVD in each group.
   - *How to read it:* The high-risk group should be older, with higher pressure and more disease. It confirms the groups make clinical sense.
28. `fig_28_segments_vs_ensemble.png` — **Ensemble probability distribution inside each K-means segment**
   - *What it shows:* Probability given by the supervised system inside each unsupervised group.
   - *How to read it:* Boxes should rise from low to high: two independent methods reach the same conclusion.
29. `fig_29_simulation_archetypes.png` — **Per-model and final probability for simulated patient archetypes**
   - *What it shows:* Each row is an invented patient; each column the probability from each model and, last, the single answer.
   - *How to read it:* Blue = low risk, red = high. Follow case by case how the 4 models combine into 1 answer. '[refer]' = sent to a clinician.
30. `fig_30_simulation_test_patients.png` — **Per-model and final probability for 12 unseen test patients**
   - *What it shows:* Same as above with real test-set patients and their true diagnosis.
   - *How to read it:* '(ok)' = the system was right, '(WRONG)' = it was wrong. It is the full trace of each prediction.

## Tables

- `vif_candidates.csv` — VIF of all candidate features
- `vif_model_features.csv` — VIF after dropping collinear features
- `descriptive_statistics.csv` — Descriptive statistics of the clean dataset
- `split_assignment.csv` — Row-level train/test assignment
- `tuning_LR.csv` — Randomized search results for Logistic Regression
- `tuning_KNN.csv` — Randomized search results for K-Nearest Neighbors
- `tuning_RF.csv` — Randomized search results for Random Forest
- `tuning_SVM.csv` — Randomized search results for Support Vector Machine
- `oof_metrics.csv` — Cross-validated (out-of-fold) metrics
- `oof_predictions.csv` — Out-of-fold probabilities used to train the meta-model
- `test_metrics.csv` — Held-out test metrics
- `permutation_importance.csv` — Permutation importance of the stacking ensemble
- `logistic_coefficients.csv` — Standardised logistic-regression coefficients and odds ratios
- `plausibility_audit.csv` — Share of patients whose risk moves against the clinical direction (> 1 pp)
- `kmeans_k_scan.csv` — Elbow and silhouette for k = 2..8
- `risk_segments_profile.csv` — Clinical profile and CVD prevalence per risk segment (test set)
- `simulation_archetypes.csv` — Simulated patient archetypes through the system
- `simulation_test_patients.csv` — 12 unseen test patients through the system

## Exported models

| File | Description | SHA-256 |
|---|---|---|
| `stacking_ensemble.joblib` | StackingEnsemble: 4 tuned base models + logistic meta-model | `b2173379b12b4ac1…` |
| `risk_segmenter.joblib` | RiskSegmenter: PCA + K-means (k = 3) -> Low / Medium / High risk | `ae5e11b396dc4857…` |

## Notes

- Dropped ['map', 'pulse_pressure', 'weight'] from the model inputs: map and pulse_pressure are linear combinations of ap_hi/ap_lo (VIF -> inf) and weight is explained by bmi and height (VIF ~ 100).
