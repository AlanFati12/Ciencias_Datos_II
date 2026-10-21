# Results summary — Hito 2

Generated 2026-10-27 14:48 from `outputs/*/run_manifest.json`. Every number below can be traced to its tier's `TRACEABILITY.md`.

## Tier 1 — clinical risk screening

* Modelling dataset: **67,772 balanced records** (train 54,217 / test 13,555).
* Final system (stacking of 4 models): **accuracy 72.51%** (95 % CI 71.80% – 73.26%).
* Best single model: Logistic Regression; McNemar p-value stacking vs. best model = 0.329.
* Referral rule: 24.9% of cases referred to a clinician; accuracy on the rest = 78.30%.

### Held-out test metrics

|                        |   accuracy |   precision |   recall_sensitivity |   specificity |     f1 |   roc_auc |    mcc |   balanced_accuracy |
|:-----------------------|-----------:|------------:|---------------------:|--------------:|-------:|----------:|-------:|--------------------:|
| Logistic Regression    |     0.7232 |      0.7483 |               0.6726 |        0.7738 | 0.7084 |    0.7875 | 0.4487 |              0.7232 |
| K-Nearest Neighbors    |     0.7209 |      0.7769 |               0.6197 |        0.8221 | 0.6895 |    0.7883 | 0.4511 |              0.7209 |
| Random Forest          |     0.7177 |      0.7613 |               0.6341 |        0.8013 | 0.6919 |    0.788  | 0.4415 |              0.7177 |
| Support Vector Machine |     0.7219 |      0.7586 |               0.6507 |        0.793  | 0.7006 |    0.7881 | 0.4483 |              0.7219 |
| Hard voting            |     0.7239 |      0.7638 |               0.6484 |        0.7995 | 0.7014 |  nan      | 0.4531 |              0.7239 |
| Soft voting            |     0.7247 |      0.7643 |               0.6497 |        0.7996 | 0.7023 |    0.7931 | 0.4545 |              0.7247 |
| Stacking ensemble      |     0.7251 |      0.7576 |               0.6619 |        0.7883 | 0.7066 |    0.7931 | 0.4539 |              0.7251 |

### Meta-model weights

```json
{
  "intercept": -2.7160381860881513,
  "LR": 1.2528686567635454,
  "KNN": 0.7796103349404183,
  "RF": 1.970432764623543,
  "SVM": 1.549723611382434
}
```

## Tier 2 — ECG beat diagnosis

* Modelling dataset: **65,414 balanced records** (train 52,331 / test 13,083).
* Final system (stacking of 4 models): **accuracy 96.87%** (95 % CI 96.56% – 97.18%).
* Best single model: Random Forest; McNemar p-value stacking vs. best model = 0.000.
* Referral rule: 4.0% of cases referred to a clinician; accuracy on the rest = 97.80%.

### Held-out test metrics

|                        |   accuracy |   precision |   recall_sensitivity |   specificity |     f1 |   roc_auc |    mcc |   balanced_accuracy |
|:-----------------------|-----------:|------------:|---------------------:|--------------:|-------:|----------:|-------:|--------------------:|
| Logistic Regression    |     0.9329 |      0.9424 |               0.9222 |        0.9436 | 0.9322 |    0.9789 | 0.866  |              0.9329 |
| K-Nearest Neighbors    |     0.9584 |      0.9701 |               0.946  |        0.9708 | 0.9579 |    0.9917 | 0.9171 |              0.9584 |
| Random Forest          |     0.974  |      0.9691 |               0.9792 |        0.9688 | 0.9741 |    0.9959 | 0.9481 |              0.974  |
| Support Vector Machine |     0.9431 |      0.9552 |               0.9298 |        0.9564 | 0.9424 |    0.9832 | 0.8866 |              0.9431 |
| Hard voting            |     0.965  |      0.9664 |               0.9635 |        0.9665 | 0.9649 |  nan      | 0.93   |              0.965  |
| Soft voting            |     0.9649 |      0.9666 |               0.9632 |        0.9667 | 0.9649 |    0.9938 | 0.9298 |              0.9649 |
| Stacking ensemble      |     0.9687 |      0.9577 |               0.9806 |        0.9567 | 0.969  |    0.995  | 0.9376 |              0.9687 |

### Meta-model weights

```json
{
  "intercept": -3.398022981408816,
  "LR": 1.6190051217806056,
  "KNN": 0.0,
  "RF": 6.265795050512607,
  "SVM": 0.0
}
```

### Additional analysis: leakage audit and generalisation to new patients

* The official accuracy above follows the standard MIT-BIH protocol (stratified split by beat, *intra-patient*): the test beats were never used for training, but 100% of them come from patients that also gave other beats to training, and for 94% of them the most similar training beat belongs to the same patient.
* As an additional robustness analysis, patient-wise 5-fold cross-validation (no patient in train and test at the same time, *inter-patient*) measures the accuracy on completely new patients. As in the inter-patient literature (de Chazal et al., 2004), it is lower; it is reported as a limitation of the system:

|                        | accuracy (balanced test)   | sensitivity   | specificity   |   roc_auc | accuracy (natural prevalence)   | balanced accuracy (natural prevalence)   |
|:-----------------------|:---------------------------|:--------------|:--------------|----------:|:--------------------------------|:-----------------------------------------|
| Logistic Regression    | 91.40% ± 4.51%             | 88.89%        | 93.90%        |    0.9609 | 94.52%                          | 84.21%                                   |
| K-Nearest Neighbors    | 81.31% ± 5.15%             | 73.81%        | 88.80%        |    0.8878 | 92.30%                          | 77.02%                                   |
| Random Forest          | 89.45% ± 6.36%             | 84.13%        | 94.78%        |    0.9688 | 94.58%                          | 79.14%                                   |
| Support Vector Machine | 89.03% ± 5.24%             | 83.68%        | 94.37%        |    0.9598 | 94.97%                          | 84.54%                                   |
| Stacking ensemble      | 90.46% ± 4.42%             | 87.89%        | 93.03%        |    0.9695 | 95.25%                          | 87.94%                                   |

* Monitoring session on unseen record `mitdb_201`: 1,881 beats, true abnormal burden 13.5%, predicted 24.0%, beat accuracy 83.52%.

## Rubric checklist (Hito 2)

| Criterion | Status |
|---|---|
| Dataset balanced and > 35,000 records — Tier 1 — clinical risk screening | ✅ 67,772 |
| Dataset balanced and > 35,000 records — Tier 2 — ECG beat diagnosis | ✅ 65,414 |
| 4 ML models with theoretical support | ✅ LR, KNN, RF, SVM (`docs/mathematical_model.md` §4) |
| Results simulation with traceability | ✅ `TRACEABILITY.md`, `run_manifest.json`, `*_trace.json` |
| Mathematical model with equations and graphs | ✅ `docs/mathematical_model.md` + numbered figures |
| Accuracy > 95 % — Tier 1 — clinical risk screening | ❌ 72.51% |
| Accuracy > 95 % — Tier 2 — ECG beat diagnosis | ✅ 96.87% (held-out test beats, standard MIT-BIH beat-level protocol) |
| Application with 10 functionalities | ✅ `web-app/` (Streamlit, 10 functionalities + PDF report + bilingual UI) |
