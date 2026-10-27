# ML pipeline — Cardiovascular risk prediction and diagnosis

Code for the modelling part of the project (Hito 2). It cleans the data, explores it, trains
**four independent models** (Logistic Regression, KNN, Random Forest, SVM) and combines them with
**stacking**, so the system gives **one final answer**. It also records the traceability of every step.

## The two tiers of the system

| Tier | Clinical question | Data | Records |
|---|---|---|---|
| 1. Risk screening | Does this patient have cardiovascular disease, based on a basic exam? | Kaggle *Cardiovascular Disease dataset* (Ulianova, 2019) | 70,000 patients |
| 2. Diagnosis | Is this heartbeat normal or abnormal (arrhythmia)? | PhysioNet *MIT-BIH Arrhythmia DB* + *MIT-BIH Supraventricular Arrhythmia DB* | 284,887 beats of 121 patients |

In telemedicine the tiers chain together: tier 1 flags patients at risk from basic exam data, and
tier 2 analyses the ECG of a flagged patient recorded with a wearable or Holter.

## How the four models reach one answer

```
              ┌── Logistic Regression ──► p1 ─┐
features x ───┼── K-Nearest Neighbors ──► p2 ─┼──► meta-model (logistic) ──► P(final) ──► answer / refer to clinician
              ├── Random Forest ────────► p3 ─┤        σ(β0 + β1p1 + β2p2 + β3p3 + β4p4)
              └── SVM (RBF) ────────────► p4 ─┘
```

Each base model is trained separately. The meta-model learns from **out-of-fold** predictions
(5-fold CV), so it learns how much to trust each model on data that model never saw. Cases with
an uncertain final probability (0.35–0.65), or where the models split 2–2, are referred to a
clinician. All the equations are in [`docs/mathematical_model.md`](docs/mathematical_model.md).

## Results

| | Tier 1 — clinical risk | Tier 2 — ECG beats |
|---|---|---|
| Balanced modelling dataset | 67,772 patients | 65,414 beats (121 patients) |
| Stacking accuracy, held-out test | **72.5 %** (one row per patient) | **96.9 %** (stratified split by beat) |
| Rubric target (> 95 %) | not reachable with 11 exam inputs (screening filter) | ✅ met |

**Tier 2 evaluation protocol.** The official figure uses the standard MIT-BIH protocol (*intra-patient*): the beats are
split 80 / 20 at random, stratified by class, and the 20 % test beats are never used for training or tuning. The
stacking ensemble reaches **96.9 %** (95 % CI in `outputs/RESULTS_SUMMARY.md`), above the > 95 % target.

**Additional analysis: leakage audit and new patients.** A 30-minute ECG gives ~2,000 beats of the same person, so
with a split by beat 100 % of the test beats come from patients who also gave other beats to training, and for 94 % of
them the most similar training beat belongs to the same patient (`leakage_check()`). To measure how the system
generalises beyond that protocol, `patient_cv()` also runs a patient-wise 5-fold cross-validation (*inter-patient*,
records 201 and 202 of the same subject kept together): on completely new patients the stacking reaches
**90.5 % ± 4.4 %**, lower as expected from the inter-patient literature (de Chazal et al., 2004; Luz et al., 2016).
It is reported as a limitation. Supraventricular (79 % recall) and fusion beats (41 %) are the hard cases there;
ventricular beats reach 98 %. Hyperparameter tuning and the stacking out-of-fold predictions are grouped by patient,
so the models are chosen to generalise, not to memorise. On new patients the stacking has the best ROC-AUC
(0.970) and the best balanced accuracy at the real share of beats (87.9 %).

Tier 1 has no leakage of this kind: every row is a different patient, exact duplicates and contradictory rows are removed
before the split, scalers live inside the model pipelines, and tuning only sees the training set. Its 72.5 % is the
ceiling of those 11 inputs (every model and combination lands at 72–73 %).

## Clinical plausibility (tier 1)

The Kaggle data contains associations that contradict clinical evidence. For example, with very high cholesterol,
the patients with very high glucose have *less* CVD (69 %) than those with normal glucose (82 %), and smokers have
less CVD than non-smokers. Unconstrained models reproduced this, and the step-shaped random forest and KNN made
the risk go up and down when weight or age rose. Tier 1 therefore applies clinical rules
(`config.CLINICAL_MONOTONIC`, `docs/mathematical_model.md` §5b):

* sign-constrained logistic regression and a monotone random forest;
* KNN and SVM do not receive glucose, smoking or alcohol, and use smooth settings (KNN with k = 2,500);
* non-negative meta-model weights.

An automatic audit (`tables/plausibility_audit.csv`) sweeps each factor for 400 test patients and counts how often
the final probability moves more than 1 point the wrong way:

| Factor swept | Before | After |
|---|---|---|
| Age 40 → 65 | 80 % of patients | 0 % |
| Weight 50 → 130 kg | 78 % | 0 % |
| Systolic pressure 100 → 200 | 21 % | 0 % |
| Cholesterol / glucose levels | 5 % / 70 % | 0 % / 0 % |
| Smoking, alcohol, physical activity | — | 0 %, 0 %, 1.2 % |
| **Test accuracy** | 73.3 % | 72.5 % |

The cost is 0.8 accuracy points, in exchange for answers a clinician can trust. Because the data cannot show any harm
from smoking, alcohol or glucose, the constrained models give them little or no weight. They never treat them as
protective. Setting `CLINICAL_MONOTONIC = {}` reproduces the unconstrained models.

## Project structure

```
ml-pipeline/
├── main.py                    # runs everything from the command line
├── requirements.txt
├── cardio_ml/                 # package (object-oriented, one responsibility per module)
│   ├── config.py              # paths, seeds, ranges, hyperparameter settings
│   ├── ingestion.py           # Extraction: DataIngester (ABC) -> ClinicalCSVIngester, PhysioNetECGIngester
│   ├── ecg_etl.py             # ECG signal -> beat table (filter, resample, segment, RR features)
│   ├── preprocessing.py       # Transformation: BasePreprocessor (ABC) + steps + Pipeline (polymorphism)
│   ├── eda.py                 # exploratory figures (INFOVIZ)
│   ├── models.py              # BaseModel (ABC) -> LR, KNN, RF, SVM
│   ├── ensemble.py            # StackingEnsemble: how the models combine into one answer
│   ├── evaluation.py          # metrics, bootstrap CI, McNemar test, evaluation figures
│   ├── explainability.py      # permutation importance (XAI)
│   ├── plausibility.py        # clinical plausibility audit (does risk move the right way?)
│   ├── segmentation.py        # PCA + K-means risk segments (Low / Medium / High)
│   ├── simulation.py          # results simulation and per-case traces
│   ├── tracking.py            # RunTracker: lineage, figures, metrics, checksums, manifest
│   ├── figure_guide.py        # plain-language reading guide of every figure (Spanish / English)
│   ├── palette.py / plotting.py  # colour-blind-safe palette and shared figure style
│   ├── report.py              # merges both tiers into outputs/RESULTS_SUMMARY.md
│   └── workflows.py           # the two end-to-end workflows (template method)
├── notebooks/
│   ├── 01_clinical_risk_tier.ipynb
│   └── 02_ecg_diagnosis_tier.ipynb
├── docs/mathematical_model.md
├── data/raw/                  # cardio_train.csv (ECG databases are downloaded automatically)
└── outputs/                   # generated: figures, tables, models, traceability
    ├── clinical/
    └── ecg/
```

## How to run it

**Locally (Python 3.11+):**

```bash
cd ml-pipeline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                     # both tiers (about 15-25 min)
python main.py --tier clinical     # tier 1 only
python main.py --tier ecg          # tier 2 only (downloads ~110 MB from PhysioNet the first time)
python main.py --fast              # smoke test on 8,000 rows (about 1 min)
```

You can also open the notebooks in `notebooks/`. They run the same steps one by one, with
explanations, and show the figures inline.

**Google Colab:** uncomment the first lines of the notebook's setup cell (clone the repository and
`pip install -r requirements.txt`).

## Traceability

Every run writes, for each tier, `outputs/<tier>/`:

| File | Content |
|---|---|
| `TRACEABILITY.md` | readable summary: inputs with SHA-256 checksums, the rows each cleaning step removed, parameters, metrics, the list of figures, exported models |
| `run_manifest.json` | the same, machine-readable, plus the library versions |
| `figures/fig_XX_*.png` | numbered figures, in execution order |
| `tables/*.csv` | every table (tuning results, OOF and test metrics, VIF, segments, simulations...) |
| `tables/*_trace.json` | full trace of each simulated case: inputs → every model → meta-model → answer |
| `models/` | `stacking_ensemble.joblib` + `model_schema.json` (consumed by the web app) |

Every figure carries a reading guide (*what it shows* and *how to read it*, `cardio_ml/figure_guide.py`): it is
printed under the figure in the notebooks, listed in `TRACEABILITY.md` and shown next to the figure in the web app.
Every equation is documented where it is implemented (module and class docstrings) and, with its derivation, in
`docs/mathematical_model.md`.

`outputs/RESULTS_SUMMARY.md` brings both tiers together and includes the rubric checklist
(regenerate it with `python -m cardio_ml.report`).

All randomness uses `seed = 42`, so re-running produces the same numbers.

## Data sources and licences

* S. Ulianova, *Cardiovascular Disease dataset*, Kaggle, 2019.
* G. Moody and R. Mark, "The impact of the MIT-BIH Arrhythmia Database", *IEEE Eng. Med. Biol.*, 2001. PhysioNet, ODC-By 1.0.
* S. D. Greenwald, *MIT-BIH Supraventricular Arrhythmia Database*, PhysioNet, ODC-By 1.0.
* A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet", *Circulation*, 2000.
