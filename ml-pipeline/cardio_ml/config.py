"""Central configuration: paths, seeds and hyperparameter search spaces.

Every number that changes a result lives here, so a run can be reproduced
from the manifest that the RunTracker writes next to the outputs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Probability band in which the system refers the case to a clinician
# instead of answering on its own (see ensemble.StackingEnsemble.referral_mask).
REFERRAL_BAND = (0.35, 0.65)

# ---------------------------------------------------------------- clinical tier
CLINICAL_CSV = DATA_RAW / "cardio_train.csv"
CLINICAL_TARGET = "cardio"

# Physiologically plausible ranges used to discard measurement errors.
# Sources: AHA blood-pressure guideline and adult anthropometric limits.
CLINICAL_RANGES = {
    "ap_hi": (80, 250),     # systolic mmHg
    "ap_lo": (40, 160),     # diastolic mmHg
    "height": (130, 210),   # cm
    "weight": (35, 200),    # kg
    "bmi": (13, 60),        # kg/m^2
}

CLINICAL_CANDIDATE_FEATURES = [
    "age_years", "gender", "height", "weight", "bmi",
    "ap_hi", "ap_lo", "map", "pulse_pressure", "bp_stage",
    "cholesterol", "gluc", "smoke", "alco", "active",
]
# map and pulse_pressure are exact linear combinations of ap_hi and ap_lo
# (VIF -> infinity) and weight is almost fully explained by bmi and height
# (VIF ~ 100), see the multicollinearity figures. They are not fed to the
# models but are still used for EDA and for the risk segmentation.
CLINICAL_COLLINEAR_DROP = ["map", "pulse_pressure", "weight"]
CLINICAL_FEATURES = [f for f in CLINICAL_CANDIDATE_FEATURES if f not in CLINICAL_COLLINEAR_DROP]

# Clinical direction of each risk factor (+1 risk can only go up, -1 only down,
# missing = free). The raw data contradicts some of them (e.g. smokers and people
# with very high glucose show LOWER CVD prevalence, a known bias of this dataset),
# so the logistic regression and the random forest are constrained to respect
# them, and the meta-model weights are kept >= 0. Set to {} to train without them.
CLINICAL_MONOTONIC = {
    "age_years": 1, "bmi": 1, "ap_hi": 1, "ap_lo": 1, "bp_stage": 1,
    "cholesterol": 1, "gluc": 1, "smoke": 1, "alco": 1, "active": -1,
}

# KNN and SVM cannot be constrained, so they do not receive the inputs whose
# association in this dataset contradicts clinical evidence. Only the
# constrained models (LR, RF) use them, and only to raise the risk.
CLINICAL_UNCONSTRAINED_EXCLUDE = ["gluc", "smoke", "alco"]

# What-if sweeps used by the plausibility audit (values of the raw inputs).
PLAUSIBILITY_SWEEPS = {
    "age_years": list(range(40, 66)), "weight": list(range(50, 132, 2)), "ap_hi": list(range(100, 205, 5)),
    "cholesterol": [1, 2, 3], "gluc": [1, 2, 3], "smoke": [0, 1], "alco": [0, 1], "active": [0, 1],
}
PLAUSIBILITY_EXPECTED = {"age_years": 1, "weight": 1, "ap_hi": 1, "cholesterol": 1, "gluc": 1,
                         "smoke": 1, "alco": 1, "active": -1}

# --------------------------------------------------------------------- ECG tier
PHYSIONET_DATABASES = {
    # name: (zip url, folder created by the zip)
    "mitdb": ("https://physionet.org/content/mitdb/get-zip/1.0.0/",
              "mit-bih-arrhythmia-database-1.0.0"),
    "svdb": ("https://physionet.org/content/svdb/get-zip/1.0.0/",
             "mit-bih-supraventricular-arrhythmia-database-1.0.0"),
}
ECG_DIR = DATA_RAW / "ecg"
ECG_TARGET = "abnormal"

# Paced records are excluded, as recommended by the AAMI EC57 standard.
ECG_EXCLUDED_RECORDS = {"mitdb": {"102", "104", "107", "217"}}

# Records 201 and 202 of MIT-BIH come from the same subject: for every split
# "by patient" they must stay together, otherwise the patient would leak.
ECG_SAME_PATIENT = {"mitdb_202": "mitdb_201"}

ECG_FS_TARGET = 128          # Hz, common rate for both databases
ECG_WINDOW = (0.25, 0.45)    # seconds before / after the R peak
ECG_BANDPASS = (0.5, 40.0)   # Hz, removes baseline wander and muscle noise
ECG_LOCAL_RR_BEATS = 10      # beats used for the local average RR

# AAMI EC57 grouping of the MIT-BIH beat symbols.
AAMI_CLASSES = {
    "N": ["N", "L", "R", "e", "j"],   # normal and bundle-branch beats
    "S": ["A", "a", "J", "S"],        # supraventricular ectopic
    "V": ["V", "E"],                  # ventricular ectopic
    "F": ["F"],                       # fusion
    "Q": ["/", "f", "Q"],             # paced / unclassifiable (excluded)
}

# ------------------------------------------------------- model search spaces
# Tuned with a randomized search on a stratified subsample (see models.py).
TUNING_SAMPLE = 12000
TUNING_ITER = 8

# SVM training cost grows between O(n^2) and O(n^3), so the SVM is trained on
# a stratified subsample of at most this many rows. The value is logged.
SVM_MAX_TRAIN = 20000
