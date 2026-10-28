"""Tier 1 logic (no UI): validate inputs, predict, explain and simulate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model_store import Tier

from cardio_ml import config  # noqa: E402  (path set by model_store)
from cardio_ml.preprocessing import ClinicalFeatureEngineer  # noqa: E402

RAW_INPUTS = ["age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
              "cholesterol", "gluc", "smoke", "alco", "active"]

# Factor -> raw inputs replaced by a typical value to measure its effect.
FACTORS = {
    "age": ["age_years"], "sex": ["gender"], "body": ["height", "weight"],
    "blood_pressure": ["ap_hi", "ap_lo"], "cholesterol": ["cholesterol"], "glucose": ["gluc"],
    "smoking": ["smoke"], "alcohol": ["alco"], "activity": ["active"],
}
CATEGORICAL = {"gender", "cholesterol", "gluc", "smoke", "alco", "active"}
# allowed codes of the categorical inputs (as in the Kaggle dataset) and the accepted age range
LEVELS = {"gender": (1, 2), "cholesterol": (1, 2, 3), "gluc": (1, 2, 3), "smoke": (0, 1), "alco": (0, 1),
          "active": (0, 1)}
AGE_RANGE = (18, 100)


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    return ClinicalFeatureEngineer().transform(raw[RAW_INPUTS].astype(float))


def validate(inputs: dict) -> tuple[list[str], list[str]]:
    """Return (errors that block the prediction, warnings about extrapolation).

    Values may come from a CSV, so they are first converted to numbers: an empty cell or a text such as
    "alto" is an error, and so is a code outside the dataset's categories (e.g. cholesterol = 5).
    """
    errors, warnings = [], []
    values = {key: pd.to_numeric(inputs.get(key), errors="coerce") for key in RAW_INPUTS}
    errors += [f"missing:{key}" for key, v in values.items() if pd.isna(v)]
    if errors:
        return errors, warnings
    inputs = {key: float(v) for key, v in values.items()}
    errors += [f"category:{key}:{'/'.join(map(str, levels))}" for key, levels in LEVELS.items()
               if inputs[key] not in levels]
    if not AGE_RANGE[0] <= inputs["age_years"] <= AGE_RANGE[1]:
        errors.append(f"range:age_years:{AGE_RANGE[0]}:{AGE_RANGE[1]}")
    ranges = config.CLINICAL_RANGES
    for key in ("ap_hi", "ap_lo", "height", "weight"):
        lo, hi = ranges[key]
        if not lo <= inputs[key] <= hi:
            errors.append(f"range:{key}:{lo}:{hi}")
    if inputs["ap_hi"] <= inputs["ap_lo"]:
        errors.append("bp_order")
    bmi = inputs["weight"] / (inputs["height"] / 100) ** 2
    if not ranges["bmi"][0] <= bmi <= ranges["bmi"][1]:
        errors.append(f"bmi:{bmi:.1f}")
    # 97 % of the training patients are 40-65 years old (only 4 are younger than 35)
    if not 40 <= inputs["age_years"] <= 65:
        warnings.append("age_extrapolation")
    return errors, warnings


def risk_level(probability: float, referred: bool) -> str:
    """'low' / 'mid' (inconclusive, referred to a clinician) / 'high'."""
    if referred:
        return "mid"
    return "high" if probability >= 0.5 else "low"


def bmi_category(bmi: float) -> int:
    """WHO BMI class: 0 underweight, 1 normal, 2 overweight, 3 obesity."""
    return 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3


def predict_frame(tier: Tier, raw: pd.DataFrame) -> pd.DataFrame:
    """Vectorised prediction for many patients (raw inputs -> full result table)."""
    df = engineer(raw)
    Z = tier.ensemble.base_probabilities(df[tier.features])
    p = tier.ensemble.combine(Z)
    out = df.copy()
    for name in Z.columns:
        out[f"p_{name}"] = Z[name].values
    out["probability"] = p
    out["prediction"] = (p >= 0.5).astype(int)
    out["referred"] = tier.ensemble.referral_mask(p, Z)
    out["risk_segment"] = tier.segmenter.assign(df).values if tier.segmenter is not None else ""
    return out


def assess(tier: Tier, inputs: dict) -> dict:
    df = engineer(pd.DataFrame([inputs]))
    trace = tier.ensemble.explain_case(df[tier.features])
    trace["risk_segment"] = tier.segmenter.assign(df).iloc[0] if tier.segmenter is not None else None
    trace["derived"] = {k: float(df[k].iloc[0]) for k in ("bmi", "bp_stage", "map", "pulse_pressure")}
    trace["weights"] = tier.ensemble.weights()
    return trace


def typical_value(tier: Tier, key: str) -> float:
    stats = tier.schema["training_stats"][key]
    return stats["mode"] if key in CATEGORICAL else stats["median"]


def explain(tier: Tier, inputs: dict) -> pd.DataFrame:
    """Local effect of each factor: P(patient) - P(patient with that factor set to typical)."""
    rows = [dict(inputs)]
    for keys in FACTORS.values():
        modified = dict(inputs)
        for k in keys:
            modified[k] = typical_value(tier, k)
        if modified["ap_hi"] <= modified["ap_lo"]:
            modified["ap_lo"] = modified["ap_hi"] - 10
        rows.append(modified)
    p = predict_frame(tier, pd.DataFrame(rows))["probability"].values
    return pd.DataFrame({"factor": list(FACTORS), "effect": p[0] - p[1:],
                         "probability_if_typical": p[1:]}).sort_values("effect", ascending=False)


def sweep(tier: Tier, inputs: dict, key: str, values: np.ndarray) -> pd.DataFrame:
    """Probability as one input moves across `values`, everything else fixed."""
    rows = []
    for v in values:
        r = dict(inputs)
        r[key] = float(v)
        if r["ap_hi"] > r["ap_lo"]:
            rows.append(r)
    frame = pd.DataFrame(rows)
    frame["probability"] = predict_frame(tier, frame)["probability"].values
    return frame[[key, "probability"]]
