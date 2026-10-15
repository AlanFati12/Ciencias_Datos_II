"""Loads the artefacts exported by the ML pipeline (../ml-pipeline/outputs).

The app never trains anything: it reuses the fitted StackingEnsemble objects,
their schema (features, prior, valid ranges) and the run manifests.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = APP_ROOT.parent / "ml-pipeline"
OUTPUTS = ML_ROOT / "outputs"
SAMPLE_DATA = APP_ROOT / "sample_data"

# The pickled models reference classes of the cardio_ml package.
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


@dataclass
class Tier:
    name: str
    ensemble: object
    schema: dict
    segmenter: object | None = None

    @property
    def features(self) -> list[str]:
        return self.schema["features"]


class ModelsMissing(FileNotFoundError):
    pass


@st.cache_resource(show_spinner="Loading models…")
def _load(name: str) -> Tier:
    folder = OUTPUTS / name / "models"
    model, schema = folder / "stacking_ensemble.joblib", folder / "model_schema.json"
    if not (model.exists() and schema.exists()):
        raise ModelsMissing(name)          # exceptions are not cached, so training later just works
    segmenter = folder / "risk_segmenter.joblib"
    return Tier(name, joblib.load(model), json.loads(schema.read_text()),
                joblib.load(segmenter) if segmenter.exists() else None)


def require_tier(name: str) -> Tier:
    """Return the tier or stop the page with instructions to train it."""
    from .i18n import t
    try:
        return _load(name)
    except ModelsMissing:
        st.error(t("common.models_missing", tier=name))
        st.code(f"cd ml-pipeline\npython main.py --tier {name}", language="bash")
        st.stop()


@st.cache_data(show_spinner=False, ttl=60)
def load_manifest(name: str) -> dict | None:
    path = OUTPUTS / name / "run_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def tier_kpis(name: str) -> dict:
    """Headline numbers of a tier, read from its run manifest."""
    m = load_manifest(name)
    if not m:
        return {}
    s = m["metrics"].get("system summary", {})
    kpis = {"accuracy": s.get("stacking_accuracy"), "ci_low": s.get("accuracy_95ci_low"),
            "ci_high": s.get("accuracy_95ci_high"),
            "auc": m["metrics"].get("Stacking ensemble", {}).get("roc_auc")}
    rows = next((x["rows_after"] for x in m["lineage"] if x["step"].startswith("Balance classes")), None)
    kpis["rows"] = rows
    cv = m["metrics"].get("patient-wise CV Stacking ensemble")
    if cv:                                   # accuracy on NEW patients (patient-wise cross-validation)
        kpis.update(new_accuracy=cv["accuracy_mean"], new_accuracy_sd=cv["accuracy_std"],
                    new_natural_accuracy=cv["natural_accuracy_mean"],
                    new_natural_balanced=cv["natural_balanced_accuracy_mean"], new_auc=cv["roc_auc_mean"])
    leak = m["metrics"].get("leakage audit (beat-level split)")
    if leak:
        kpis.update(leak_overlap=leak["test_beats_from_training_patients"],
                    leak_nn=leak["nearest_training_beat_same_patient"])
    auto = m["metrics"].get("automatic detection (unseen patients)")
    if auto:
        kpis.update(auto_accuracy=auto["accuracy"], auto_balanced=auto["balanced_accuracy"],
                    auto_detection=auto["detection_sensitivity"])
    return kpis
