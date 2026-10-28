"""Shared test setup: import the app's core package and load the exported models once."""
import json
import logging
import sys
from pathlib import Path

import joblib
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
logging.getLogger("streamlit").setLevel(logging.ERROR)    # no "missing ScriptRunContext" noise outside the app

from core.model_store import OUTPUTS, Tier  # noqa: E402


def _tier(name: str) -> Tier:
    folder = OUTPUTS / name / "models"
    if not (folder / "stacking_ensemble.joblib").exists():
        pytest.skip(f"the {name} models are not trained yet (run ml-pipeline/main.py)")
    segmenter = folder / "risk_segmenter.joblib"
    return Tier(name, joblib.load(folder / "stacking_ensemble.joblib"), json.loads((folder / "model_schema.json").read_text()),
                joblib.load(segmenter) if segmenter.exists() else None)


@pytest.fixture(scope="session")
def ecg_tier() -> Tier:
    return _tier("ecg")


@pytest.fixture(scope="session")
def clinical_tier() -> Tier:
    return _tier("clinical")
