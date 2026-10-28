"""Validation of patient data and clinical behaviour of the risk model (core/clinical.py, core/files.py)."""
import pytest

from core import clinical, files
from core.i18n import validation_message

PATIENT = {"age_years": 55, "gender": 2, "height": 170, "weight": 80, "ap_hi": 130, "ap_lo": 85,
           "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}


def check(**changes):
    return clinical.validate({**PATIENT, **changes})


def test_valid_patient_has_no_errors():
    assert check() == ([], [])


@pytest.mark.parametrize("changes, code", [
    ({"ap_hi": 16020}, "range:ap_hi"), ({"height": 1.70}, "range:height"), ({"ap_hi": 80, "ap_lo": 90}, "bp_order"),
    ({"weight": 190, "height": 140}, "bmi"), ({"age_years": 20000}, "range:age_years"),
    ({"cholesterol": 5}, "category:cholesterol"), ({"gender": 0}, "category:gender"), ({"smoke": 2}, "category:smoke"),
    ({"gluc": "alto"}, "missing:gluc"), ({"ap_lo": None}, "missing:ap_lo"), ({"weight": ""}, "missing:weight"),
])
def test_impossible_or_unreadable_data_is_rejected(changes, code):
    errors, _ = check(**changes)
    assert any(e.startswith(code) for e in errors), errors
    assert all(validation_message(e) for e in errors)            # every code has a readable message


def test_numbers_written_as_text_are_accepted():
    assert check(ap_hi="140", weight="80.5")[0] == []


def test_age_outside_the_training_range_is_a_warning_only():
    assert check(age_years=30) == ([], ["age_extrapolation"])


def test_risk_never_goes_down_when_blood_pressure_goes_up(clinical_tier):
    risks = [clinical.assess(clinical_tier, {**PATIENT, "ap_hi": sbp})["final_probability"]
             for sbp in (110, 130, 150, 170)]
    assert risks == sorted(risks)


def test_risk_never_goes_down_when_the_patient_starts_smoking(clinical_tier):
    no = clinical.assess(clinical_tier, PATIENT)["final_probability"]
    yes = clinical.assess(clinical_tier, {**PATIENT, "smoke": 1})["final_probability"]
    assert yes >= no


def test_spanish_excel_csv_is_read():
    data = "ap_hi;ap_lo;weight\n120;80;70,5\n140;90;82,0\n".encode()
    df = files.read_table(data)
    assert list(df.columns) == ["ap_hi", "ap_lo", "weight"] and df["weight"].tolist() == [70.5, 82.0]


def test_single_column_without_header_is_read():
    df = files.read_table(b"0.12\n0.15\n0.98\n")
    assert len(df) == 3 and df.shape[1] == 1
