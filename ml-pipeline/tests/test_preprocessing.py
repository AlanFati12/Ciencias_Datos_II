"""Cleaning steps and clinical feature engineering (preprocessing.py)."""
import pandas as pd
import pytest

from cardio_ml.preprocessing import (AgeInYears, BloodPressureConsistency, ClassBalancer, ClinicalFeatureEngineer,
                                     ConflictingLabelRemover, DuplicateRemover, RangeFilter)


def test_duplicated_records_are_kept_once():
    df = pd.DataFrame({"ap_hi": [120, 120, 140], "cardio": [0, 0, 1]})
    assert len(DuplicateRemover().transform(df)) == 2


def test_contradictory_records_are_removed():
    df = pd.DataFrame({"ap_hi": [120, 120, 140], "cardio": [0, 1, 1]})
    out = ConflictingLabelRemover("cardio").transform(df)
    assert out["ap_hi"].tolist() == [140]


def test_range_filter_removes_impossible_values():
    df = pd.DataFrame({"ap_hi": [120, 16020, 70, 250]})
    out = RangeFilter({"ap_hi": (80, 250)}).transform(df)
    assert out["ap_hi"].tolist() == [120, 250]


def test_systolic_must_exceed_diastolic():
    df = pd.DataFrame({"ap_hi": [120, 80, 100], "ap_lo": [80, 120, 100]})
    assert BloodPressureConsistency().transform(df)["ap_hi"].tolist() == [120]


def test_age_is_converted_from_days_to_years():
    out = AgeInYears().transform(pd.DataFrame({"age": [18262]}))
    assert out["age_years"].iloc[0] == pytest.approx(50.0)
    assert "age" not in out.columns


def test_bmi_map_and_pulse_pressure():
    df = pd.DataFrame({"weight": [80.0], "height": [200.0], "ap_hi": [120], "ap_lo": [80]})
    out = ClinicalFeatureEngineer().transform(df).iloc[0]
    assert out["bmi"] == pytest.approx(20.0)                 # 80 / 2.0^2
    assert out["map"] == pytest.approx(93.3, abs=0.05)       # (120 + 2*80) / 3
    assert out["pulse_pressure"] == 40                       # 120 - 80


@pytest.mark.parametrize("sbp, dbp, stage", [(115, 75, 0), (125, 75, 1), (130, 79, 2), (125, 85, 2),
                                             (145, 85, 3), (120, 95, 3), (190, 100, 4), (150, 125, 4)])
def test_blood_pressure_stage_follows_aha_2017(sbp, dbp, stage):
    out = ClinicalFeatureEngineer.bp_stage(pd.Series([sbp]), pd.Series([dbp]))
    assert out.iloc[0] == stage


def test_undersampling_gives_two_equal_classes():
    df = pd.DataFrame({"x": range(100), "y": [0] * 70 + [1] * 30})
    balancer = ClassBalancer("y")
    out = balancer.transform(df)
    assert out["y"].value_counts().to_dict() == {0: 30, 1: 30}
    assert balancer.counts_before == {0: 70, 1: 30}
