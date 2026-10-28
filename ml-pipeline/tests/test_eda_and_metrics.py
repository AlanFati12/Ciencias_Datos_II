"""EDA checklist tables (eda.py) and evaluation metrics (evaluation.py)."""
import numpy as np
import pandas as pd
import pytest

from cardio_ml import eda
from cardio_ml.evaluation import classification_metrics


def test_missing_values_table_counts_empty_and_infinite_cells():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.inf, 2.0, 3.0], "c": ["x", None, "z"]})
    table = eda.missing_values_table(df).set_index("variable")
    assert table.loc["a", "missing"] == 1 and table.loc["b", "non_finite"] == 1 and table.loc["c", "missing"] == 1
    assert table.loc["a", "missing_pct"] == pytest.approx(1 / 3)


def test_frequency_table_has_relative_frequencies_and_mode():
    df = pd.DataFrame({"chol": [1, 1, 1, 2, 3], "y": [0, 1, 0, 1, 1]})
    table = eda.frequency_table(df, ["chol"], "y")
    assert table["relative_freq"].sum() == pytest.approx(1.0)
    assert table.loc[table["is_mode"], "level"].tolist() == [1]
    assert table.set_index("level").loc[1, "target_rate"] == pytest.approx(1 / 3)


def test_assumption_tests_tell_normal_from_skewed_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"normal": rng.normal(size=2000), "skewed": rng.exponential(size=2000),
                       "y": np.repeat([0, 1], 1000)})
    df.loc[df["y"] == 1, "skewed"] *= 3                    # class 1 has a larger spread
    table = eda.assumption_tests(df, ["normal", "skewed"], "y").set_index("variable")
    assert table.loc["normal", "normal_all_classes"]
    assert not table.loc["skewed", "normal_all_classes"]
    assert table.loc["normal", "equal_variances"] and not table.loc["skewed", "equal_variances"]


def test_metrics_match_the_confusion_matrix_by_hand():
    # TP = 3, FN = 1, TN = 4, FP = 2
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.2, 0.3, 0.6, 0.7]
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall_sensitivity"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["mcc"] == pytest.approx((3 * 4 - 2 * 1) / np.sqrt(5 * 4 * 6 * 5))
