"""Stacking ensemble: prior-shift correction, referral rule and an end-to-end fit (ensemble.py)."""
import numpy as np
import pandas as pd
import pytest

from cardio_ml.ensemble import StackingEnsemble
from cardio_ml.models import KNNModel, LogisticRegressionModel, RandomForestModel, SVMModel
from cardio_ml.tracking import RunTracker


def test_prior_shift_moves_one_half_to_the_real_prevalence():
    p = np.array([0.1, 0.5, 0.9])
    q = StackingEnsemble.adjust_prior(p, prior=0.11)
    assert q[1] == pytest.approx(0.11)                 # logit(0.5) = 0, so p' = prior
    assert np.all(np.diff(q) > 0)                      # the ranking does not change
    assert np.all(q < p)                               # a rarer class lowers every probability


def test_prior_shift_does_nothing_on_balanced_data():
    p = np.array([0.2, 0.7])
    assert StackingEnsemble.adjust_prior(p, prior=0.5) == pytest.approx(p)


def test_referral_for_uncertain_answers_and_split_votes():
    ensemble = StackingEnsemble([LogisticRegressionModel(), KNNModel(), RandomForestModel(), SVMModel()])
    Z = pd.DataFrame([[0.9, 0.9, 0.9, 0.9], [0.6, 0.6, 0.4, 0.4], [0.45, 0.45, 0.45, 0.45], [0.1, 0.1, 0.1, 0.1]],
                     columns=ensemble.names)
    p = np.array([0.9, 0.8, 0.45, 0.1])
    assert ensemble.referral_mask(p, Z).tolist() == [False, True, True, False]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + 0.5 * X["b"] + rng.normal(0, 0.3, 400) > 0).astype(int))
    tracker = RunTracker("test", root=tmp_path_factory.mktemp("run"))
    models = [LogisticRegressionModel(), KNNModel(), RandomForestModel(), SVMModel()]
    return StackingEnsemble(models, cv=3).fit(X.iloc[:300], y.iloc[:300], tracker), X.iloc[300:], y.iloc[300:]


def test_meta_model_weights_are_never_negative(fitted):
    ensemble, _, _ = fitted
    weights = ensemble.weights()
    assert all(weights[name] >= 0 for name in ensemble.names)


def test_stacking_learns_a_simple_rule(fitted):
    ensemble, X_test, y_test = fitted
    p = ensemble.predict_proba(X_test)
    assert ((p >= 0.5) == y_test.values).mean() > 0.85
    assert np.all((p >= 0) & (p <= 1))
