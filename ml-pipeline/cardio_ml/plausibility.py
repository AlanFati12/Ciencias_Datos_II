"""Clinical plausibility audit: does the risk move in the right direction?

For a sample of real patients, one input at a time is swept across its range
(e.g. systolic pressure 100 -> 200 mmHg) with everything else fixed. A
violation is any step where the probability moves AGAINST the clinical
direction by more than `tol` (1 percentage point by default). The audit is run
for every base model and for the final answer of the ensemble.

Equation: for patient i, input j swept over v_1 < ... < v_m with expected
direction s_j (+1 risk must rise, -1 must fall):
    worst_ij = max_t max(0, -s_j * (P(v_{t+1}) - P(v_t)))
    share_violating_j = (1/n) sum_i 1[worst_ij > tol]
What it answers: "for how many patients does the model say that MORE blood
pressure (or cholesterol, age...) means LESS risk?" A trustworthy clinical
model should be near 0 %.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .ensemble import StackingEnsemble


def monotonicity_audit(ensemble: StackingEnsemble, raw: pd.DataFrame, featurize: Callable[[pd.DataFrame], pd.DataFrame],
                       sweeps: dict[str, list], expected: dict[str, int], tol: float = 0.01) -> pd.DataFrame:
    rows = []
    for feature, values in sweeps.items():
        cube = []
        for v in values:
            r = raw.copy()
            r[feature] = v
            if feature == "ap_hi":                     # keep the reading physiologically valid
                r["ap_lo"] = np.minimum(r["ap_lo"], v - 10)
            Z = ensemble.base_probabilities(featurize(r))
            Z["Stacking"] = ensemble.combine(Z)
            cube.append(Z.to_numpy())
        cube = np.stack(cube)                          # values x patients x models
        against = -expected[feature] * np.diff(cube, axis=0)
        worst = np.maximum(against, 0).max(axis=0)     # largest wrong-way step per patient
        for k, name in enumerate(Z.columns):
            rows.append({"feature": feature, "model": name,
                         "share_violating": float((worst[:, k] > tol).mean()),
                         "mean_worst_step_pp": float(worst[:, k].mean() * 100),
                         "net_effect_pp": float(expected[feature] * (cube[-1, :, k] - cube[0, :, k]).mean() * 100)})
    return pd.DataFrame(rows)


def plot_audit(table: pd.DataFrame, title: str):
    """Figure - share of patients with a wrong-way change, per factor (rows) and model (columns).

    How to read it: white / 0 % = always moves in the clinical direction;
    dark red = the model contradicts medicine for many patients.
    """
    from .plotting import plt, sns
    pivot = table.pivot(index="feature", columns="model", values="share_violating")
    pivot = pivot[[c for c in ["LR", "KNN", "RF", "SVM", "Stacking"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pivot) + 1.6))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds", vmin=0, vmax=1, cbar=False, ax=ax, linewidths=0.5)
    ax.set(xlabel="", ylabel="", title=title)
    return fig
