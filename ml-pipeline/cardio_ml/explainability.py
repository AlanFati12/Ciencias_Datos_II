"""Model-agnostic explainability (XAI): permutation importance of the ensemble.

Equation:  I(G) = accuracy(X) - (1/R) sum_r accuracy(X with the columns of G shuffled, repeat r)
What it does: shuffling the column(s) of G breaks their link with the label
while keeping their distribution. If the accuracy drops a lot, the system
relies on G; if it barely changes, G is not used. It works on the whole
ensemble (model-agnostic), not on one model's internals.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .plotting import BLUE, plt


def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y: pd.Series,
                           groups: dict[str, list[str]], n_repeats: int = 3,
                           seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = float((predict(X) == y).mean())
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]          # shuffle the whole group together
            drops.append(base - float((predict(Xp) == y).mean()))
        rows.append({"feature_group": name, "n_columns": len(cols), "importance": np.mean(drops),
                     "std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def plot_importance(table: pd.DataFrame, title: str):
    """Figure - permutation importance with +/- 1 SD error bars over the repeats.

    How to read it: longer bar = the combined system depends more on that
    input; bars near 0 = the input could be removed without losing accuracy.
    """
    t = table.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(t) + 1.2))
    ax.barh(t["feature_group"], t["importance"], xerr=t["std"], color=BLUE)
    ax.axvline(0, c="black", lw=0.8)
    ax.set(xlabel="Accuracy drop when the feature is shuffled", title=title)
    return fig
