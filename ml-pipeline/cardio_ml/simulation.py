"""Results simulation: trace new cases through the four models and the meta-model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ensemble import StackingEnsemble
from .plotting import plt, sns


def simulate_cases(ensemble: StackingEnsemble, X: pd.DataFrame, case_ids: list[str],
                   y_true: pd.Series | None = None) -> pd.DataFrame:
    Z = ensemble.base_probabilities(X)
    p = ensemble.combine(Z)
    table = Z.copy()
    table.insert(0, "case", case_ids)
    table["soft_vote"] = ensemble.soft_vote(Z)
    table["final_probability"] = p
    table["final_prediction"] = (p >= 0.5).astype(int)
    table["models_agreeing"] = np.maximum((Z >= 0.5).sum(axis=1), (Z < 0.5).sum(axis=1)).values
    table["referred_to_clinician"] = ensemble.referral_mask(p, Z)
    if y_true is not None:
        table["true_label"] = np.asarray(y_true)
        table["correct"] = table["final_prediction"] == table["true_label"]
    return table.reset_index(drop=True)


def save_case_traces(ensemble: StackingEnsemble, X: pd.DataFrame, case_ids: list[str], path: Path,
                     extra: dict[str, dict] | None = None) -> None:
    """One JSON entry per case: inputs -> each model -> meta-model -> answer."""
    traces = []
    for i, cid in enumerate(case_ids):
        row = X.iloc[[i]]
        traces.append({"case": cid,
                       "inputs": {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                                  for k, v in row.iloc[0].items()},
                       **ensemble.explain_case(row), **((extra or {}).get(cid, {}))})
    path.write_text(json.dumps(traces, indent=2))


def plot_simulation(table: pd.DataFrame, names: list[str], title: str, positive: str = "Positive"):
    """Figure - simulation heat map: one row per case, one column per model.

    How to read it: each cell is P(positive) from 0 (blue) to 1 (red). The
    columns left of the black line are the four models on their own; the last
    column is the single final answer of the stacking meta-model. "[refer]"
    marks cases sent to a clinician and "(WRONG)" errors against the truth.
    """
    cols = names + ["soft_vote", "final_probability"]
    data = table.set_index("case")[cols].rename(columns={"soft_vote": "Soft vote",
                                                          "final_probability": "STACKING\n(final)"})
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 4, 0.48 * len(table) + 1.6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="RdBu_r", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": f"P({positive})"}, linewidths=0.5)
    ax.axvline(len(names), c="black", lw=2)
    labels = []
    for _, r in table.iterrows():
        tag = "  [refer]" if r["referred_to_clinician"] else ""
        if "true_label" in r:
            tag += "  (ok)" if r["correct"] else "  (WRONG)"
        labels.append(f"{r['case']}{tag}")
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    ax.set(xlabel="", ylabel="", title=title)
    return fig
