"""Shared visual style so every figure in the report looks consistent.

Colours come from a colour-blind-safe categorical palette (checked with a
CVD simulator: neighbouring colours stay distinguishable for protan, deutan
and tritan vision). Rules used in every figure:
  * one fixed colour per entity (a model or a class keeps its colour everywhere);
  * status colours (green / amber / red) only for good / warning / bad meanings,
    always together with a text label;
  * hairline grids, no top/right frame, values labelled directly when useful.
"""
import matplotlib.pyplot as plt
import seaborn as sns

from .palette import (AAMI_COLORS, AQUA, BLUE, CLASS_COLORS, CRITICAL, ENSEMBLE_COLOR, GOOD, GRID, INK,  # noqa: F401
                      INK_2, MODEL_COLORS, MUTED, ORANGE, RISK_COLORS, VIOLET, VOTE_COLORS, WARNING, YELLOW)


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 110, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, VIOLET]),
    })


apply_style()
