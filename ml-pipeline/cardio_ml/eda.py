"""Exploratory data analysis figures (INFOVIZ, weeks 7-8).

Covers the checklist from the week-7 class: univariate (histogram + KDE,
IQR boxplots), bivariate (split violins, scatter), multivariate (pairplot),
Pearson vs. Spearman triangular heatmaps and multicollinearity (VIF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .config import AAMI_CLASSES as AAMI_SYMBOLS
from .plotting import AAMI_COLORS, BLUE, CLASS_COLORS, CRITICAL, GOOD, MUTED, ORANGE, WARNING, plt, sns

CLINICAL_LABELS = {0: "No CVD", 1: "CVD"}
ECG_LABELS = {0: "Normal beat", 1: "Abnormal beat"}

UNITS = {"age_years": "Age (years)", "height": "Height (cm)", "weight": "Weight (kg)", "bmi": "BMI (kg/m²)",
         "ap_hi": "Systolic BP (mmHg)", "ap_lo": "Diastolic BP (mmHg)", "map": "Mean arterial pressure (mmHg)",
         "pulse_pressure": "Pulse pressure (mmHg)", "pre_rr": "Previous RR interval (s)",
         "post_rr": "Next RR interval (s)", "local_rr": "Local mean RR (s)", "rr_ratio": "Previous RR / local RR",
         "post_pre_ratio": "Next RR / previous RR", "amplitude_mv": "Beat amplitude (mV)",
         "template_corr": "Correlation with patient's dominant beat", "template_dist": "RMS distance to dominant beat",
         "amplitude_rel": "Amplitude / patient median", "rr_rel": "Previous RR / patient median RR"}


def _palette(labels: dict) -> dict:
    return {labels[k]: CLASS_COLORS[k] for k in labels}


# ================================================================== generic
def plot_cleaning_funnel(lineage: pd.DataFrame):
    """Figure - data lineage: rows left after each cleaning step (traceability).

    How to read it: read top to bottom; each bar is the dataset after one step
    and the number in brackets is how many rows that step removed and why
    (the reason is in TRACEABILITY.md). Nothing is removed silently.
    """
    data = lineage[lineage["rows_before"] > 0]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(data) + 1.2))
    ax.barh(data["step"], data["rows_after"], color=BLUE)
    for i, (after, removed) in enumerate(zip(data["rows_after"], data["removed"])):
        ax.text(after, i, f"  {after:,}" + (f"  (−{removed:,})" if removed else ""), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, data["rows_before"].max() * 1.25)
    ax.set(xlabel="Rows remaining", title="Data lineage: rows after each cleaning step")
    return fig


def plot_class_balance(before: dict, after: dict, labels: dict):
    """Figure - class counts before and after random undersampling.

    Undersampling keeps every case of the minority class and draws the same
    number n = min(n_0, n_1) at random from the majority class, so the
    modelling dataset is 50/50 (rubric: balanced dataset). With balanced
    classes, accuracy cannot be inflated by always answering the common class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    top = max(max(before.values()), max(after.values())) * 1.22
    for ax, counts, title in ((axes[0], before, "Before balancing"), (axes[1], after, "After balancing")):
        keys = sorted(counts)
        bars = ax.bar([labels[k] for k in keys], [counts[k] for k in keys], color=[CLASS_COLORS[k] for k in keys])
        total = sum(counts.values())
        for b, k in zip(bars, keys):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{counts[k]:,}\n({counts[k] / total:.1%})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
    axes[0].set_ylim(0, top)
    axes[0].set_ylabel("Records")
    fig.suptitle("Class balance of the modelling dataset", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, cols: list[str], target: str, labels: dict, ncols: int = 3):
    """Figure - histogram + kernel density estimate (KDE) of each variable, per class.

    KDE: f(x) = 1/(n h) sum_i K((x - x_i) / h) with a Gaussian kernel K, i.e. a
    smoothed histogram. Skewness = E[(x - mu)^3] / sigma^3 (0 = symmetric,
    > 0 = long right tail) is printed above each panel.
    How to read it: if the two colours are shifted apart, the variable helps to
    separate the classes; if they overlap completely, it carries little signal.
    """
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows))
    data = df.assign(_cls=df[target].map(labels))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(data, x=col, hue="_cls", hue_order=list(labels.values()), bins=40, stat="density",
                     common_norm=False, kde=True, element="step", palette=_palette(labels), ax=ax, alpha=0.3)
        ax.set_xlabel(UNITS.get(col, col))
        ax.set_title(f"skewness = {df[col].skew():.2f}", fontsize=10, fontweight="normal")
        if ax.get_legend():
            ax.get_legend().set_title("")
    for ax in list(axes.flat)[len(cols):]:
        ax.remove()
    fig.suptitle("Univariate distributions by class (histogram + KDE)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_iqr_boxplots(df: pd.DataFrame, cols: list[str], target: str, labels: dict, ncols: int = 3):
    """Figure - boxplots by class with the IQR outlier rule.

    IQR = Q3 - Q1; a value is flagged as an outlier if it is below
    Q1 - 1.5 IQR or above Q3 + 1.5 IQR. The box spans Q1..Q3, the line is the
    median. Flagged values are investigated, not deleted automatically: only
    physiologically impossible values are removed (see RangeFilter).
    """
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    data = df.assign(_cls=df[target].map(labels))
    for ax, col in zip(axes.flat, cols):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        sns.boxplot(data, x="_cls", y=col, hue="_cls", order=list(labels.values()), palette=_palette(labels), ax=ax,
                    fliersize=1.5, legend=False)
        ax.set(xlabel="", ylabel=UNITS.get(col, col))
        ax.set_title(f"IQR outliers: {n_out:,} ({n_out / len(df):.1%})", fontsize=10, fontweight="normal")
    for ax in list(axes.flat)[len(cols):]:
        ax.remove()
    fig.suptitle("Boxplots and IQR rule (outliers investigated, not deleted automatically)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlations(df: pd.DataFrame, cols: list[str]):
    """Figure - triangular heat maps of Pearson and Spearman correlation.

    Pearson:  r = cov(X, Y) / (sd_X sd_Y)           measures LINEAR association.
    Spearman: rho = Pearson r of the ranks of X and Y  measures MONOTONIC association
              (robust to outliers and to non-linear but ordered relations).
    How to read it: red = both grow together, blue = one grows as the other
    falls, white = no relation. Large differences between the two maps reveal
    outliers or curved relations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    mask = np.triu(np.ones((len(cols), len(cols)), dtype=bool))
    for ax, method in zip(axes, ("pearson", "spearman")):
        corr = df[cols].corr(method=method)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                    annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_title(f"{method.capitalize()} correlation (triangular)")
    fig.tight_layout()
    return fig


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Variance inflation factor: VIF_j = 1 / (1 - R_j^2).

    R_j^2 is the R^2 of a linear regression that predicts variable j from all
    the others. VIF = 1 means j is independent; VIF > 10 means j is almost a
    copy of the others (multicollinearity), which makes the coefficients of a
    linear model unstable. Variables with VIF -> infinity are dropped.
    """
    X = StandardScaler().fit_transform(df[cols])
    rows = []
    for j, col in enumerate(cols):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        rows.append({"feature": col, "r2_vs_others": r2, "vif": 1 / max(1 - r2, 1e-12)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def plot_vif(vif: pd.DataFrame):
    """Figure - VIF per variable: green < 5 (fine), yellow 5-10 (moderate), red > 10 (severe)."""
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(vif) + 1))
    colors = [CRITICAL if v > 10 else WARNING if v > 5 else GOOD for v in vif["vif"]]
    ax.barh(vif["feature"], vif["vif"].clip(upper=50), color=colors)
    for i, v in enumerate(vif["vif"]):
        ax.text(min(v, 50), i, f" {v:.1f}" if v < 1e4 else " ≈ ∞ (linear combination)", va="center", fontsize=8)
    ax.axvline(5, ls="--", c=MUTED, lw=1)
    ax.axvline(10, ls="--", c=CRITICAL, lw=1)
    ax.invert_yaxis()
    ax.set(xlabel="Variance inflation factor (clipped at 50)", title="Multicollinearity check (VIF > 10 = severe)")
    return fig


def plot_pca_projection(X: pd.DataFrame, y: pd.Series, labels: dict, n: int = 6000, seed: int = 42):
    """Figure - 2-D principal component analysis (PCA) of all model inputs.

    PCA finds the orthogonal directions of largest variance: the eigenvectors
    of the covariance matrix S = Z^T Z / (n - 1) of the standardised data Z.
    PC1 and PC2 are the two directions that keep the most information; the
    axis labels give the share of variance each one explains.
    How to read it: if the two colours form separate clouds, the classes are
    easy to separate; heavy overlap anticipates a lower accuracy.
    """
    idx = X.sample(min(n, len(X)), random_state=seed).index
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(StandardScaler().fit_transform(X.loc[idx]))
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for k, label in labels.items():
        m = (y.loc[idx] == k).values
        ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.4, c=CLASS_COLORS[k], label=label)
    ev = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.1%} variance)", ylabel=f"PC2 ({ev[1]:.1%} variance)",
           title="2-D PCA projection: how separable are the classes?")
    ax.legend(markerscale=4)
    return fig


# ================================================================= clinical
def plot_raw_outliers(raw: pd.DataFrame):
    """Figure - boxplots of the RAW measurements before cleaning.

    Shows the impossible values in the source data (e.g. systolic pressure of
    16,020 mmHg or negative values) that justify the range filters.
    """
    cols = [("ap_hi", "Systolic BP (mmHg)"), ("ap_lo", "Diastolic BP (mmHg)"),
            ("height", "Height (cm)"), ("weight", "Weight (kg)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (col, label) in zip(axes, cols):
        sns.boxplot(y=raw[col], ax=ax, color="#BBBBBB", fliersize=2)
        if col.startswith("ap_"):
            ax.set_yscale("symlog", linthresh=300)
        ax.set(ylabel=label, title=f"min {raw[col].min():,.0f} · max {raw[col].max():,.0f}")
    fig.suptitle("Raw data: impossible measurements before cleaning (BP axes in symlog scale)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_split_violins(df: pd.DataFrame, target: str):
    """Figure - split violin plots: a categorical factor (x) vs. a numeric variable (y).

    Each violin is a mirrored KDE; the left half is one class and the right
    half the other, with dashed quartiles. How to read it: when the right half
    sits higher than the left, that variable is higher in the sick group.
    """
    data = df.assign(Diagnosis=df[target].map(CLINICAL_LABELS))
    specs = [("cholesterol", "ap_hi", "Cholesterol level (1 normal · 2 above · 3 well above)"),
             ("gluc", "bmi", "Glucose level (1 normal · 2 above · 3 well above)"),
             ("bp_stage", "age_years", "BP stage (0 normal … 4 crisis, AHA 2017)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, (ax, (x, y, xlabel)) in enumerate(zip(axes, specs)):
        sns.violinplot(data, x=x, y=y, hue="Diagnosis", hue_order=list(CLINICAL_LABELS.values()), split=True,
                       inner="quart", ax=ax, palette=_palette(CLINICAL_LABELS), density_norm="width", cut=0,
                       legend=i == 0)
        ax.set(xlabel=xlabel, ylabel=UNITS.get(y, y))
    handles, names = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Bivariate analysis: categorical vs. numeric with split violins (split=True)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def plot_categorical_rates(df: pd.DataFrame, cols: list[str], target: str):
    """Figure - prevalence of the disease for each level of a categorical factor.

    prevalence(level) = cases with CVD in that level / people in that level.
    How to read it: bars above the dashed line (overall prevalence) mark risk
    factors; n under each bar is the group size (small groups are less reliable).
    """
    names = {"gender": "Gender (1 female · 2 male)", "cholesterol": "Cholesterol", "gluc": "Glucose",
             "smoke": "Smoker", "alco": "Alcohol intake", "active": "Physically active", "bp_stage": "BP stage"}
    fig, axes = plt.subplots(2, int(np.ceil(len(cols) / 2)), figsize=(4 * int(np.ceil(len(cols) / 2)), 7))
    for ax, col in zip(axes.flat, cols):
        rate = df.groupby(col)[target].agg(["mean", "size"])
        bars = ax.bar(rate.index.astype(str), rate["mean"], color=ORANGE)
        for b, (m, n) in zip(bars, rate.values):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0%}\nn={int(n):,}", ha="center", va="bottom", fontsize=8)
        ax.axhline(df[target].mean(), ls="--", c="grey", lw=1)
        ax.set(ylim=(0, 1.15), xlabel=names.get(col, col), ylabel="CVD prevalence")
    for ax in list(axes.flat)[len(cols):]:
        ax.remove()
    fig.suptitle("CVD prevalence by categorical factor (dashed line = overall prevalence)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_scatter_pairs(df: pd.DataFrame, target: str, n: int = 6000, seed: int = 42):
    """Figure - scatter plots coloured by class (small random jitter so repeated values are visible)."""
    s = df.sample(min(n, len(df)), random_state=seed)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (x, y) in zip(axes, (("ap_lo", "ap_hi"), ("age_years", "bmi"))):
        for k, label in CLINICAL_LABELS.items():
            m = s[target] == k
            jitter = rng.normal(0, 0.8, (m.sum(), 2))
            ax.scatter(s.loc[m, x] + jitter[:, 0], s.loc[m, y] + jitter[:, 1], s=5, alpha=0.35,
                       c=CLASS_COLORS[k], label=label)
        ax.set(xlabel=UNITS[x], ylabel=UNITS[y])
        ax.legend(markerscale=4)
    axes[0].set_title("Diastolic vs. systolic pressure (jittered)")
    axes[1].set_title("Age vs. BMI (jittered)")
    fig.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame, cols: list[str], target: str, n: int = 2500, seed: int = 42):
    """Figure - pairplot: every pair of variables as a scatter plot, KDE of each on the diagonal."""
    s = df.sample(min(n, len(df)), random_state=seed).assign(Diagnosis=lambda d: d[target].map(CLINICAL_LABELS))
    grid = sns.pairplot(s, vars=cols, hue="Diagnosis", palette=_palette(CLINICAL_LABELS), corner=True,
                        plot_kws={"s": 6, "alpha": 0.4}, diag_kind="kde", height=2.0)
    grid.figure.suptitle("Multivariate view: pairplot of the main numeric variables", fontweight="bold", y=1.01)
    return grid.figure


# ====================================================================== ECG
def plot_ecg_strip(raw: np.ndarray, fs_raw: float, clean: np.ndarray, fs_clean: float, annotation,
                   start_s: float = 0.0, seconds: float = 8.0):
    """Figure - the ETL of an unstructured signal: raw ECG (top) vs. clean ECG (bottom).

    Shows the effect of the band-pass filter (baseline wander and noise removed)
    and of the resampling, with the cardiologists' label of every beat
    (red = abnormal). This is where an unstructured signal becomes table rows.
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True)
    t_raw = np.arange(len(raw)) / fs_raw
    t_cln = np.arange(len(clean)) / fs_clean
    m1 = (t_raw >= start_s) & (t_raw < start_s + seconds)
    m2 = (t_cln >= start_s) & (t_cln < start_s + seconds)
    a1.plot(t_raw[m1], raw[m1], lw=0.8, c="grey")
    a1.set(ylabel="mV", title=f"1) Raw signal ({fs_raw:.0f} Hz) — baseline wander and noise")
    a2.plot(t_cln[m2], clean[m2], lw=0.9, c=BLUE)
    samples, symbols = np.asarray(annotation.sample), np.asarray(annotation.symbol)
    t_ann = samples / fs_raw
    beat_symbols = [s for group in ("N", "S", "V", "F") for s in AAMI_SYMBOLS[group]]
    sel = (t_ann >= start_s) & (t_ann < start_s + seconds) & np.isin(symbols, beat_symbols)
    for t, sym in zip(t_ann[sel], symbols[sel]):
        i = min(int(round(t * fs_clean)), len(clean) - 1)
        a2.annotate(sym, (t, clean[i]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9,
                    color=CRITICAL if sym not in "NLRej" else "black", fontweight="bold")
    a2.margins(y=0.2)
    a2.set(xlabel="Time (s)", ylabel="mV",
           title=f"2) Band-pass 0.5–40 Hz + resampled to {fs_clean:.0f} Hz, with cardiologist beat labels "
                 "(red = abnormal)")
    fig.tight_layout()
    return fig


def plot_beat_counts(df: pd.DataFrame):
    """Figure - heartbeats per AAMI class and database, on a log scale (each tick = x10)."""
    counts = df.groupby(["database", "aami_class"]).size().unstack(fill_value=0)[["N", "S", "V", "F"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.T.plot(kind="bar", ax=ax, color=[BLUE, ORANGE], log=True, rot=0)
    for c in ax.containers:
        ax.bar_label(c, fmt="{:,.0f}", fontsize=8)
    ax.set(xlabel="AAMI beat class (N normal · S supraventricular · V ventricular · F fusion)",
           ylabel="Beats (log scale)", title="Heartbeats extracted per database and class")
    ax.legend(title="Database")
    return fig


def plot_mean_beats(df: pd.DataFrame, wave_cols: list[str], fs: int, n_before: int):
    """Figure - mean beat shape +/- 1 SD for each AAMI class (time 0 = R peak).

    How to read it: a ventricular beat (V) is wide and deformed, a normal beat
    (N) is narrow; a supraventricular beat (S) looks almost normal, which is why
    the RR (timing) features are needed to detect it.
    """
    t = (np.arange(len(wave_cols)) - n_before) / fs * 1000
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, cls in zip(axes, ["N", "S", "V", "F"]):
        w = df.loc[df["aami_class"] == cls, wave_cols].values
        mu, sd = w.mean(axis=0), w.std(axis=0)
        ax.plot(t, mu, c=AAMI_COLORS[cls], lw=2)
        ax.fill_between(t, mu - sd, mu + sd, color=AAMI_COLORS[cls], alpha=0.2)
        ax.axvline(0, ls=":", c="grey")
        ax.set(title=f"Class {cls} (n={len(w):,})", xlabel="Time from R peak (ms)")
    axes[0].set_ylabel("Normalised amplitude (z)")
    fig.suptitle("Mean beat shape ± 1 SD per AAMI class", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rr_features(df: pd.DataFrame, cols: list[str], target: str, ncols: int = 5):
    """Figure - violin plots of the rhythm, amplitude and patient-relative features by class.

    How to read it: e.g. abnormal beats have a smaller rr_ratio (they arrive
    early: previous RR shorter than the local average) and a lower correlation
    with the patient's dominant beat.
    """
    data = df.assign(Beat=df[target].map(ECG_LABELS))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.6 * nrows))
    for ax in list(axes.flat)[len(cols):]:
        ax.remove()
    for ax, col in zip(axes.flat, cols):
        lo, hi = df[col].quantile([0.005, 0.995])
        sns.violinplot(data[data[col].between(lo, hi)], x="Beat", y=col, hue="Beat", order=list(ECG_LABELS.values()),
                       palette=_palette(ECG_LABELS), ax=ax, cut=0, inner="quart", legend=False)
        ax.set(xlabel="", ylabel=UNITS.get(col, col))
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Rhythm, amplitude and patient-relative features by class (0.5–99.5 percentile shown)",
                 fontweight="bold")
    fig.tight_layout()
    return fig