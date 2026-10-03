"""Unsupervised risk segmentation (Hito 1, objective 3): PCA + K-means.

K-means minimises the within-cluster sum of squares
    J = sum_k sum_{x in C_k} ||x - mu_k||^2
on the PCA-reduced, standardised clinical profile. Clusters are then ranked
by their observed CVD prevalence and named Low / Medium / High risk.

    What it does: groups patients with similar profiles WITHOUT using the
    diagnosis (unsupervised). Each patient goes to the nearest centroid mu_k.
Choice of k:
    Elbow - J(k) always falls as k grows; the "elbow" is where it stops falling fast.
    Silhouette s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is the mean
    distance to the own cluster and b(i) to the nearest other cluster
    (1 = well separated, 0 = on the border). k = 3 matches the clinical
    Low / Medium / High scale used in Hito 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

SEGMENT_FEATURES = ["age_years", "bmi", "ap_hi", "ap_lo", "pulse_pressure", "cholesterol", "gluc"]
RISK_LEVELS = ["Low", "Medium", "High"]


class RiskSegmenter:
    def __init__(self, k: int = 3, features: list[str] = SEGMENT_FEATURES, variance: float = 0.90,
                 seed: int = config.SEED) -> None:
        self.k = k
        self.features = features
        self.variance = variance
        self.seed = seed

    def _embed(self, df: pd.DataFrame) -> np.ndarray:
        return self.pca_.transform(self.scaler_.transform(df[self.features]))

    def scan_k(self, df: pd.DataFrame, ks=range(2, 9), sample: int = 8000) -> pd.DataFrame:
        s = df.sample(min(sample, len(df)), random_state=self.seed)
        Z = PCA(self.variance, random_state=self.seed).fit_transform(StandardScaler().fit_transform(s[self.features]))
        rows = []
        for k in ks:
            km = KMeans(k, n_init=10, random_state=self.seed).fit(Z)
            rows.append({"k": k, "inertia": km.inertia_,
                         "silhouette": silhouette_score(Z, km.labels_, random_state=self.seed)})
        return pd.DataFrame(rows)

    def fit(self, df: pd.DataFrame, target: str) -> "RiskSegmenter":
        self.scaler_ = StandardScaler().fit(df[self.features])
        self.pca_ = PCA(self.variance, random_state=self.seed).fit(self.scaler_.transform(df[self.features]))
        self.kmeans_ = KMeans(self.k, n_init=10, random_state=self.seed).fit(self._embed(df))
        prevalence = pd.Series(df[target].values).groupby(self.kmeans_.labels_).mean().sort_values()
        names = RISK_LEVELS if self.k == 3 else [f"Level {i + 1}" for i in range(self.k)]
        self.cluster_to_level_ = {int(c): names[i] for i, c in enumerate(prevalence.index)}
        return self

    def assign(self, df: pd.DataFrame) -> pd.Series:
        clusters = self.kmeans_.predict(self._embed(df))
        return pd.Series([self.cluster_to_level_[int(c)] for c in clusters], index=df.index, name="risk_level")

    def profile(self, df: pd.DataFrame, levels: pd.Series, target: str) -> pd.DataFrame:
        table = df[self.features + [target]].groupby(levels).mean()
        table["patients"] = levels.value_counts()
        table = table.rename(columns={target: "cvd_prevalence"})
        return table.reindex([lvl for lvl in RISK_LEVELS if lvl in table.index])


def plot_k_scan(scan: pd.DataFrame, chosen: int):
    """Figure - elbow (J vs. k) and mean silhouette vs. k; the dashed line marks the chosen k."""
    from .plotting import ORANGE, plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(scan["k"], scan["inertia"], marker="o")
    a1.set(xlabel="Number of clusters k", ylabel="Within-cluster sum of squares J", title="Elbow method")
    a2.plot(scan["k"], scan["silhouette"], marker="o", c=ORANGE)
    a2.set(xlabel="Number of clusters k", ylabel="Mean silhouette", title="Silhouette coefficient")
    for ax in (a1, a2):
        ax.axvline(chosen, ls="--", c="grey")
    fig.tight_layout()
    return fig


def plot_segments(seg: RiskSegmenter, df: pd.DataFrame, levels: pd.Series, n: int = 6000):
    """Figure - test patients in the plane of the first two principal components, coloured by segment."""
    from .plotting import RISK_COLORS, plt
    s = df.sample(min(n, len(df)), random_state=seg.seed)
    Z = seg._embed(s)
    lv = levels.loc[s.index]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for level in RISK_LEVELS:
        m = (lv == level).values
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.45, c=RISK_COLORS[level], label=f"{level} risk")
    ev = seg.pca_.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.1%})", ylabel=f"PC2 ({ev[1]:.1%})",
           title="Patient segments found by K-means (PCA space)")
    ax.legend(markerscale=4)
    return fig


def plot_segment_profile(profile: pd.DataFrame, features: list[str]):
    """Figure - mean profile of each segment (left) and its observed CVD prevalence (right).

    Left cells show the real mean; the colour is that mean standardised across
    segments (red = highest of the three). How to read it: the High-risk
    segment should be older, with higher pressure and more CVD.
    """
    from .plotting import RISK_COLORS, plt, sns
    z = (profile[features] - profile[features].mean()) / profile[features].std(ddof=0).replace(0, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    sns.heatmap(z, annot=profile[features].round(1), fmt="", cmap="RdYlGn_r", center=0, ax=a1, cbar=False)
    a1.set(title="Mean clinical profile per segment (colour = relative level)", ylabel="")
    bars = a2.bar(profile.index, profile["cvd_prevalence"], color=[RISK_COLORS[i] for i in profile.index])
    for b, (p, n) in zip(bars, profile[["cvd_prevalence", "patients"]].values):
        a2.text(b.get_x() + b.get_width() / 2, p, f"{p:.0%}\nn={int(n):,}", ha="center", va="bottom", fontsize=8)
    a2.set(ylim=(0, 1.15), title="Observed CVD prevalence", ylabel="")
    fig.tight_layout()
    return fig
