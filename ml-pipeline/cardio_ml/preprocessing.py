"""Transformation step of the ETL, built with ABC + polymorphism (week 4).

Each preprocessor has one responsibility and the Pipeline runs any list of
them through the same .transform() interface, logging the rows that every
step removes so the cleaning is fully traceable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import config
from .tracking import RunTracker


class BasePreprocessor(ABC):
    name: str = "preprocessor"

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new, transformed DataFrame (never modify the input)."""

    def detail(self) -> str:
        """Short human-readable explanation stored in the lineage log."""
        return ""


class Pipeline:
    def __init__(self, steps: list[BasePreprocessor], tracker: RunTracker) -> None:
        self.steps = steps
        self.tracker = tracker

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            before = len(df)
            df = step.transform(df)              # polymorphic call
            self.tracker.log_step(step.name, before, len(df), step.detail())
        return df.reset_index(drop=True)


# --------------------------------------------------------------- generic steps
class DropColumns(BasePreprocessor):
    name = "Drop identifier columns"

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def transform(self, df):
        return df.drop(columns=[c for c in self.columns if c in df.columns])

    def detail(self):
        return f"dropped {self.columns} (no predictive meaning)"


class DuplicateRemover(BasePreprocessor):
    name = "Remove duplicated records"

    def __init__(self, subset: list[str] | None = None) -> None:
        self.subset = subset

    def transform(self, df):
        return df.drop_duplicates(subset=self.subset)

    def detail(self):
        return "identical rows (all features and label) kept once"


class ConflictingLabelRemover(BasePreprocessor):
    """Removes feature vectors that appear with both labels (pure label noise)."""
    name = "Remove contradictory records"

    def __init__(self, target: str) -> None:
        self.target = target

    def transform(self, df):
        features = [c for c in df.columns if c != self.target]
        n_labels = df.groupby(features, dropna=False)[self.target].transform("nunique")
        return df[n_labels == 1]

    def detail(self):
        return "same feature values with different diagnosis"


class RangeFilter(BasePreprocessor):
    name = "Filter physiologically impossible values"

    def __init__(self, ranges: dict[str, tuple[float, float]], name: str | None = None) -> None:
        self.ranges = ranges
        self.name = name or self.name
        self._removed: dict[str, int] = {}

    def transform(self, df):
        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in self.ranges.items():
            if col in df.columns:
                bad = ~df[col].between(lo, hi)
                self._removed[col] = int((bad & mask).sum())
                mask &= ~bad
        return df[mask]

    def detail(self):
        parts = [f"{c} outside [{lo}, {hi}]: {self._removed.get(c, 0)}" for c, (lo, hi) in self.ranges.items()
                 if c in self._removed]
        return "; ".join(parts)


class FiniteFilter(BasePreprocessor):
    name = "Remove non-finite values"

    def transform(self, df):
        numeric = df.select_dtypes("number")
        return df[np.isfinite(numeric).all(axis=1)]

    def detail(self):
        return "NaN or infinite values in any numeric column"


class ClassBalancer(BasePreprocessor):
    """Random undersampling of the majority class to a 50/50 split."""
    name = "Balance classes (random undersampling)"

    def __init__(self, target: str, seed: int = config.SEED) -> None:
        self.target = target
        self.seed = seed
        self.counts_before: dict = {}
        self.counts_after: dict = {}

    def transform(self, df):
        self.counts_before = df[self.target].value_counts().sort_index().to_dict()
        n = min(self.counts_before.values())
        parts = [g.sample(n=n, random_state=self.seed) for _, g in df.groupby(self.target)]
        out = pd.concat(parts).sample(frac=1, random_state=self.seed)
        self.counts_after = out[self.target].value_counts().sort_index().to_dict()
        return out

    def detail(self):
        return f"class counts {self.counts_before} -> {self.counts_after}"


# -------------------------------------------------------------- clinical steps
class AgeInYears(BasePreprocessor):
    name = "Convert age from days to years"

    def transform(self, df):
        out = df.copy()
        out["age_years"] = (out["age"] / 365.25).round(1)
        return out.drop(columns="age")

    def detail(self):
        return "age_years = age_days / 365.25"


class BloodPressureConsistency(BasePreprocessor):
    name = "Check systolic > diastolic"

    def transform(self, df):
        return df[df["ap_hi"] > df["ap_lo"]]

    def detail(self):
        return "systolic pressure must exceed diastolic pressure"


class ClinicalFeatureEngineer(BasePreprocessor):
    """Derived clinical variables used by physicians.

    BMI = weight_kg / height_m^2        (body-mass index)
        Normalises weight by height: 18.5-25 normal, 25-30 overweight, >= 30 obesity.
    MAP = (SBP + 2 * DBP) / 3           (mean arterial pressure)
        Average pressure over one heartbeat; the heart spends ~2/3 of the cycle
        in diastole, hence the double weight of the diastolic pressure.
    PP  = SBP - DBP                     (pulse pressure)
        Reflects arterial stiffness; a wide PP is a risk marker in older adults.
    bp_stage = AHA/ACC 2017 category from SBP/DBP (see bp_stage()).
    MAP and PP are exact linear combinations of SBP and DBP, so they are kept
    for the EDA but not given to the models (VIF -> infinity).
    """
    name = "Engineer clinical features"

    def transform(self, df):
        out = df.copy()
        out["bmi"] = (out["weight"] / (out["height"] / 100) ** 2).round(2)
        out["map"] = ((out["ap_hi"] + 2 * out["ap_lo"]) / 3).round(1)
        out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
        out["bp_stage"] = self.bp_stage(out["ap_hi"], out["ap_lo"])
        return out

    @staticmethod
    def bp_stage(sbp: pd.Series, dbp: pd.Series) -> pd.Series:
        """AHA/ACC 2017 blood-pressure category (the first rule that matches wins):
        4 crisis SBP > 180 or DBP > 120 | 3 stage-2 SBP >= 140 or DBP >= 90 |
        2 stage-1 SBP >= 130 or DBP >= 80 | 1 elevated SBP >= 120 | 0 normal."""
        stage = np.select(
            [(sbp > 180) | (dbp > 120), (sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120],
            [4, 3, 2, 1], default=0)
        return pd.Series(stage, index=sbp.index)

    def detail(self):
        return "added bmi, map, pulse_pressure, bp_stage (AHA 2017)"
