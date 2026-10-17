"""Tier 2 logic (no UI): read an ECG, find and classify every beat."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .files import read_table
from .model_store import SAMPLE_DATA, Tier

from cardio_ml.ecg_etl import BeatExtractor  # noqa: E402

MIN_SECONDS = 20


def demo_catalog() -> list[dict]:
    path = SAMPLE_DATA / "ecg" / "demos.json"
    return json.loads(path.read_text()) if path.exists() else []


def load_demo(entry: dict) -> tuple[np.ndarray, float]:
    df = pd.read_csv(SAMPLE_DATA / "ecg" / entry["file"])
    return df["ecg_mv"].to_numpy(float), float(entry["fs"])


def read_csv(data: bytes) -> pd.DataFrame:
    """Numeric columns of an uploaded CSV (see files.read_table for the accepted formats)."""
    return read_table(data).select_dtypes("number")


def read_wfdb(files: dict[str, bytes]) -> tuple[np.ndarray, float, list[str]]:
    """files: {filename: content} with at least one .hea and its .dat."""
    import wfdb
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            (Path(tmp) / name).write_bytes(content)
        header = next(n for n in files if n.endswith(".hea"))
        record = wfdb.rdrecord(str(Path(tmp) / header[:-4]))
        return np.nan_to_num(record.p_signal), float(record.fs), list(record.sig_name)


def analyze(tier: Tier, signal: np.ndarray, fs: float) -> dict:
    """Detect the beats, build the features and classify them with the stacking ensemble."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) / fs < MIN_SECONDS:
        raise ValueError("too_short")
    extractor = BeatExtractor()
    beats, clean, peaks = extractor.extract_signal(signal, fs)
    if len(beats) < 5:
        raise ValueError("no_beats")
    lo, hi = tier.schema.get("rr_valid_range_s", (0.2, 3.0))
    prior = tier.schema.get("deployment_prior")
    valid = beats["pre_rr"].between(lo, hi) & beats["post_rr"].between(lo, hi)
    Z = tier.ensemble.base_probabilities(beats.loc[valid, tier.features])
    beats["t"] = beats["sample"] / fs
    beats["p"] = np.nan
    beats.loc[valid, "p"] = tier.ensemble.combine(Z, prior)
    for name in Z.columns:            # base-model opinions on the same (corrected) scale
        beats[f"p_{name}"] = np.nan
        beats.loc[valid, f"p_{name}"] = tier.ensemble.adjust_prior(Z[name].values, prior)
    beats["status"] = np.where(~valid, "unclassifiable", np.where(beats["p"] >= 0.5, "abnormal", "normal"))
    beats["referred"] = False
    beats.loc[valid, "referred"] = tier.ensemble.referral_mask(beats.loc[valid, "p"].values, Z)
    return {"beats": beats, "clean": clean, "fs_clean": extractor.fs_target, "n_before": extractor.n_before,
            "waveform_columns": extractor.waveform_columns, "summary": summarize(beats, len(signal) / fs)}


def summarize(beats: pd.DataFrame, duration_s: float) -> dict:
    ok = beats[beats["status"] != "unclassifiable"]
    normal_rr = ok.loc[ok["status"] == "normal", "pre_rr"]
    abnormal = (ok["status"] == "abnormal").astype(int).values
    # longest run of consecutive abnormal beats (2 = couplet, >= 3 = run)
    longest, current = 0, 0
    for a in abnormal:
        current = current + 1 if a else 0
        longest = max(longest, current)
    nn = normal_rr.values * 1000
    return {
        "duration_s": float(duration_s), "beats": int(len(beats)), "classified": int(len(ok)),
        "unclassifiable": int((beats["status"] == "unclassifiable").sum()),
        "abnormal": int(abnormal.sum()), "burden": float(abnormal.mean()) if len(ok) else 0.0,
        "heart_rate": float(60 / np.median(ok["pre_rr"])) if len(ok) else float("nan"),
        "hr_min": float(60 / ok["pre_rr"].max()) if len(ok) else float("nan"),
        "hr_max": float(60 / ok["pre_rr"].min()) if len(ok) else float("nan"),
        "sdnn_ms": float(np.std(nn)) if len(nn) > 2 else float("nan"),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(nn) ** 2))) if len(nn) > 2 else float("nan"),
        "longest_abnormal_run": int(longest), "referred_beats": int(beats["referred"].sum()),
    }


def burden_level(burden: float) -> str:
    """Rough clinical reading of the abnormal-beat burden (premature-beat burden bands)."""
    if burden < 0.01:
        return "low"
    if burden < 0.10:
        return "moderate"
    return "high"
