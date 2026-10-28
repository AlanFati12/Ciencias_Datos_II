"""Beat detection and rhythm features on a synthetic ECG with one premature beat (ecg_etl.py)."""
import numpy as np
import pytest

from cardio_ml.ecg_etl import BeatExtractor

FS = 360


def synthetic_ecg(beat_times: np.ndarray, seconds: float = 40.0, seed: int = 0) -> np.ndarray:
    """P wave, QRS complex and T wave as Gaussian bumps around every beat time, plus a little noise."""
    t = np.arange(int(seconds * FS)) / FS
    x = np.random.default_rng(seed).normal(0, 0.01, len(t))
    for b in beat_times:
        x += 0.15 * np.exp(-((t - b + 0.16) / 0.025) ** 2)      # P wave
        x += 1.50 * np.exp(-((t - b) / 0.012) ** 2)             # QRS (R peak)
        x += 0.30 * np.exp(-((t - b - 0.25) / 0.040) ** 2)      # T wave
    return x


@pytest.fixture(scope="module")
def beats():
    times = list(np.arange(1.0, 20.0, 0.8))                     # regular rhythm, 75 bpm
    times += [times[-1] + 0.45, times[-1] + 0.45 + 1.15]          # premature beat + compensatory pause
    times += list(np.arange(times[-1] + 0.8, 39.0, 0.8))
    table, _, _ = BeatExtractor().extract_signal(synthetic_ecg(np.array(times)), FS)
    table["t"] = table["sample"] / FS
    return table, np.array(times)


def test_every_beat_is_found(beats):
    table, times = beats
    assert abs(len(table) - (len(times) - 2)) <= 2              # first and last beat have no neighbours


def test_regular_rhythm_has_rr_close_to_0_8_seconds(beats):
    table, _ = beats
    assert table["pre_rr"].median() == pytest.approx(0.8, abs=0.02)
    assert table["template_corr"].median() > 0.9                # all beats look like the usual beat


def test_premature_beat_arrives_early_and_is_followed_by_a_pause(beats):
    table, times = beats
    premature_time = times[24]                                  # the beat 0.45 s after the previous one
    row = table.iloc[(table["t"] - premature_time).abs().argmin()]
    assert row["rr_ratio"] < 0.7                                # arrives early
    assert row["post_pre_ratio"] > 2.0                          # long pause after it
