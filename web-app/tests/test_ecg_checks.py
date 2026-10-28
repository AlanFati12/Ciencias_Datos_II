"""The ECG analysis accepts real recordings and rejects signals that would mislead the models (core/ecg.py)."""
import numpy as np
import pytest

from core import ecg


@pytest.fixture(scope="module")
def demos():
    """{file name: (signal in mV, sampling rate)} of the real PhysioNet examples shipped with the app."""
    return {d["file"]: ecg.load_demo(d) for d in ecg.demo_catalog()}


def test_real_recordings_are_analysed(ecg_tier, demos):
    for name, (signal, fs) in demos.items():
        result = ecg.analyze(ecg_tier, signal, fs)
        assert result["quality"]["is_ecg"], name
        assert result["summary"]["classified"] > 100, name


def test_normal_rhythm_has_no_abnormal_beats(ecg_tier, demos):
    signal, fs = demos["mitdb_100_10s.csv"]
    assert ecg.analyze(ecg_tier, signal, fs)["summary"]["burden"] == 0


def test_ventricular_bigeminy_is_detected(ecg_tier, demos):
    signal, fs = demos["mitdb_119_1360s.csv"]
    assert ecg.analyze(ecg_tier, signal, fs)["summary"]["burden"] > 0.25


@pytest.mark.parametrize("name, make", [
    ("white noise", lambda rng, t: rng.normal(0, 1, len(t))),
    ("weather-like readings", lambda rng, t: 24 + 6 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 0.4, len(t))),
    ("square wave", lambda rng, t: np.sign(np.sin(2 * np.pi * t))),
    ("random walk", lambda rng, t: np.cumsum(rng.normal(0, 1, len(t)))),
])
def test_signals_that_are_not_an_ecg_are_rejected(ecg_tier, name, make):
    t = np.arange(40 * 360) / 360
    with pytest.raises(ValueError, match="not_ecg"):
        ecg.analyze(ecg_tier, make(np.random.default_rng(0), t), 360)


def test_too_short_signal(ecg_tier, demos):
    signal, fs = demos["mitdb_100_10s.csv"]
    with pytest.raises(ValueError, match="too_short"):
        ecg.analyze(ecg_tier, signal[: int(10 * fs)], fs)


def test_microvolts_read_as_millivolts(ecg_tier, demos):
    signal, fs = demos["mitdb_100_10s.csv"]
    with pytest.raises(ValueError, match="units_uv"):
        ecg.analyze(ecg_tier, signal * 1000, fs)


def test_millivolts_read_as_microvolts(ecg_tier, demos):
    signal, fs = demos["mitdb_100_10s.csv"]
    with pytest.raises(ValueError, match="units_small"):
        ecg.analyze(ecg_tier, signal / 1000, fs)


@pytest.mark.parametrize("demo, wrong_fs, reason", [
    ("mitdb_100_10s.csv", 128, "heart_rate"),     # 360 Hz read as 128 Hz: about 26 bpm
    ("mitdb_100_10s.csv", 100, "heart_rate"),
    ("svdb_866_1280s.csv", 500, "not_ecg"),       # 128 Hz read as 500 Hz: the filter destroys the beat shapes
])
def test_wrong_samples_per_second(ecg_tier, demos, demo, wrong_fs, reason):
    signal, _ = demos[demo]
    with pytest.raises(ValueError, match=reason):
        ecg.analyze(ecg_tier, signal, wrong_fs)


def test_a_few_empty_values_are_filled_but_many_are_rejected(ecg_tier, demos):
    signal, fs = demos["mitdb_100_10s.csv"]
    few = signal.copy()
    few[1000:1010] = np.nan
    assert ecg.analyze(ecg_tier, few, fs)["summary"]["burden"] == 0
    many = signal.copy()
    many[::20] = np.nan                                  # 5 % empty
    with pytest.raises(ValueError, match="missing"):
        ecg.analyze(ecg_tier, many, fs)
