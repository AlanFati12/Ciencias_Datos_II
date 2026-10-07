"""Transformation of raw ECG signals (unstructured data) into a beat table.

For every annotated heartbeat the extractor produces:
  * 5 rhythm features from the RR intervals (time between consecutive beats),
  * the beat amplitude in millivolts,
  * 4 patient-relative features (comparison with the patient's dominant beat),
  * the beat waveform: 90 samples (0.25 s before to 0.45 s after the R peak)
    at 128 Hz, z-score normalised per beat.

Pipeline per record: read lead 0 -> band-pass 0.5-40 Hz -> resample to
128 Hz -> cut windows around the cardiologist-annotated R peaks.

Equations used (what each one does):
  Band-pass Butterworth filter, |H(f)|^2 = 1 / (1 + (f / f_c)^(2n)) per edge, n = 3
      keeps 0.5-40 Hz: removes the slow baseline wander caused by breathing
      (< 0.5 Hz) and muscle / power-line noise (> 40 Hz). Applied forwards and
      backwards (zero phase) so the waves are not shifted in time.
  Polyphase resampling to 128 Hz (ratio L/M, e.g. 128/360 = 16/45)
      both databases (360 Hz and 128 Hz) and any uploaded ECG end up on the
      same time grid, so a beat always has 90 samples.
  RR_i = (t_i - t_{i-1})                      seconds between consecutive R peaks
      pre_rr = RR_i, post_rr = RR_{i+1},
      local_rr = mean of the previous 10 RR intervals,
      rr_ratio = pre_rr / local_rr  (< 1: the beat came EARLY -> premature beat),
      post_pre_ratio = post_rr / pre_rr  (> 1: compensatory pause after it).
  amplitude = max(w) - min(w)                 peak-to-peak voltage of the beat (mV)
  z-score per beat: w' = (w - mean(w)) / sd(w)
      removes the gain of each device, so only the SHAPE is compared.
  Dominant beat (template) T = median of all z-scored beats of the recording
      template_corr = Pearson r(w', T)        1 = same shape as the usual beat
      template_dist = sqrt(mean((w' - T)^2))  RMS difference to the usual beat
      amplitude_rel = amplitude / median(amplitude),  rr_rel = pre_rr / median(pre_rr)
      These compare each beat with the patient's own usual beat, the way a
      cardiologist does, and need no label (they work on any new ECG).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import config

RR_FEATURES = ["pre_rr", "post_rr", "local_rr", "rr_ratio", "post_pre_ratio"]
AMPLITUDE_FEATURE = "amplitude_mv"
# Patient-relative features: each beat is compared with the patient's own
# dominant beat (median of all beats in the recording, computed WITHOUT labels),
# the way a cardiologist compares an ectopic beat with the usual one.
PATIENT_FEATURES = ["template_corr", "template_dist", "amplitude_rel", "rr_rel"]


class BeatExtractor:
    def __init__(self, fs_target: int = config.ECG_FS_TARGET,
                 window: tuple[float, float] = config.ECG_WINDOW,
                 bandpass: tuple[float, float] = config.ECG_BANDPASS,
                 local_rr_beats: int = config.ECG_LOCAL_RR_BEATS) -> None:
        self.fs_target = fs_target
        self.window = window
        self.bandpass = bandpass
        self.local_rr_beats = local_rr_beats
        self.n_before = round(window[0] * fs_target)
        self.n_after = round(window[1] * fs_target)
        self.symbol_to_class = {s: c for c, symbols in config.AAMI_CLASSES.items() for s in symbols}
        # Every beat counts for the RR intervals, even the ones we do not classify.
        self.rr_symbols = set(self.symbol_to_class) | {"B", "r", "n", "?"}

    @property
    def waveform_columns(self) -> list[str]:
        return [f"w{i:03d}" for i in range(self.n_before + self.n_after)]

    def describe(self) -> dict:
        return {"fs_target_hz": self.fs_target, "window_s": self.window,
                "samples_per_beat": self.n_before + self.n_after,
                "bandpass_hz": self.bandpass, "filter": "Butterworth order 3, zero-phase (sosfiltfilt)",
                "local_rr_beats": self.local_rr_beats, "lead": "channel 0 (MLII in MIT-BIH)",
                "peak_alignment": f"largest |band-passed| deflection within ±{self.ALIGN_WINDOW * 1000:.0f} ms",
                "peak_detection_in_app": "XQRS (wfdb.processing) + the same alignment",
                "normalisation": "z-score per beat", "label": "AAMI EC57: N = normal, S/V/F = abnormal"}

    def _filter(self, signal: np.ndarray, fs: float) -> np.ndarray:
        sos = butter(3, self.bandpass, btype="band", fs=fs, output="sos")
        return sosfiltfilt(sos, signal)

    def _resample(self, signal: np.ndarray, fs: float) -> np.ndarray:
        ratio = Fraction(self.fs_target, int(round(fs))).limit_denominator(1000)
        if ratio == 1:
            return signal
        return resample_poly(signal, ratio.numerator, ratio.denominator)

    def prepare(self, raw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (band-passed signal at the original rate, clean signal at fs_target)."""
        filtered = self._filter(np.nan_to_num(raw), fs)
        return filtered, self._resample(filtered, fs)

    def align_peaks(self, filtered: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
        """Move every peak to the largest absolute deflection within ±75 ms.

        Annotated and automatically detected beats then share the same fiducial
        point, so the model sees identical windows in training and in the app.
        """
        half = int(round(self.ALIGN_WINDOW * fs))
        out = np.empty(len(peaks), dtype=int)
        for i, p in enumerate(peaks):
            lo = max(0, int(p) - half)
            out[i] = lo + int(np.argmax(np.abs(filtered[lo:int(p) + half + 1])))
        return out

    ALIGN_WINDOW = 0.075

    def detect_peaks(self, raw: np.ndarray, fs: float) -> np.ndarray:
        """R peaks of an ECG without annotations (XQRS detector + alignment)."""
        from wfdb import processing
        xqrs = processing.XQRS(sig=np.nan_to_num(raw), fs=fs)
        xqrs.detect(verbose=False)
        filtered, _ = self.prepare(raw, fs)
        return np.unique(self.align_peaks(filtered, np.asarray(xqrs.qrs_inds), fs))

    def preprocess_signal(self, record_path: Path) -> tuple[np.ndarray, np.ndarray, float, wfdb.Annotation]:
        """Return (raw signal, clean resampled signal, original fs, annotations)."""
        record = wfdb.rdrecord(str(record_path), channels=[0])
        annotation = wfdb.rdann(str(record_path), "atr")
        raw = np.nan_to_num(record.p_signal[:, 0])
        _, clean = self.prepare(raw, record.fs)
        return raw, clean, record.fs, annotation

    def extract_record(self, record_path: Path, database: str) -> pd.DataFrame:
        """Training path: beats located with the cardiologists' annotations."""
        raw, signal, fs, ann = self.preprocess_signal(record_path)
        symbols = np.asarray(ann.symbol)
        is_beat = np.isin(symbols, list(self.rr_symbols))
        filtered, _ = self.prepare(raw, fs)
        peaks = self.align_peaks(filtered, np.asarray(ann.sample)[is_beat], fs)
        beats = self.beats_from_peaks(signal, peaks, fs, symbols[is_beat])
        beats.insert(0, "record", f"{database}_{record_path.name}")
        beats.insert(0, "database", database)
        return beats

    def extract_signal(self, raw: np.ndarray, fs: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Deployment path: any single-lead ECG -> (beat table, clean signal, peaks)."""
        _, clean = self.prepare(raw, fs)
        peaks = self.detect_peaks(raw, fs)
        return self.beats_from_peaks(clean, peaks, fs), clean, peaks

    def beats_from_peaks(self, signal: np.ndarray, peaks: np.ndarray, fs: float,
                         symbols: np.ndarray | None = None) -> pd.DataFrame:
        """Feature table for the beats at `peaks` (original-rate samples).

        `signal` is the clean signal at fs_target. With `symbols` (training) only
        AAMI N/S/V/F beats are kept and labelled; without them every beat is kept.
        """
        peaks = np.asarray(peaks)
        centers = np.round(peaks * self.fs_target / fs).astype(int)

        rr = np.diff(centers) / self.fs_target              # seconds between beats
        # beat i needs a previous and a next beat (rr[i-1] and rr[i])
        idx = np.arange(1, len(centers) - 1)
        pre_rr, post_rr = rr[idx - 1], rr[idx]
        csum = np.concatenate([[0.0], np.cumsum(rr)])
        start = np.maximum(idx - self.local_rr_beats, 0)
        local_rr = (csum[idx] - csum[start]) / (idx - start)

        fits = (centers[idx] - self.n_before >= 0) & (centers[idx] + self.n_after <= len(signal))
        if symbols is not None:
            aami = np.array([self.symbol_to_class.get(s, "") for s in np.asarray(symbols)[idx]])
            keep = np.isin(aami, ["N", "S", "V", "F"]) & fits
            aami = aami[keep]
        else:
            keep = fits
        idx, pre_rr, post_rr, local_rr = idx[keep], pre_rr[keep], post_rr[keep], local_rr[keep]

        offsets = np.arange(-self.n_before, self.n_after)
        windows = signal[centers[idx][:, None] + offsets[None, :]]
        amplitude = windows.max(axis=1) - windows.min(axis=1)
        std = windows.std(axis=1, keepdims=True)
        windows = (windows - windows.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1.0)

        # dominant beat of this patient (unsupervised: no label is used)
        template = np.median(windows, axis=0)
        t_std = template.std() or 1.0
        template_corr = (windows * (template - template.mean())).mean(axis=1) / t_std
        template_dist = np.sqrt(((windows - template) ** 2).mean(axis=1))

        meta = pd.DataFrame({"sample": peaks[idx]})
        if symbols is not None:
            meta["symbol"] = np.asarray(symbols)[idx]
            meta["aami_class"] = aami
        meta = meta.assign(
            pre_rr=pre_rr, post_rr=post_rr, local_rr=local_rr,
            rr_ratio=pre_rr / local_rr, post_pre_ratio=post_rr / pre_rr,
            **{AMPLITUDE_FEATURE: amplitude},
            template_corr=template_corr, template_dist=template_dist,
            amplitude_rel=amplitude / np.median(amplitude), rr_rel=pre_rr / np.median(pre_rr))
        if symbols is not None:
            meta[config.ECG_TARGET] = (aami != "N").astype(int)
        waves = pd.DataFrame(windows.astype(np.float32), columns=self.waveform_columns)
        return pd.concat([meta, waves], axis=1)
