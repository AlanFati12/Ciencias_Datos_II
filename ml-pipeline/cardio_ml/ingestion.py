"""Extraction step of the ETL: each ingester only reads data (SRP)."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from . import config
from .ecg_etl import BeatExtractor
from .tracking import RunTracker


class DataIngester(ABC):
    """Contract shared by every data source of the system."""

    def __init__(self, tracker: RunTracker) -> None:
        self.tracker = tracker

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the raw data as a DataFrame."""


class ClinicalCSVIngester(DataIngester):
    """Reads the Kaggle Cardiovascular Disease dataset (Ulianova, 2019)."""

    def __init__(self, tracker: RunTracker, path: Path = config.CLINICAL_CSV) -> None:
        super().__init__(tracker)
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep=";")
        self.tracker.log_input(self.path, "Kaggle Cardiovascular Disease dataset (S. Ulianova, 2019)")
        self.tracker.log_step("Load raw CSV", len(df), len(df), f"{df.shape[1]} columns, separator ';'")
        return df


class PhysioNetECGIngester(DataIngester):
    """Downloads (once) and segments the PhysioNet arrhythmia databases.

    Each record is a long ECG signal; the BeatExtractor turns it into one row
    per heartbeat, so the output is a tabular dataset like the clinical one.
    """

    def __init__(self, tracker: RunTracker, databases: dict = config.PHYSIONET_DATABASES,
                 extractor: BeatExtractor | None = None) -> None:
        super().__init__(tracker)
        self.databases = databases
        self.extractor = extractor or BeatExtractor()

    def _ensure_database(self, name: str) -> Path:
        url, folder = self.databases[name]
        target = config.ECG_DIR / folder
        if target.exists():
            return target
        config.ECG_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = config.ECG_DIR / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {name} from {url} (this can take several minutes)...")
            with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out:
                shutil.copyfileobj(response, out)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(config.ECG_DIR)
        return target

    def load(self) -> pd.DataFrame:
        frames = []
        for name in self.databases:
            folder = self._ensure_database(name)
            zip_path = config.ECG_DIR / f"{name}.zip"
            if zip_path.exists():
                self.tracker.log_input(zip_path, f"PhysioNet {name} v1.0.0 (ODC-By 1.0)")
            records = sorted(p.stem for p in folder.glob("*.hea"))
            excluded = config.ECG_EXCLUDED_RECORDS.get(name, set())
            kept = [r for r in records if r not in excluded]
            beats = pd.concat([self.extractor.extract_record(folder / r, name) for r in kept],
                              ignore_index=True)
            self.tracker.log_step(f"Segment beats ({name})", 0, len(beats),
                                  f"{len(kept)} records read, {len(excluded)} paced records excluded "
                                  f"({', '.join(sorted(excluded)) or 'none'})")
            frames.append(beats)
        df = pd.concat(frames, ignore_index=True)
        self.tracker.log_param("ecg_extraction", self.extractor.describe())
        return df
