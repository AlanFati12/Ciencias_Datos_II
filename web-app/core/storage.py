"""Assessment history in a local SQLite database (web-app/data/history.db)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd

from .model_store import APP_ROOT

DB_PATH = APP_ROOT / "data" / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,              -- 'clinical' or 'ecg'
    patient_code TEXT,
    probability REAL,                -- clinical: P(CVD); ecg: abnormal-beat burden
    result TEXT,                     -- short label
    risk_segment TEXT,
    referred INTEGER,
    inputs_json TEXT,
    details_json TEXT
)
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(SCHEMA)
    return con


def save(kind: str, patient_code: str, probability: float, result: str, inputs: dict, details: dict,
         risk_segment: str | None = None, referred: bool = False) -> int:
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO assessments (created_at, kind, patient_code, probability, result, risk_segment, referred,"
            " inputs_json, details_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), kind, patient_code.strip() or None, float(probability),
             result, risk_segment, int(referred), json.dumps(inputs, default=float),
             json.dumps(details, default=float)))
        return int(cur.lastrowid)


def load_all() -> pd.DataFrame:
    with _connect() as con:
        return pd.read_sql_query("SELECT * FROM assessments ORDER BY id DESC", con)


def delete(ids: list[int]) -> None:
    with _connect() as con:
        con.executemany("DELETE FROM assessments WHERE id = ?", [(int(i),) for i in ids])
