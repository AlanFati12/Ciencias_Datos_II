"""Reading uploaded files.

The uploaders do not restrict file types in the browser: on some systems the
browser's type filter greys out perfectly valid .csv files (depending on the
program the operating system associates with them). The type is checked here
instead, after the file is chosen, with a clear message.
"""
from __future__ import annotations

import csv
import io
from pathlib import PurePath

import pandas as pd


def extension(name: str) -> str:
    return PurePath(name).suffix.lower()


def is_table(name: str) -> bool:
    return extension(name) in {".csv", ".txt", ".tsv"}


def read_table(data: bytes) -> pd.DataFrame:
    """Comma, semicolon, tab or space separated text, with or without a header row.

    Spanish Excel exports (';' separator and decimal comma) are also understood.
    """
    text = data.decode("utf-8-sig", errors="replace")
    sample = text[:20000]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
    except csv.Error:                                  # a single column has no separator
        sep = ","
    first_line = sample.splitlines()[0] if sample.strip() else ""
    has_header = any(ch.isalpha() for ch in first_line)
    decimal = "," if sep == ";" and "," in sample else "."
    df = pd.read_csv(io.StringIO(text), sep=sep, header=0 if has_header else None, decimal=decimal,
                     skipinitialspace=True)
    if not has_header:
        df.columns = [f"columna_{i + 1}" for i in range(df.shape[1])]
    return df
