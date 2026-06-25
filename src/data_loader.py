"""CSV loading helpers for the Streamlit app."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .preprocessing import EXPECTED_COLUMNS, clean_international_match_data, clean_match_data


def safe_read_csv(
    file: str | Path | Any,
    expected_columns: Iterable[str] | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Safely load a CSV and always return a dataframe with expected columns.

    Missing paths, empty/headerless files, parser errors, and incomplete schemas
    return or are padded to an empty dataframe with the requested columns.
    """
    columns = list(expected_columns or [])
    try:
        if isinstance(file, (str, Path)) and not Path(file).exists():
            return pd.DataFrame(columns=columns)
        frame = pd.read_csv(file, encoding_errors="ignore", **read_csv_kwargs)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return pd.DataFrame(columns=columns)

    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def load_uploaded_files(
    files: Iterable, data_source: str = "football-data.co.uk"
) -> pd.DataFrame:
    """Read and combine uploaded CSV files for the selected data source."""
    frames: list[pd.DataFrame] = []
    for file in files:
        raw = safe_read_csv(file, EXPECTED_COLUMNS)
        raw["SourceFile"] = getattr(file, "name", "uploaded_csv")
        frames.append(raw)
    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    if data_source == "international_csv":
        return clean_international_match_data(combined)
    return clean_match_data(combined)
