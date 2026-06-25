"""Runtime safety smoke tests for empty/missing local CSV files."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from scripts.predict_next_48h import PREDICTION_COLUMNS
from scripts.update_upcoming_fixtures import update_upcoming_fixtures
from src.data_loader import safe_read_csv
from src.preprocessing import EXPECTED_COLUMNS


def test_safe_read_csv_missing_and_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = safe_read_csv(tmp_path / "missing.csv", ["A", "B"])
        assert list(missing.columns) == ["A", "B"]
        assert missing.empty

        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")
        empty = safe_read_csv(empty_path, ["A", "B"])
        assert list(empty.columns) == ["A", "B"]
        assert empty.empty

        incomplete_path = tmp_path / "incomplete.csv"
        incomplete_path.write_text("A\n1\n")
        incomplete = safe_read_csv(incomplete_path, ["A", "B"])
        assert list(incomplete.columns) == ["A", "B"]
        assert pd.isna(incomplete.loc[0, "B"])


def test_empty_prediction_csv_has_safe_headers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prediction_path = Path(tmp) / "next_48h_predictions.csv"
        prediction_path.write_text("")
        predictions = safe_read_csv(prediction_path, PREDICTION_COLUMNS)
        assert predictions.empty
        assert list(predictions.columns) == PREDICTION_COLUMNS


def test_missing_and_empty_upcoming_fixtures_are_safe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing_upcoming = normalize_upcoming_frame(safe_read_csv(tmp_path / "missing.csv", UPCOMING_COLUMNS))
        assert missing_upcoming.empty
        assert list(missing_upcoming.columns) == UPCOMING_COLUMNS

        empty_upcoming_path = tmp_path / "empty_upcoming.csv"
        empty_upcoming_path.write_text("")
        empty_upcoming = normalize_upcoming_frame(safe_read_csv(empty_upcoming_path, UPCOMING_COLUMNS))
        assert empty_upcoming.empty
        assert list(empty_upcoming.columns) == UPCOMING_COLUMNS


def test_update_upcoming_writes_headers_without_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "upcoming_fixtures.csv"
        result = update_upcoming_fixtures(output=output)
        assert output.exists()
        loaded = safe_read_csv(output, UPCOMING_COLUMNS)
        assert list(loaded.columns) == UPCOMING_COLUMNS
        assert result.empty or list(result.columns) == UPCOMING_COLUMNS


def test_missing_historical_csv_is_safe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        historical = safe_read_csv(Path(tmp) / "historical_matches.csv", EXPECTED_COLUMNS, parse_dates=["Date"])
        assert historical.empty
        assert list(historical.columns) == EXPECTED_COLUMNS


if __name__ == "__main__":
    test_safe_read_csv_missing_and_empty()
    test_empty_prediction_csv_has_safe_headers()
    test_missing_and_empty_upcoming_fixtures_are_safe()
    test_update_upcoming_writes_headers_without_data()
    test_missing_historical_csv_is_safe()
