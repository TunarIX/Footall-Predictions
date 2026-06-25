"""Build the shared international historical matches CSV.

The updater prefers a manual local CSV at ``data/raw/international_matches.csv``.
A future automatic provider can be wired into ``load_source``; until then the
script fails clearly instead of writing an empty processed file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_international_match_data, parse_dates

RAW_INTERNATIONAL = Path("data/raw/international_matches.csv")
PROCESSED_INTERNATIONAL = Path("data/processed/international_matches.csv")
INTERNATIONAL_COLUMNS = [
    "Date",
    "Competition",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "Neutral",
    "Country",
    "SourceFile",
]
NO_SOURCE_MESSAGE = (
    "Automatic international historical data source is not configured. "
    "Add a CSV to data/raw/international_matches.csv or configure an API/source."
)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding_errors="ignore")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read international source CSV {path}: {exc}") from exc


def load_source(source_csv: Path = RAW_INTERNATIONAL) -> pd.DataFrame:
    """Load national-team match rows from the configured source."""
    if source_csv.exists():
        frame = _read_csv(source_csv)
        frame["SourceFile"] = source_csv.name
        return frame
    raise FileNotFoundError(NO_SOURCE_MESSAGE)


def normalize_international_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize common national-team CSV schemas to the processed contract."""
    cleaned = clean_international_match_data(raw)
    normalized = cleaned.copy()
    if "Competition" not in normalized.columns:
        normalized["Competition"] = pd.NA
    if "Neutral" not in normalized.columns:
        normalized["Neutral"] = pd.NA
    if "Country" not in normalized.columns:
        normalized["Country"] = pd.NA
    if "SourceFile" not in normalized.columns:
        normalized["SourceFile"] = pd.NA
    normalized = normalized[INTERNATIONAL_COLUMNS].copy()
    normalized["Date"] = parse_dates(normalized["Date"])
    normalized = normalized.drop_duplicates(
        ["Date", "Competition", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last"
    )
    return normalized.sort_values("Date").reset_index(drop=True)


def validate_international_matches(data: pd.DataFrame) -> None:
    """Raise a clear error if processed international data is unusable."""
    missing = [column for column in INTERNATIONAL_COLUMNS if column not in data.columns]
    errors: list[str] = []
    if data.empty:
        errors.append("rows > 0 validation failed")
    if missing:
        errors.append(f"missing required columns: {', '.join(missing)}")
    if "Date" in data.columns and parse_dates(data["Date"]).isna().any():
        errors.append("Date is parsed validation failed")
    if "FTR" in data.columns:
        invalid = sorted(set(data["FTR"].dropna().astype(str).str.upper()) - {"H", "D", "A"})
        if invalid or data["FTR"].isna().any():
            errors.append("FTR contains H/D/A validation failed")
    if errors:
        raise ValueError("Invalid international historical data: " + "; ".join(errors))


def update_international_data(
    source_csv: Path = RAW_INTERNATIONAL,
    output: Path = PROCESSED_INTERNATIONAL,
) -> pd.DataFrame:
    """Process international data and write it only after validation succeeds."""
    raw = load_source(source_csv)
    normalized = normalize_international_matches(raw)
    validate_international_matches(normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    print(f"wrote {len(normalized):,} international rows to {output}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=RAW_INTERNATIONAL)
    parser.add_argument("--output", type=Path, default=PROCESSED_INTERNATIONAL)
    args = parser.parse_args()
    try:
        update_international_data(args.source_csv, args.output)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise failure
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
