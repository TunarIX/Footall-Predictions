"""Shared international historical and fixtures handling tests."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from scripts.update_international_data import update_international_data
from src.fixtures import _filter_competition
from src.predictor import train_baseline_model
from src.preprocessing import clean_international_match_data


def test_fifa_world_cup_filters_international_historical_rows() -> None:
    raw = pd.DataFrame(
        [
            {"date": "2022-11-20", "home_team": "Qatar", "away_team": "Ecuador", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup"},
            {"date": "2022-09-22", "home_team": "France", "away_team": "Austria", "home_score": 2, "away_score": 0, "tournament": "UEFA Nations League"},
        ]
    )
    cleaned = clean_international_match_data(raw)
    filtered = _filter_competition(cleaned, "FIFA World Cup")
    assert len(filtered) == 1
    assert filtered.iloc[0]["Competition"] == "FIFA World Cup"


def test_fifa_world_cup_filters_international_upcoming_fixtures() -> None:
    fixtures = normalize_upcoming_frame(
        pd.DataFrame(
            [
                {"Date": "2030-06-13", "Time": "20:00", "Competition": "FIFA World Cup", "HomeTeam": "Spain", "AwayTeam": "Brazil"},
                {"Date": "2030-06-14", "Time": "18:00", "Competition": "International Friendly", "HomeTeam": "Germany", "AwayTeam": "Japan"},
            ]
        )
    )
    filtered = _filter_competition(fixtures, "FIFA World Cup")
    assert len(filtered) == 1
    assert filtered.iloc[0]["HomeTeam"] == "Spain"


def test_training_does_not_require_odds() -> None:
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(40):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        hg = i % 3
        ag = (i + 1) % 3
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_score": hg,
                "away_score": ag,
                "tournament": "FIFA World Cup" if i % 5 == 0 else "International Friendly",
            }
        )
    cleaned = clean_international_match_data(pd.DataFrame(rows))
    model, training = train_baseline_model(cleaned)
    assert not training.empty
    assert {"ImpHome", "ImpDraw", "ImpAway"}.issubset(training.columns)


def test_manual_raw_international_csv_is_processed_into_non_empty_file(tmp_path: Path) -> None:
    raw = tmp_path / "international_matches.csv"
    output = tmp_path / "processed_international_matches.csv"
    raw.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,neutral,country\n"
        "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,True,Qatar\n"
        "2022-09-22,France,Austria,2,0,UEFA Nations League,False,France\n"
    )

    processed = update_international_data(raw, output)

    assert output.exists()
    assert len(processed) == 2
    assert {
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
    }.issubset(processed.columns)
    assert set(processed["FTR"]) == {"A", "H"}


def test_empty_input_does_not_overwrite_valid_processed_file(tmp_path: Path) -> None:
    raw = tmp_path / "international_matches.csv"
    output = tmp_path / "processed_international_matches.csv"
    valid_contents = (
        "Date,Competition,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Neutral,Country,SourceFile\n"
        "2022-11-20,FIFA World Cup,Qatar,Ecuador,0,2,A,True,Qatar,seed.csv\n"
    )
    output.write_text(valid_contents)
    raw.write_text("date,home_team,away_team,home_score,away_score,tournament,neutral\n")

    try:
        update_international_data(raw, output)
    except SystemExit as exc:  # pragma: no cover - defensive for CLI-like failures
        assert "validation" in str(exc).lower() or "invalid" in str(exc).lower()
    except ValueError as exc:
        assert "rows > 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty input should fail validation")

    assert output.read_text() == valid_contents
