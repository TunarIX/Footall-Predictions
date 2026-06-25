from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from scripts.update_international_fixtures import update_international_fixtures
from src.fixture_sources.manual_csv import ManualCsvFixtureSource
from src.fixture_sources.worldcup_static import WorldCupStaticFixtureSource
from src.odds import odds_to_probabilities


def test_worldcup_fixture_source_creates_and_loads_template(tmp_path: Path) -> None:
    path = tmp_path / "worldcup_2026_fixtures.csv"

    result = WorldCupStaticFixtureSource(path=path).load()

    assert path.exists()
    assert list(pd.read_csv(path).columns) == [
        "Date",
        "Time",
        "Competition",
        "HomeTeam",
        "AwayTeam",
        "HomeOdds",
        "DrawOdds",
        "AwayOdds",
        "Over25Odds",
        "Under25Odds",
        "OddsSource",
    ]
    assert not result.fixtures.empty
    assert result.fixtures.iloc[0]["Competition"] == "FIFA World Cup"


def test_manual_csv_source_loads_missing_and_present_odds(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    path.write_text(
        "Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,Over25Odds,Under25Odds,OddsSource\n"
        "2026-07-01,20:00,FIFA World Cup,Spain,Brazil,,,,,,Manual no odds\n"
        "2026-07-02,20:00,FIFA World Cup,France,Germany,2.1,3.2,3.4,1.9,1.8,Manual odds\n"
    )

    fixtures = ManualCsvFixtureSource(path).load().fixtures

    assert len(fixtures) == 2
    assert fixtures.loc[fixtures["HomeTeam"] == "Spain", "HomeOdds"].isna().all()
    assert fixtures.loc[fixtures["HomeTeam"] == "France", "HomeOdds"].iloc[0] == 2.1


def test_update_international_fixtures_writes_normalized_manual_csv(tmp_path: Path) -> None:
    source = tmp_path / "worldcup.csv"
    output = tmp_path / "international_fixtures.csv"
    source.write_text(
        "Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,OddsSource\n"
        "2026-07-03,18:00,FIFA World Cup,Argentina,England,,,,Unavailable\n"
    )

    written = update_international_fixtures(source_csv=str(source), output=output)

    assert output.exists()
    saved = pd.read_csv(output)
    assert len(written) == 1
    assert saved.iloc[0]["HomeTeam"] == "Argentina"
    assert "Over25Odds" in saved.columns


def test_predictions_probability_inputs_work_without_and_with_odds() -> None:
    assert all(pd.isna(value) for value in odds_to_probabilities(float("nan"), float("nan"), float("nan")))

    home, draw, away = odds_to_probabilities(2.0, 3.5, 4.0)

    assert home + draw + away == pytest.approx(1.0, abs=0.001)
    assert home > draw > away
