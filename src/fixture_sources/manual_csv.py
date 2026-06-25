"""Manual CSV source adapter for upcoming fixtures and odds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data_loader import safe_read_csv
from scripts.data_sources import UPCOMING_COLUMNS
from .base import FixtureSourceResult, empty_result, normalize_fixture_source_frame


@dataclass(frozen=True)
class ManualCsvFixtureSource:
    """Load manually maintained or uploaded upcoming fixtures from CSV."""

    path: str | Path
    competition: str | None = None
    name: str = "manual_csv"

    def load(self) -> FixtureSourceResult:
        source_path = Path(self.path)
        if not source_path.exists():
            return empty_result(f"manual fixture CSV not found: {source_path}")
        frame = safe_read_csv(source_path, UPCOMING_COLUMNS)
        fixtures = normalize_fixture_source_frame(frame, competition=self.competition)
        return FixtureSourceResult(fixtures, [f"loaded {len(fixtures):,} fixtures from {source_path}"])
