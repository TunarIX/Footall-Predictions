"""Source adapter interfaces for upcoming fixture imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame


@dataclass(frozen=True)
class FixtureSourceResult:
    """Normalized fixture source output plus human-readable status messages."""

    fixtures: pd.DataFrame
    messages: list[str]


class FixtureSource(Protocol):
    """Protocol implemented by upcoming-fixture source adapters."""

    name: str

    def load(self) -> FixtureSourceResult:
        """Return normalized upcoming fixtures. Missing odds are allowed."""


def empty_result(message: str) -> FixtureSourceResult:
    return FixtureSourceResult(pd.DataFrame(columns=UPCOMING_COLUMNS), [message])


def normalize_fixture_source_frame(frame: pd.DataFrame, competition: str | None = None) -> pd.DataFrame:
    """Normalize adapter output and keep the canonical upcoming-fixture columns."""

    return normalize_upcoming_frame(frame, competition=competition)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
