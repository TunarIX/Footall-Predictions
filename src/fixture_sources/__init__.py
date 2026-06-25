"""Upcoming fixture source adapters."""

from .base import FixtureSourceResult
from .manual_csv import ManualCsvFixtureSource
from .worldcup_static import WORLD_CUP_2026_FIXTURES, WorldCupStaticFixtureSource, ensure_worldcup_template

__all__ = [
    "FixtureSourceResult",
    "ManualCsvFixtureSource",
    "WORLD_CUP_2026_FIXTURES",
    "WorldCupStaticFixtureSource",
    "ensure_worldcup_template",
]
