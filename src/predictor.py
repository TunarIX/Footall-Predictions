"""Multi-feature baseline football prediction model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_match_features, upcoming_features

FEATURE_COLUMNS = [
    "HomePPG5",
    "HomePPG10",
    "AwayPPG5",
    "AwayPPG10",
    "HomeGF5",
    "HomeGA5",
    "HomeGF10",
    "HomeGA10",
    "AwayGF5",
    "AwayGA5",
    "AwayGF10",
    "AwayGA10",
    "HomeVenueWinRate",
    "AwayVenueWinRate",
    "HomeH2HPPG",
    "H2HMatches",
    "HomeElo",
    "AwayElo",
    "EloDiff",
    "ImpHome",
    "ImpDraw",
    "ImpAway",
]
CLASS_LABELS = ["H", "D", "A"]
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def train_baseline_model(df: pd.DataFrame) -> tuple[Pipeline | None, pd.DataFrame]:
    """Train the baseline model on the multi-feature match dataset."""
    dataset = build_match_features(df)
    if len(dataset) < 30 or dataset["FTR"].nunique() < 2:
        return None, dataset

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=4,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    model.fit(dataset[FEATURE_COLUMNS], dataset["FTR"])
    return model, dataset


def _score_estimate(features: pd.DataFrame) -> tuple[float, float]:
    """Estimate expected home and away goals from recent scoring/concession trends."""
    row = features.iloc[0]
    home_goals = max(
        0.1,
        0.55 * row["HomeGF5"] + 0.25 * row["HomeGF10"] + 0.20 * row["AwayGA5"] + 0.15 * row["ImpHome"],
    )
    away_goals = max(
        0.1,
        0.55 * row["AwayGF5"] + 0.25 * row["AwayGF10"] + 0.20 * row["HomeGA5"] + 0.15 * row["ImpAway"],
    )
    return round(float(home_goals), 2), round(float(away_goals), 2)


def prediction_explanation(features: pd.DataFrame, home_team: str, away_team: str) -> list[str]:
    """Build human-readable notes from the same feature row used by the model."""
    row = features.iloc[0]
    return [
        f"Recent form: {home_team} last-5 PPG {row['HomePPG5']:.2f} vs {away_team} {row['AwayPPG5']:.2f}; last-10 PPG {row['HomePPG10']:.2f} vs {row['AwayPPG10']:.2f}.",
        f"Goal trends: {home_team} GF/GA last 5 is {row['HomeGF5']:.2f}/{row['HomeGA5']:.2f}; {away_team} GF/GA last 5 is {row['AwayGF5']:.2f}/{row['AwayGA5']:.2f}.",
        f"Venue profile: home win rate {row['HomeVenueWinRate']:.1%}; away win rate {row['AwayVenueWinRate']:.1%}.",
        f"Head-to-head context: {row['H2HMatches']:.0f} recent meetings, {home_team} averaged {row['HomeH2HPPG']:.2f} points.",
        f"Elo: {home_team} {row['HomeElo']:.0f}, {away_team} {row['AwayElo']:.0f}, home-adjusted difference {row['EloDiff']:.0f}.",
        f"Market context: normalized implied probabilities H/D/A {row['ImpHome']:.1%}/{row['ImpDraw']:.1%}/{row['ImpAway']:.1%}.",
    ]


def predict_match(
    model: Pipeline,
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    implied_probs: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return prediction table, feature row, and explanation notes for an upcoming match."""
    features = upcoming_features(df, home_team, away_team, implied_probs)
    probabilities = dict.fromkeys(CLASS_LABELS, 0.0)

    for klass, probability in zip(model.classes_, model.predict_proba(features[FEATURE_COLUMNS])[0], strict=False):
        probabilities[klass] = float(probability)

    home_xg, away_xg = _score_estimate(features)
    confidence = max(probabilities.values()) - np.median(list(probabilities.values()))
    table = pd.DataFrame(
        [
            {
                "Outcome": OUTCOME_LABELS[label],
                "Model probability": probabilities[label],
                "Bookmaker implied": implied_probs[index],
                "Value signal": probabilities[label] > implied_probs[index],
                "Confidence contribution": confidence if probabilities[label] == max(probabilities.values()) else 0.0,
            }
            for index, label in enumerate(CLASS_LABELS)
        ]
    )
    table["Estimated score"] = f"{home_team} {home_xg:.2f} - {away_xg:.2f} {away_team}"
    table["Confidence score"] = float(np.clip(0.5 + confidence, 0, 1))
    return table, features, prediction_explanation(features, home_team, away_team)


def similar_historical_matches(training_data: pd.DataFrame, features: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Find historical matches with similar engineered features, not only similar odds."""
    if training_data.empty:
        return pd.DataFrame()

    weights = np.array(
        [1.3 if column in {"HomePPG5", "AwayPPG5", "EloDiff", "ImpHome", "ImpDraw", "ImpAway"} else 1.0 for column in FEATURE_COLUMNS]
    )
    hist = training_data.dropna(subset=FEATURE_COLUMNS).copy()
    scale = hist[FEATURE_COLUMNS].std().replace(0, 1)
    target = features.iloc[0][FEATURE_COLUMNS]
    hist["SimilarityDistance"] = ((((hist[FEATURE_COLUMNS] - target) / scale) * weights) ** 2).sum(axis=1) ** 0.5
    return hist.sort_values("SimilarityDistance").head(limit)
