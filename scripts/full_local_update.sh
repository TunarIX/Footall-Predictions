#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
START_YEAR="${START_YEAR:-2018}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Virtual environment not found. Running local setup first..."
  bash scripts/setup_local.sh
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python scripts/update_historical_data.py --start-year "$START_YEAR"
python scripts/update_upcoming_fixtures.py "$@"
python scripts/predict_next_48h.py

echo "Local data update complete. Predictions: data/predictions/next_48h_predictions.csv"
