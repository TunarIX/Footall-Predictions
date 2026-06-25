#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Virtual environment not found. Running local setup first..."
  bash scripts/setup_local.sh
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

exec streamlit run app.py
