#!/bin/bash

set -e

PROJECT_DIR="$HOME/Desktop/ADAPT_DEPLOY"

cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
  echo "ERROR: virtual environment not found at $PROJECT_DIR/venv"
  echo "Create it first with:"
  echo "  python3 -m venv venv"
  exit 1
fi

source venv/bin/activate

echo "============================================================"
echo "ADAPT_DEPLOY — running full daily workflow"
echo "Project dir: $PROJECT_DIR"
echo "============================================================"

python run_all.py

echo
echo "============================================================"
echo "ADAPT_DEPLOY — signal files updated"
echo "============================================================"
echo "core     -> outputs/signals/core_signal_latest.json"
echo "alpha    -> outputs/signals/alpha_signal_latest.json"
echo "combined -> outputs/signals/combined_signal_latest.json"
