#!/bin/bash

# Move to project directory
cd "$(dirname "$0")"

echo "Launching ADAPT Strategy Terminal..."
echo "Using Python environment: ./venv"

# Activate environment and run dashboard
./venv/bin/streamlit run app.py
