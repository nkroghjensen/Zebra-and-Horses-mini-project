#!/usr/bin/env bash
set -e

# Create virtual environment in folder "venv"
python3 -m venv venv

# Activate venv (only for this script run)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt

echo "Done! Remember to run: source venv/bin/activate before working in this project."
