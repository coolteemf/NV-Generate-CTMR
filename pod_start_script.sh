#!/bin/bash
set -e

# --- 1. System Dependencies (Ephemeral - Must run every start) ---
# apt packages reside in system folders, so they reset on restart.
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y git curl

# --- 2. Install Poetry (Ephemeral Binary, Persistent Config) ---
# We reinstall the binary to ensuring it's in the current container's PATH/libs
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

export PATH="/root/.local/bin:$PATH"

# --- 3. Project Setup (Persistent) ---
# CHANGED: Target the persistent volume
PROJECT_DIR="/workspace/Projects/NV-Generate-CTMR"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Configure Poetry to store venv LOCALLY in the project folder
# Since the project is in /workspace, the .venv will also be in /workspace
poetry config virtualenvs.in-project true