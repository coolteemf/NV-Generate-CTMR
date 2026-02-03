#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- 1. System Dependencies ---
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y git curl

# --- 2. Install Poetry ---
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Ensure Poetry is in PATH for this session
export PATH="/root/.local/bin:$PATH"

# --- 3. Project Setup ---
# Define your project root
PROJECT_DIR="/root/Projects/NV-Generate-CTMR"

# Create directory if it doesn't exist (assuming repo might be cloned here later or mounted)
mkdir -p "$PROJECT_DIR"

# Navigate to project
cd "$PROJECT_DIR"

# Configure Poetry (Local Virtualenv is safer for debugging)
poetry config virtualenvs.in-project true

# Install Dependencies
# Only run install if pyproject.toml exists to avoid errors on empty folders
if [ -f "pyproject.toml" ]; then
    echo "Installing Dependencies..."
    poetry install
else
    echo "Warning: pyproject.toml not found in $PROJECT_DIR"
fi

# --- 4. Ready for Inference ---
echo "Setup Complete. You can now run the inference script."
# If you want to auto-start the inference immediately, uncomment the line below:
# poetry run python -m scripts.outpaint_inference -e configs/environment_rflow-ct.json -c configs/config_infer_24g_512x512x512.json -t configs/config_network_rflow.json -i ./heart_cropped.nii.gz --crop_center "120,120,86"
