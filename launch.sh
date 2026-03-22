#!/usr/bin/env bash
# launch.sh — LoRA validation training pipeline launcher.
#
# Creates a Python virtual environment, installs all dependencies,
# then launches training with accelerate + FSDP.
#
# Usage:
#   bash launch.sh [--train_data <path> ...] [--adversarial_data <path> ...]
#                  [--output_dir <dir>] [--model_id <hf_id>] [--skip_eval]
#
# All arguments after the script name are forwarded to train.py.
# The launch uses the fsdp_config.yaml in this directory.
# Set num_processes in fsdp_config.yaml to match your GPU count.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

echo "=== LoRA Validation Training Pipeline ==="
echo "Repo:  ${REPO_DIR}"
echo "Venv:  ${VENV_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    echo "[1/3] Creating Python virtual environment…"
    python3 -m venv "${VENV_DIR}"
else
    echo "[1/3] Virtual environment already exists, skipping creation."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "[2/3] Installing dependencies…"
pip install --upgrade pip --quiet
pip install -r "${REPO_DIR}/requirements.txt" --quiet

# Flash attention requires a special install flag
pip install flash-attn --no-build-isolation --quiet || \
    echo "  Warning: flash-attn installation failed — continuing without it."

echo "  Dependencies installed."

# ---------------------------------------------------------------------------
# 3. Launch training
# ---------------------------------------------------------------------------
echo "[3/3] Launching training with accelerate + FSDP…"
echo ""

# Default data arguments — override by passing arguments to this script.
# Example invocation (all tiers, no adversarial):
#   bash launch.sh \
#     --train_data data/tier1.jsonl data/tier2.jsonl data/tier3.jsonl \
#     --output_dir ./checkpoints

accelerate launch \
    --config_file "${REPO_DIR}/fsdp_config.yaml" \
    "${REPO_DIR}/train.py" \
    "$@"
