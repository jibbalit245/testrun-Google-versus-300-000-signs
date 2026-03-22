#!/usr/bin/env bash
# deploy.sh — Deployment script for the LoRA validation training pipeline
#
# Steps:
#   1. Create and activate a Python virtual environment
#   2. Install all dependencies
#   3. Auto-detect GPUs (count + per-GPU VRAM)
#   4. Single-step dry-run on 1 GPU to verify no errors
#   5. Full multi-GPU training launch using all detected GPUs
#
# Usage:
#   bash deploy.sh [--dry-run-only] [--standard_data <path>] [--adversarial_data <path>] [--output_dir <path>]
#
# Environment variables:
#   OPENAI_API_KEY      — required if using OpenAI backend for data generation
#   ANTHROPIC_API_KEY   — required if using Anthropic backend for data generation
#   HF_TOKEN            — optional, for gated model access on HuggingFace
#   WANDB_API_KEY       — optional, for Weights & Biases logging

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-./venv}"
STANDARD_DATA="${STANDARD_DATA:-}"
ADVERSARIAL_DATA="${ADVERSARIAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
DRY_RUN_ONLY=0
PYTHON="${PYTHON:-python3}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run-only)   DRY_RUN_ONLY=1; shift ;;
    --standard_data)  STANDARD_DATA="$2"; shift 2 ;;
    --adversarial_data) ADVERSARIAL_DATA="$2"; shift 2 ;;
    --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Logging helpers ───────────────────────────────────────────────────────────
log()  { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }
info() { log "INFO  $*"; }
warn() { log "WARN  $*"; }
die()  { log "ERROR $*" >&2; exit 1; }

# ── 1. Virtual environment ────────────────────────────────────────────────────
info "Creating virtual environment at $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Python: $(python --version)"

# ── 2. Install dependencies ───────────────────────────────────────────────────
info "Installing dependencies from requirements.txt"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Dependencies installed."

# ── 3. GPU auto-detection ─────────────────────────────────────────────────────
info "Detecting GPUs..."
GPU_INFO=$(python - <<'EOF'
import torch, sys
n = torch.cuda.device_count()
if n == 0:
    print("NO_GPU")
else:
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU{i}: {props.name} — {vram_gb:.1f} GB VRAM")
    print(f"TOTAL_GPUS={n}")
EOF
)

echo "$GPU_INFO"

# Extract GPU count
if echo "$GPU_INFO" | grep -q "NO_GPU"; then
    warn "No CUDA GPUs detected. Debug mode only."
    GPU_COUNT=0
else
    GPU_COUNT=$(echo "$GPU_INFO" | grep "^TOTAL_GPUS=" | cut -d= -f2)
    info "Detected $GPU_COUNT GPU(s)"
fi

# ── 4. Single-step dry-run ────────────────────────────────────────────────────
info "Running single-step dry-run (debug mode) to verify configuration..."
python train.py --debug
DRY_RUN_EXIT=$?

if [[ $DRY_RUN_EXIT -ne 0 ]]; then
    die "Dry-run failed with exit code $DRY_RUN_EXIT. Fix errors before full training."
fi
info "Dry-run passed."

if [[ $DRY_RUN_ONLY -eq 1 ]]; then
    info "--dry-run-only specified. Exiting after successful dry-run."
    exit 0
fi

# ── 5. Full multi-GPU training ────────────────────────────────────────────────
if [[ -z "$STANDARD_DATA" && -z "$ADVERSARIAL_DATA" ]]; then
    warn "No training data specified. Set --standard_data and/or --adversarial_data."
    warn "Skipping full training launch."
    exit 0
fi

# Build data arguments
DATA_ARGS=""
[[ -n "$STANDARD_DATA" ]] && DATA_ARGS="$DATA_ARGS --standard_data $STANDARD_DATA"
[[ -n "$ADVERSARIAL_DATA" ]] && DATA_ARGS="$DATA_ARGS --adversarial_data $ADVERSARIAL_DATA"

mkdir -p "$OUTPUT_DIR"

if [[ $GPU_COUNT -le 1 ]]; then
    info "Launching single-GPU training (GPU_COUNT=$GPU_COUNT)..."
    python train.py $DATA_ARGS --output_dir "$OUTPUT_DIR"
else
    info "Launching multi-GPU training with $GPU_COUNT GPUs via torchrun..."
    torchrun \
        --nproc_per_node="$GPU_COUNT" \
        --master_port=29500 \
        train.py \
        $DATA_ARGS \
        --output_dir "$OUTPUT_DIR"
fi

info "Training complete. Outputs in $OUTPUT_DIR"
