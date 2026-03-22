# LoRA Validation Training Pipeline — Qwen 2.5 72B

A 12-hour proof-of-concept training run to validate that the OBSERVE/PROBE
data format produces genuine reasoning behaviour in Qwen 2.5 72B Base via
QLoRA fine-tuning.

---

## Repository Structure

```
.
├── train.py                    # Main QLoRA/FSDP training script
├── data_utils.py               # Data loading, validation, tokenisation, packing
├── generate_assistant_turns.py # Generate OBSERVE/PROBE turns from raw source text
├── dream_state.py              # Extract stated connections from tier-1 trained model
├── adversary_challenge.py      # Run adversarial exchanges → tier-2 training data
├── evaluate.py                 # 5 evaluation probes with pass/fail scoring
├── training_format_spec.md     # Full training format specification
├── requirements.txt            # Python dependencies
└── deploy.sh                   # Deployment script with GPU auto-detection
```

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <this-repo>
cd testrun-Google-versus-300-000-signs

# 2. Deploy (creates venv, installs deps, dry-run verification, then trains)
bash deploy.sh \
    --standard_data data/tier1.jsonl \
    --output_dir checkpoints/

# 3. Debug mode — 5 steps on 1 GPU, then exits
python train.py --debug
```

---

## Hardware Requirements

| Configuration | Minimum | Recommended |
|---------------|---------|-------------|
| GPUs | 4× 48 GB | 8× 80 GB |
| Total VRAM | 192 GB | 640 GB |
| CUDA | 12.4+ | 12.4+ |
| Python | 3.11+ | 3.11+ |

GPU count and batch size are **auto-detected at runtime** — never hardcoded.

---

## Data Pipeline

### Step 1 — Generate OBSERVE/PROBE turns from raw source text

```bash
python generate_assistant_turns.py \
    --source_dir data/raw/ \
    --output data/tier1.jsonl \
    --tier 1 \
    --backend anthropic \
    --model_name claude-opus-4-5
```

Backends: `local` (Qwen2.5-14B-Instruct), `openai`, `anthropic`.

### Step 2 — Train tier 1

```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    train.py --standard_data data/tier1.jsonl --output_dir checkpoints/tier1/
```

### Step 3 — Adversarial integration (tier 1 → tier 2)

```bash
# 3a. Extract stated connections from tier-1 model
python dream_state.py \
    --model_path checkpoints/tier1/final_adapter \
    --tier1_data data/tier1.jsonl \
    --output data/dream_connections.jsonl

# 3b. Run adversarial challenges and format as tier-2 training data
python adversary_challenge.py \
    --connections data/dream_connections.jsonl \
    --adversary_model adversary-forge/qwen-adversary \
    --student_model checkpoints/tier1/final_adapter \
    --output data/tier1_adversarial.jsonl
```

### Step 4 — Train tier 2 (with adversarial data)

```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    train.py \
    --standard_data data/tier2.jsonl \
    --adversarial_data data/tier1_adversarial.jsonl \
    --output_dir checkpoints/tier2/
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-72B (base, not instruct) |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Distribution | FSDP full shard — Qwen2DecoderLayer wrap |
| Epochs | 3 |
| LR schedule | Cosine with 3% warmup |
| Checkpoint every | 200 steps |
| Max sequence length | 4096 tokens |
| Packing | Enabled |

---

## Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/tier2/final_adapter \
    --output_log eval_results.json
```

### Probes

| # | Name | Tests |
|---|------|-------|
| 1 | Novel reasoning from observation | Salt + boiling water |
| 2 | Cross-domain structural connection | Trophic cascade analog |
| 3 | Boundary mapping | Perpetual motion |
| 4 | Honest uncertainty | Consciousness |
| 5 | Collaborative contribution | Gravity / spin hypothesis |

### Pass Criteria

- **4/5** → VALIDATED. Data direction confirmed.
- **2–3/5** → PARTIAL. Adjust format or data and retry.
- **0–1/5** → FAILED. Format problem — revisit data generation.

---

## Training Data Format

See [`training_format_spec.md`](training_format_spec.md) for the full
specification including:
- Exact ChatML token format (`<|im_start|>` / `<|im_end|>`)
- All tier system prompts (tiers 1–3 for this validation run)
- Required assistant turn tag structure
- JSONL schema
- Loss masking rules

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | If using OpenAI backend | Data generation |
| `ANTHROPIC_API_KEY` | If using Anthropic backend | Data generation |
| `HF_TOKEN` | Optional | Gated model access |
| `WANDB_API_KEY` | Optional | Weights & Biases logging |