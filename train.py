"""
train.py — QLoRA (4-bit NF4) fine-tuning of Qwen2.5-72B-Base with FSDP full
shard, packing, and cosine LR schedule.

Usage
-----
  # Full multi-GPU training (uses all detected GPUs):
  torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \\
      train.py --standard_data data/tier1.jsonl --output_dir checkpoints/

  # 1-GPU debug mode (5 steps, then exit):
  python train.py --debug

  # Include adversarial data:
  python train.py --standard_data data/tier1.jsonl \\
                  --adversarial_data data/tier1_adversarial.jsonl \\
                  --output_dir checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-72B-Base"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

NUM_EPOCHS = 3
SAVE_STEPS = 200
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 4096
DEBUG_STEPS = 5

# ── GPU auto-detection ────────────────────────────────────────────────────────


def detect_gpus() -> tuple[int, list[int]]:
    """Return (gpu_count, vram_per_gpu_mb) using torch.cuda."""
    n = torch.cuda.device_count()
    vrams: list[int] = []
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vrams.append(props.total_memory // (1024 * 1024))
    return n, vrams


def compute_batch_size(gpu_count: int, vram_mb_per_gpu: int, debug: bool) -> tuple[int, int]:
    """Heuristically choose per-device batch size and gradient accumulation steps.

    Rules of thumb (4-bit 72B LoRA):
      - < 40 GB VRAM → batch 1, accum 16
      - 40–79 GB     → batch 2, accum 8
      - ≥ 80 GB      → batch 4, accum 4
    """
    if debug:
        return 1, 1
    if vram_mb_per_gpu < 40_000:
        per_device, accum = 1, 16
    elif vram_mb_per_gpu < 80_000:
        per_device, accum = 2, 8
    else:
        per_device, accum = 4, 4
    return per_device, accum


# ── FSDP configuration ────────────────────────────────────────────────────────


def build_fsdp_config() -> dict:
    """Return accelerate-compatible FSDP config for Qwen2 with full sharding."""
    return {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_offload_params": False,
        "fsdp_cpu_ram_efficient_loading": True,
        # PCIe interconnect — conservative backward prefetch
        "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        "fsdp_use_orig_params": True,
        "fsdp_sync_module_states": True,
    }


# ── QLoRA model loading ───────────────────────────────────────────────────────


def load_model_and_tokenizer(model_name: str, debug: bool = False):
    """Load the base model with 4-bit NF4 quantisation and apply LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model %s with 4-bit NF4 quantisation", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=None,  # FSDP manages device placement
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────


def build_training_args(
    output_dir: str,
    per_device_batch: int,
    grad_accum: int,
    num_epochs: int,
    debug: bool,
    use_fsdp: bool,
):
    """Construct TrainingArguments."""
    from transformers import TrainingArguments

    max_steps = DEBUG_STEPS if debug else -1

    fsdp_value = "full_shard" if use_fsdp else ""
    fsdp_config = build_fsdp_config() if use_fsdp else {}

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs if not debug else 1,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        fsdp=fsdp_value,
        fsdp_config=fsdp_config if fsdp_config else None,
        remove_unused_columns=False,
        # Do not train on inputs (assistant-turn-only loss is handled in data_utils)
        label_names=["labels"],
    )


def run_training(args: argparse.Namespace) -> None:
    """Main training entry point."""
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from data_utils import load_training_data

    start_time = time.time()

    # ── GPU detection ──────────────────────────────────────────────────────
    gpu_count, vrams = detect_gpus()
    if gpu_count == 0:
        logger.warning("No CUDA GPUs detected — falling back to CPU (debug only)")
        vram_min = 0
    else:
        vram_min = min(vrams)
        logger.info(
            "Detected %d GPU(s): %s",
            gpu_count,
            ", ".join(f"GPU{i} {v} MB" for i, v in enumerate(vrams)),
        )

    use_fsdp = (gpu_count > 1) and not args.debug
    per_device_batch, grad_accum = compute_batch_size(gpu_count, vram_min, args.debug)

    logger.info(
        "Training config: per_device_batch=%d, grad_accum=%d, fsdp=%s",
        per_device_batch,
        grad_accum,
        use_fsdp,
    )

    # ── Model + tokenizer ──────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, debug=args.debug)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = load_training_data(
        standard_path=args.standard_data,
        adversarial_path=args.adversarial_data,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
    )
    logger.info("Total training samples: %d", len(dataset))

    # ── Training arguments ─────────────────────────────────────────────────
    training_args = build_training_args(
        output_dir=args.output_dir,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        num_epochs=NUM_EPOCHS,
        debug=args.debug,
        use_fsdp=use_fsdp,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=None,  # data is already tokenised
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )

    logger.info("Starting training%s", " (DEBUG — 5 steps)" if args.debug else "")
    trainer.train()

    # ── Time budget check ──────────────────────────────────────────────────
    elapsed_hours = (time.time() - start_time) / 3600
    logger.info("Training finished in %.2f hours", elapsed_hours)
    if elapsed_hours > 12:
        logger.warning("Training exceeded 12-hour budget (%.2f hours)", elapsed_hours)

    if not args.debug:
        output_path = Path(args.output_dir) / "final_adapter"
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        logger.info("Saved final adapter to %s", output_path)

        # Run evaluation probes
        from evaluate import run_all_probes
        run_all_probes(str(output_path), tokenizer=tokenizer)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA training for Qwen2.5-72B reasoning validation"
    )
    parser.add_argument(
        "--standard_data",
        type=str,
        default=None,
        help="Path to standard training JSONL file",
    )
    parser.add_argument(
        "--adversarial_data",
        type=str,
        default=None,
        help="Path to adversarial exchange JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints and final adapter",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run 5 steps on 1 GPU and exit (configuration validation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    args = parse_args()

    if not args.debug and args.standard_data is None and args.adversarial_data is None:
        logger.error("Provide --standard_data and/or --adversarial_data for non-debug runs.")
        sys.exit(1)

    run_training(args)
