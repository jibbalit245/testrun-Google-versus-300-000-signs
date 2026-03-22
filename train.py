#!/usr/bin/env python3
"""
LoRA validation training pipeline — Qwen 2.5 72B (base).

QLoRA: 4-bit NF4, LoRA rank 64, alpha 128, all linear layers targeted.
Distributed: FSDP full shard, Qwen2DecoderLayer wrap unit.
Loss: assistant turn only (train_on_input=False).
Schedule: 3 epochs, cosine LR, 3% warmup, checkpoint every 200 steps.
After training: 5 evaluation probes are run and results logged.

Usage (via launch.sh):
    accelerate launch --config_file fsdp_config.yaml train.py \\
        --train_data data/tier1.jsonl data/tier2.jsonl data/tier3.jsonl \\
        --output_dir ./checkpoints \\
        [--adversarial_data data/adversarial.jsonl ...]
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-72B"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TIER_SYSTEM_PROMPTS = {
    1: (
        "You are learning to see. Every passage contains structure — "
        "constraints that define what is possible, relationships that reveal "
        "what connects, and gaps that show where understanding breaks down. "
        "Your task is not to summarize or retrieve. Your task is to observe "
        "what is actually present, identify what follows from it, and ask the "
        "question that opens it further. When you encounter a boundary, map it. "
        "The boundary defines the shape of what lies beyond it."
    ),
    2: (
        "You have axioms. Now derive from them. Every passage builds "
        "on principles you already hold. Your task is to follow the chain of "
        "reasoning — from premise through logic to conclusion — and verify that "
        "each step holds. When a derivation skips a step, identify what was "
        "assumed. When a proof reaches its conclusion, ask what else follows "
        "from the same premises that the author did not pursue."
    ),
    3: (
        "You have principles and derivations. Now apply them. Every "
        "passage puts theory into contact with reality. Your task is to observe "
        "where the model fits the world and where it doesn't. When application "
        "succeeds, note what made the mapping work. When it fails, map exactly "
        "where the abstraction breaks against the concrete. The gap between "
        "model and reality is where the next insight lives."
    ),
}

# ChatML tokens used by Qwen
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
INSTRUCTION_TEMPLATE = "<|im_start|>user\n"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    """Load a newline-delimited JSON file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_sample(sample: dict) -> bool:
    """
    Validate that a sample conforms to the expected training format.

    Accepted shapes:
    1. ``{"messages": [{"role": ..., "content": ...}, ...]}``
    2. ``{"text": "<|im_start|>system\\n..."}``  (pre-formatted ChatML)
    """
    if "messages" in sample:
        msgs = sample["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            return False
        roles = [m.get("role") for m in msgs]
        if "assistant" not in roles:
            return False
        return True
    if "text" in sample:
        text = sample["text"]
        return (
            "<|im_start|>" in text
            and "<|im_end|>" in text
            and "assistant" in text
        )
    return False


def load_and_validate_datasets(
    paths: List[str],
    label: str = "training",
) -> Dataset:
    """Load, validate and merge multiple JSONL files into one Dataset."""
    all_records: List[dict] = []
    for path in paths:
        raw = load_jsonl(path)
        valid = [r for r in raw if validate_sample(r)]
        dropped = len(raw) - len(valid)
        if dropped:
            logger.warning(
                "%s: dropped %d/%d invalid records from %s",
                label,
                dropped,
                len(raw),
                path,
            )
        logger.info("%s: loaded %d records from %s", label, len(valid), path)
        all_records.extend(valid)

    if not all_records:
        raise ValueError(f"No valid {label} records found in: {paths}")

    logger.info("%s: %d total records", label, len(all_records))
    return Dataset.from_list(all_records)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def apply_chat_template(sample: dict, tokenizer) -> dict:
    """
    Convert a sample to a formatted text string using the tokenizer's
    chat template (ChatML).  Accepts both ``messages`` and ``text`` shapes.
    """
    if "text" in sample:
        return sample
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Model + LoRA setup
# ---------------------------------------------------------------------------

def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config() -> LoraConfig:
    """LoRA rank 64, alpha 128 targeting all linear projection layers."""
    return LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(model_id: str):
    """Load the base model with 4-bit quantisation and the tokenizer."""
    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model from %s (4-bit NF4)", model_id)
    bnb_config = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def build_training_arguments(output_dir: str) -> TrainingArguments:
    """Return TrainingArguments matching the spec."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_steps=200,
        save_total_limit=10,
        logging_steps=10,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        group_by_length=False,   # packing handles length grouping
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        fsdp="full_shard",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA validation training — Qwen 2.5 72B"
    )
    parser.add_argument(
        "--train_data",
        nargs="+",
        required=True,
        help="Path(s) to standard training JSONL file(s).",
    )
    parser.add_argument(
        "--adversarial_data",
        nargs="*",
        default=[],
        help="Path(s) to adversarial exchange JSONL file(s) (optional).",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints",
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--model_id",
        default=MODEL_ID,
        help="HuggingFace model ID for the base model.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation probes after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Data ---
    logger.info("Loading standard training data from: %s", args.train_data)
    train_ds = load_and_validate_datasets(args.train_data, label="standard")

    if args.adversarial_data:
        logger.info(
            "Loading adversarial data from: %s", args.adversarial_data
        )
        adv_ds = load_and_validate_datasets(
            args.adversarial_data, label="adversarial"
        )
        train_ds = concatenate_datasets([train_ds, adv_ds])
        logger.info(
            "Merged dataset size: %d records", len(train_ds)
        )

    # --- Model + tokenizer ---
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # --- Format dataset ---
    train_ds = train_ds.map(
        lambda s: apply_chat_template(s, tokenizer),
        desc="Applying chat template",
    )

    # --- Data collator: loss on assistant turn only ---
    response_ids = tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    instruction_ids = tokenizer.encode(
        INSTRUCTION_TEMPLATE, add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids,
        instruction_template=instruction_ids,
        tokenizer=tokenizer,
    )

    # --- Training arguments ---
    training_args = build_training_arguments(args.output_dir)

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        dataset_text_field="text",
        packing=True,
        max_seq_length=4096,
        tokenizer=tokenizer,
    )

    logger.info("Starting training…")
    trainer.train()

    logger.info("Saving final model to %s/final", args.output_dir)
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    # --- Evaluation probes ---
    if not args.skip_eval:
        logger.info("Running evaluation probes…")
        from evaluate import run_evaluation_probes
        results = run_evaluation_probes(model, tokenizer)
        probe_log_path = Path(args.output_dir) / "eval_results.json"
        with open(probe_log_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Evaluation results written to %s", probe_log_path)
        passed = sum(1 for r in results["probes"] if r["pass"])
        total = len(results["probes"])
        logger.info("Evaluation: %d/%d probes passed", passed, total)
        if passed >= 4:
            logger.info("RESULT: Data direction VALIDATED (≥4/5 passed).")
        elif passed >= 2:
            logger.info("RESULT: Marginal (2-3/5 passed) — adjust and retry.")
        else:
            logger.warning("RESULT: Format problem (0-1/5 passed) — revisit training format.")


if __name__ == "__main__":
    main()
