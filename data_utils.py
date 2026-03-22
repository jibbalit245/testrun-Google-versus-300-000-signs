"""
data_utils.py — Data loading, validation, packing, and merging for the
LoRA validation training pipeline.

Training data format (ChatML with Qwen tokens):

  <|im_start|>system
  {tier_system_prompt}<|im_end|>
  <|im_start|>user
  {raw_source_content}<|im_end|>
  <|im_start|>assistant
  [TIER:{n}] [PRIOR:{prior_name}] [DOMAIN:{domains}] [SOURCE:{type}]
  [OBSERVE] ...
  [PROBE] ...<|im_end|>

Loss is computed ONLY on the assistant turn (train_on_input: false).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ── Tier system prompts ────────────────────────────────────────────────────────

TIER_SYSTEM_PROMPTS: dict[int, str] = {
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

# Required assistant-turn tags
_REQUIRED_TAGS = re.compile(
    r"\[TIER:\d+\].*\[PRIOR:[^\]]+\].*\[DOMAIN:[^\]]+\].*\[SOURCE:[^\]]+\]"
    r".*\[OBSERVE\].*\[PROBE\]",
    re.DOTALL,
)

# ── Validation helpers ─────────────────────────────────────────────────────────


def validate_sample(sample: dict[str, Any]) -> bool:
    """Return True if *sample* conforms to the required format."""
    required_keys = {"system", "user", "assistant", "tier"}
    if not required_keys.issubset(sample.keys()):
        return False
    tier = sample["tier"]
    if tier not in TIER_SYSTEM_PROMPTS:
        return False
    if sample["system"].strip() != TIER_SYSTEM_PROMPTS[tier].strip():
        return False
    if not _REQUIRED_TAGS.search(sample["assistant"]):
        return False
    return True


# ── ChatML formatting ─────────────────────────────────────────────────────────


def format_chatml(sample: dict[str, Any]) -> str:
    """Return the full ChatML string for *sample*."""
    return (
        f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['user']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['assistant']}<|im_end|>"
    )


# ── Tokenisation with loss masking ────────────────────────────────────────────


def tokenise_and_mask(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4096,
) -> dict[str, list[int]]:
    """Tokenise *sample* and set labels to -100 for all non-assistant tokens.

    Only the assistant turn is supervised (train_on_input: false).
    """
    full_text = format_chatml(sample)
    # Build the prefix (system + user) to locate where assistant turn begins
    prefix = (
        f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['user']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    encoded_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    encoded_prefix = tokenizer(
        prefix,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )

    input_ids = encoded_full["input_ids"]
    attention_mask = encoded_full["attention_mask"]
    prefix_len = len(encoded_prefix["input_ids"])

    labels = [-100] * prefix_len + input_ids[prefix_len:]
    # Pad labels to same length
    labels = labels[: len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ── JSONL loading ─────────────────────────────────────────────────────────────


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a newline-delimited JSON file and return a list of records."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d of %s: %s", lineno, path, exc)
    return records


def load_and_validate_jsonl(
    path: str | Path,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Load *path* and validate each record.

    If *strict* is True, raise ValueError on the first invalid record.
    Otherwise, skip invalid records and log a warning.
    """
    raw = load_jsonl(path)
    valid: list[dict[str, Any]] = []
    for i, sample in enumerate(raw):
        if validate_sample(sample):
            valid.append(sample)
        else:
            msg = f"Invalid sample at index {i} in {path}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
    logger.info("Loaded %d / %d valid samples from %s", len(valid), len(raw), path)
    return valid


# ── Dataset building ──────────────────────────────────────────────────────────


def build_hf_dataset(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4096,
) -> Dataset:
    """Convert *samples* to a HuggingFace Dataset with tokenised columns."""
    tokenised = [tokenise_and_mask(s, tokenizer, max_length) for s in samples]
    return Dataset.from_list(tokenised)


def merge_datasets(*datasets: Dataset) -> Dataset:
    """Concatenate multiple HuggingFace datasets into one."""
    return concatenate_datasets(list(datasets))


def load_training_data(
    standard_path: str | Path | None,
    adversarial_path: str | Path | None,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4096,
    strict_validation: bool = False,
) -> Dataset:
    """Load and merge standard + adversarial JSONL files into a single dataset.

    Either *standard_path* or *adversarial_path* may be None; at least one
    must be provided.
    """
    parts: list[Dataset] = []
    if standard_path is not None:
        std_samples = load_and_validate_jsonl(standard_path, strict=strict_validation)
        parts.append(build_hf_dataset(std_samples, tokenizer, max_length))
    if adversarial_path is not None:
        adv_samples = load_and_validate_jsonl(adversarial_path, strict=strict_validation)
        parts.append(build_hf_dataset(adv_samples, tokenizer, max_length))
    if not parts:
        raise ValueError("At least one of standard_path or adversarial_path must be provided.")
    return merge_datasets(*parts) if len(parts) > 1 else parts[0]
