"""
dream_state.py — Run the trained model over tier 1 training data to generate
"stated connections" (structural observations and inferred relationships) that
will seed the adversarial challenge step.

Adversarial Integration step 1 of 3 (between tier 1 and tier 2 training).

Usage
-----
  python dream_state.py \\
      --model_path checkpoints/final_adapter \\
      --tier1_data data/tier1.jsonl \\
      --output data/dream_connections.jsonl \\
      [--max_samples 500]

Output JSONL schema:
  {
    "source_user": <original user passage>,
    "tier1_assistant": <original tier1 assistant turn>,
    "stated_connections": [<list of connection strings extracted>],
    "raw_generation": <full model output>
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Connection extraction ──────────────────────────────────────────────────────

_CONNECTION_EXTRACTION_SYSTEM = (
    "You are analyzing reasoning patterns in a text. "
    "Given an OBSERVE/PROBE annotation for a passage, extract all explicit "
    "structural connections and inferred relationships stated or implied by "
    "the annotation. Return a JSON array of connection strings, each describing "
    "one structural connection or inference. Be precise and complete. "
    "Return ONLY valid JSON — no prose, no explanation."
)

_CONNECTION_EXTRACTION_USER = (
    "PASSAGE:\n{user}\n\n"
    "OBSERVE/PROBE ANNOTATION:\n{assistant}\n\n"
    "Extract all stated structural connections and inferred relationships as a "
    "JSON array of strings. Example: "
    '[\"A constrains B via mechanism X\", \"C follows from D when E holds\"]'
)


def extract_connections_from_generation(raw: str) -> list[str]:
    """Parse JSON array from the model's raw generation output."""
    # Find first JSON array in the output
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            connections = json.loads(match.group(0))
            if isinstance(connections, list):
                return [str(c) for c in connections if c]
        except json.JSONDecodeError:
            pass
    # Fallback: extract bullet points or numbered lines
    lines = [l.strip().lstrip("-•*0123456789.) ") for l in raw.split("\n")]
    return [l for l in lines if len(l) > 20]


# ── Model inference ───────────────────────────────────────────────────────────


def load_adapter_model(model_path: str):
    """Load the fine-tuned LoRA adapter for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-72B-Base"
    logger.info("Loading base model for inference: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer


def generate_connections(
    sample: dict[str, Any],
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> tuple[list[str], str]:
    """Run the model over one sample and return (connections, raw_output)."""
    import torch

    messages = [
        {"role": "system", "content": _CONNECTION_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": _CONNECTION_EXTRACTION_USER.format(
                user=sample["user"], assistant=sample["assistant"]
            ),
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    connections = extract_connections_from_generation(raw)
    return connections, raw


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    from data_utils import load_jsonl

    input_samples = load_jsonl(args.tier1_data)
    if args.max_samples:
        input_samples = input_samples[: args.max_samples]

    logger.info("Loaded %d tier-1 samples from %s", len(input_samples), args.tier1_data)

    model, tokenizer = load_adapter_model(args.model_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(input_samples):
            logger.info("Processing sample %d/%d", i + 1, len(input_samples))
            connections, raw = generate_connections(sample, model, tokenizer)
            record = {
                "source_user": sample.get("user", ""),
                "tier1_assistant": sample.get("assistant", ""),
                "stated_connections": connections,
                "raw_generation": raw,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d connection records to %s", len(input_samples), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract stated connections from tier-1 trained model"
    )
    parser.add_argument("--model_path", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--tier1_data", required=True, help="Path to tier-1 training JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path for connections")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples processed")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    main(parse_args())
