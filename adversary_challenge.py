"""
adversary_challenge.py — Load adversary model, run challenge exchanges against
the tier-1 trained model's stated connections, and format results as tier-2
training data.

Adversarial Integration step 2 of 3 (between tier 1 and tier 2 training).

Usage
-----
  python adversary_challenge.py \\
      --connections data/dream_connections.jsonl \\
      --adversary_model adversary-forge/qwen-adversary \\
      --student_model checkpoints/final_adapter \\
      --output data/tier1_adversarial.jsonl \\
      [--max_exchanges 200]

Output JSONL schema (formatted as tier-2 training samples):
  {
    "tier": 2,
    "system": <tier_2_system_prompt>,
    "user": <original passage + adversary challenge>,
    "assistant": <tier2-formatted OBSERVE/PROBE resolving the challenge>,
    "prior": <str>,
    "domain": <str>,
    "source_type": "adversarial_exchange"
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from data_utils import TIER_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

# ── Adversary prompts ──────────────────────────────────────────────────────────

_ADVERSARY_SYSTEM = (
    "You are a rigorous intellectual adversary. Your role is to challenge "
    "stated connections and structural observations with precision. "
    "When given a stated connection, identify exactly where the reasoning "
    "is weakest: what assumption is unstated, what edge case breaks the "
    "connection, or what alternative framework would give a different "
    "conclusion. Be specific. Ask the one question that, if answered, would "
    "either validate or destroy the connection. Do not be polite about gaps."
)

_ADVERSARY_USER_TEMPLATE = (
    "STATED CONNECTION:\n{connection}\n\n"
    "CONTEXT (original passage):\n{passage}\n\n"
    "Challenge this connection. Identify the weakest point. Ask the one "
    "question that tests whether this connection holds."
)

_RESOLUTION_SYSTEM = (
    "You are resolving an adversarial challenge to a structural observation. "
    "Given the original passage, the stated connection, and the adversary's "
    "challenge, produce a tier-2 OBSERVE/PROBE annotation that:\n"
    "1. Acknowledges what the adversary's challenge reveals\n"
    "2. Follows the chain of reasoning from premise to conclusion\n"
    "3. Identifies the step that was assumed or skipped\n"
    "4. Asks what else follows from the same premises\n\n"
    "Use EXACTLY this format:\n"
    "[TIER:2] [PRIOR:{prior}] [DOMAIN:{domain}] [SOURCE:adversarial_exchange]\n"
    "[OBSERVE] <2-4 sentences>\n"
    "[PROBE] <2-4 questions>"
)

_RESOLUTION_USER_TEMPLATE = (
    "ORIGINAL PASSAGE:\n{passage}\n\n"
    "STATED CONNECTION:\n{connection}\n\n"
    "ADVERSARY CHALLENGE:\n{challenge}\n\n"
    "Produce the tier-2 OBSERVE/PROBE annotation resolving this challenge."
)


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(model_path: str, quantize: bool = True):
    """Load a model (possibly with LoRA adapter) for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if quantize:
        from transformers import BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # Try to load as PEFT adapter; fall back to base model
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                base_name = json.load(f).get("base_model_name_or_path", model_path)
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int = 512,
) -> str:
    """Run a single chat-formatted generation."""
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Exchange pipeline ─────────────────────────────────────────────────────────


def run_exchange(
    connection_record: dict[str, Any],
    adversary_model,
    adversary_tokenizer,
    student_model,
    student_tokenizer,
) -> list[dict[str, Any]]:
    """Run adversary challenges for each connection in *connection_record*.

    Returns a list of tier-2 formatted training samples.
    """
    passage = connection_record.get("source_user", "")
    tier1_assistant = connection_record.get("tier1_assistant", "")
    connections = connection_record.get("stated_connections", [])

    # Infer metadata from original assistant turn
    prior = "general_reasoning"
    domain = "general"
    import re
    prior_match = re.search(r"\[PRIOR:([^\]]+)\]", tier1_assistant)
    domain_match = re.search(r"\[DOMAIN:([^\]]+)\]", tier1_assistant)
    if prior_match:
        prior = prior_match.group(1)
    if domain_match:
        domain = domain_match.group(1)

    samples: list[dict[str, Any]] = []
    for connection in connections:
        # Step 1: adversary challenges the connection
        challenge = generate(
            adversary_model,
            adversary_tokenizer,
            system=_ADVERSARY_SYSTEM,
            user=_ADVERSARY_USER_TEMPLATE.format(connection=connection, passage=passage),
        )

        # Step 2: student model resolves the challenge as tier-2 OBSERVE/PROBE
        resolution = generate(
            student_model,
            student_tokenizer,
            system=_RESOLUTION_SYSTEM.format(prior=prior, domain=domain),
            user=_RESOLUTION_USER_TEMPLATE.format(
                passage=passage,
                connection=connection,
                challenge=challenge,
            ),
        )

        samples.append({
            "tier": 2,
            "system": TIER_SYSTEM_PROMPTS[2],
            "user": (
                f"ORIGINAL PASSAGE:\n{passage}\n\n"
                f"ADVERSARY CHALLENGE:\n{challenge}"
            ),
            "assistant": resolution,
            "prior": prior,
            "domain": domain,
            "source_type": "adversarial_exchange",
        })

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    from data_utils import load_jsonl

    connections = load_jsonl(args.connections)
    if args.max_exchanges:
        connections = connections[: args.max_exchanges]

    logger.info("Loaded %d connection records", len(connections))

    logger.info("Loading adversary model from %s", args.adversary_model)
    adversary_model, adversary_tokenizer = load_model(args.adversary_model, quantize=False)

    logger.info("Loading student model from %s", args.student_model)
    student_model, student_tokenizer = load_model(args.student_model, quantize=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, record in enumerate(connections):
            logger.info("Exchange %d/%d", i + 1, len(connections))
            try:
                samples = run_exchange(
                    record,
                    adversary_model,
                    adversary_tokenizer,
                    student_model,
                    student_tokenizer,
                )
                for sample in samples:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += len(samples)
            except Exception as exc:
                logger.warning("Failed on record %d: %s", i, exc)

    logger.info("Wrote %d adversarial training samples to %s", total, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adversarial challenges and format as tier-2 training data"
    )
    parser.add_argument("--connections", required=True, help="Path to dream_state output JSONL")
    parser.add_argument("--adversary_model", required=True, help="Path or name of adversary model")
    parser.add_argument("--student_model", required=True, help="Path to trained student LoRA adapter")
    parser.add_argument("--output", required=True, help="Output JSONL path for adversarial training samples")
    parser.add_argument("--max_exchanges", type=int, default=None, help="Maximum number of records to process")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    main(parse_args())
