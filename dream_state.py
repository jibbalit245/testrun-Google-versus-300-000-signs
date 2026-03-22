#!/usr/bin/env python3
"""
dream_state.py — Tier 1 → stated connections.

After tier 1 training, run the trained model over tier 1 data to generate
"stated connections" (explicit cross-domain relationships identified by the
model's emerging reasoning capability).  These connections become the seed
material for the adversarial exchange in adversary_challenge.py.

The output is a JSONL file where each record contains:
  - "source_text": the original passage
  - "observe_probe": the model's OBSERVE/PROBE for that passage
  - "connections": list of cross-domain connections extracted from [PROBE]
  - "tier": 1

Usage:
    python dream_state.py \\
        --model_dir ./checkpoints/final \\
        --tier1_data data/tier1.jsonl \\
        --output data/dream_connections.jsonl \\
        [--max_samples 200]
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection extraction helpers
# ---------------------------------------------------------------------------

_PROBE_RE = re.compile(r"\[PROBE\](.*?)(?:\[|$)", re.DOTALL)
# Split on sentence boundaries ending with '?' — handles both newline-separated
# and space-separated questions within a paragraph.
_SENTENCE_SPLIT_RE = re.compile(r"\?\s+(?=[A-Z])")


def extract_connections(observe_probe_text: str) -> List[str]:
    """
    Extract question strings from the [PROBE] block of an OBSERVE/PROBE
    assistant turn.  Returns a list of question strings.

    Handles questions separated by newlines *or* by spaces (e.g. when the
    model emits the block as a single paragraph).
    """
    probe_match = _PROBE_RE.search(observe_probe_text)
    if not probe_match:
        return []
    probe_block = probe_match.group(1).strip()

    # First try: split on numbered/bulleted lines
    line_questions = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in probe_block.splitlines()
        if line.strip().endswith("?")
    ]
    if len(line_questions) >= 2:
        return line_questions

    # Fallback: split on sentence boundaries ending with '?'
    sentences = _SENTENCE_SPLIT_RE.split(probe_block)
    questions = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
        # Re-add the '?' that was consumed by the split (except last segment)
        if i < len(sentences) - 1:
            sent = sent + "?"
        elif not sent.endswith("?"):
            # Last sentence — only include if it is a question
            if "?" not in sent:
                continue
        questions.append(sent)
    return questions


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str):
    """Load trained model + tokenizer from a local directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Try loading as a PEFT model first; fall back to plain AutoModel
    base_config_path = Path(model_dir) / "adapter_config.json"
    if base_config_path.exists():
        import json as _json
        adapter_config = _json.loads(base_config_path.read_text())
        base_model_id = adapter_config.get("base_model_name_or_path", model_dir)
        logger.info("Loading base model %s + LoRA adapter…", base_model_id)
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        logger.info("Loading model directly from %s…", model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def generate_observe_probe(
    model,
    tokenizer,
    system_prompt: str,
    passage: str,
    max_new_tokens: int = 400,
) -> str:
    """Run inference to get an OBSERVE/PROBE output for a passage."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": passage},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TIER1_SYSTEM = (
    "You are learning to see. Every passage contains structure — "
    "constraints that define what is possible, relationships that reveal "
    "what connects, and gaps that show where understanding breaks down. "
    "Your task is not to summarize or retrieve. Your task is to observe "
    "what is actually present, identify what follows from it, and ask the "
    "question that opens it further. When you encounter a boundary, map it. "
    "The boundary defines the shape of what lies beyond it."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stated connections from tier 1 training data."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the trained model directory (checkpoint or final).",
    )
    parser.add_argument(
        "--tier1_data",
        default="data/tier1.jsonl",
        help="Path to the tier 1 training JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/dream_connections.jsonl",
        help="Path to write the output connections JSONL.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Max new tokens for generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load tier 1 data
    tier1_path = Path(args.tier1_data)
    records = []
    with open(tier1_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.max_samples is not None:
        records = records[: args.max_samples]

    logger.info("Processing %d tier 1 samples…", len(records))

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for idx, record in enumerate(records):
            # Extract passage (user turn)
            if "messages" in record:
                user_msg = next(
                    (m for m in record["messages"] if m["role"] == "user"), None
                )
                passage = user_msg["content"] if user_msg else ""
            elif "text" in record:
                # Heuristic: extract content between last user token and assistant token
                text = record["text"]
                start = text.rfind("<|im_start|>user\n") + len("<|im_start|>user\n")
                end = text.rfind("<|im_end|>", start)
                passage = text[start:end].strip() if start > 0 else text
            else:
                continue

            if not passage:
                continue

            observe_probe = generate_observe_probe(
                model,
                tokenizer,
                TIER1_SYSTEM,
                passage,
                max_new_tokens=args.max_new_tokens,
            )
            connections = extract_connections(observe_probe)

            out_record = {
                "tier": 1,
                "source_text": passage,
                "observe_probe": observe_probe,
                "connections": connections,
            }
            out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                logger.info("  processed %d/%d samples", idx + 1, len(records))

    logger.info("Wrote connections to %s", output_path)


if __name__ == "__main__":
    main()
