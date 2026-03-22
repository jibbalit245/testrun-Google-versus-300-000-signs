#!/usr/bin/env python3
"""
adversary_challenge.py — Adversarial exchange integration.

Loads the adversary model (from adversary-forge repo or a local path),
runs challenge exchanges against the stated connections produced by
dream_state.py, and formats the resulting exchanges as tier 2 training
data (JSONL) to be merged into tier 2 training.

The adversary challenges the model's stated connections by probing for:
  - Unsupported leaps in reasoning
  - Domain-specific counter-examples
  - Boundary conditions where the claimed connection breaks
  - Alternative structural interpretations

The formatted tier-2 exchange has the same ChatML structure as standard
training data.  The adversary's challenge becomes part of the [PROBE],
and the resolution becomes the [OBSERVE] update.

Usage:
    python adversary_challenge.py \\
        --connections data/dream_connections.jsonl \\
        --adversary_model path/to/adversary-forge \\
        --output data/adversarial.jsonl \\
        [--max_exchanges 100]
"""

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier 2 system prompt (exact text from spec)
# ---------------------------------------------------------------------------

TIER2_SYSTEM = (
    "You have axioms. Now derive from them. Every passage builds "
    "on principles you already hold. Your task is to follow the chain of "
    "reasoning — from premise through logic to conclusion — and verify that "
    "each step holds. When a derivation skips a step, identify what was "
    "assumed. When a proof reaches its conclusion, ask what else follows "
    "from the same premises that the author did not pursue."
)

# ---------------------------------------------------------------------------
# Adversary challenge prompt template
# ---------------------------------------------------------------------------

ADVERSARY_SYSTEM = textwrap.dedent("""\
    You are a rigorous intellectual adversary. Your role is to challenge
    stated connections by:
    1. Identifying hidden assumptions
    2. Providing domain-specific counter-examples
    3. Locating the precise point where the reasoning breaks
    4. Proposing an alternative structural interpretation

    Be specific. Be rigorous. Do not accept vague analogies as connections.
    A valid connection must have a precise structural mapping, not merely
    thematic resemblance.
""")


def build_adversary_prompt(
    passage: str,
    observe_probe: str,
    connection: str,
) -> str:
    return (
        f"Original passage:\n{passage}\n\n"
        f"Reasoning produced:\n{observe_probe}\n\n"
        f"Connection claimed:\n{connection}\n\n"
        "Challenge this connection. Be precise. Identify exactly where it "
        "holds and where it breaks."
    )


def build_resolution_prompt(
    passage: str,
    connection: str,
    challenge: str,
) -> str:
    return (
        f"Original passage:\n{passage}\n\n"
        f"Connection claimed:\n{connection}\n\n"
        f"Adversarial challenge:\n{challenge}\n\n"
        "Produce a tier 2 OBSERVE/PROBE response that:\n"
        "- Acknowledges the precise boundary where the connection breaks\n"
        "- Identifies the hidden assumption that was challenged\n"
        "- Derives what follows from the corrected/refined premise\n"
        "- Poses questions that test the refined connection\n\n"
        "Format:\n"
        "[TIER:2] [PRIOR:Adversarial] [DOMAIN:cross-domain] [SOURCE:exchange]\n"
        "[OBSERVE] ...\n"
        "[PROBE] ..."
    )


# ---------------------------------------------------------------------------
# Adversary model loader
# ---------------------------------------------------------------------------

_adversary_pipeline = None
_resolution_pipeline = None


def load_adversary_model(adversary_model_path: str):
    """Load the adversary model from the adversary-forge repo or local path."""
    global _adversary_pipeline
    if _adversary_pipeline is not None:
        return _adversary_pipeline

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    logger.info("Loading adversary model from %s…", adversary_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adversary_model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        adversary_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _adversary_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return _adversary_pipeline


def generate_adversary_response(
    adv_pipe,
    prompt: str,
    system: str = ADVERSARY_SYSTEM,
    max_new_tokens: int = 300,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    out = adv_pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return out[0]["generated_text"][-1]["content"].strip()


# ---------------------------------------------------------------------------
# Exchange formatting
# ---------------------------------------------------------------------------

def format_adversarial_exchange_as_tier2(
    passage: str,
    connection: str,
    challenge: str,
    resolution: str,
) -> dict:
    """
    Format an adversarial exchange as a tier 2 ChatML training sample.

    The user turn contains the original passage plus the adversarial
    challenge as context.  The assistant turn is the resolution.
    """
    user_content = (
        f"{passage}\n\n"
        f"[ADVERSARIAL CHALLENGE]\n{challenge}"
    )
    return {
        "messages": [
            {"role": "system", "content": TIER2_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": resolution},
        ],
        "tier": 2,
        "exchange_type": "adversarial",
        "connection_challenged": connection,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adversarial challenges and format as tier 2 training data."
    )
    parser.add_argument(
        "--connections",
        default="data/dream_connections.jsonl",
        help="Path to dream_connections.jsonl from dream_state.py.",
    )
    parser.add_argument(
        "--adversary_model",
        required=True,
        help="Path or HF model ID for the adversary model.",
    )
    parser.add_argument(
        "--resolution_model",
        default=None,
        help="Path or HF model ID for the resolution model. "
             "Defaults to the same as adversary_model.",
    )
    parser.add_argument(
        "--output",
        default="data/adversarial.jsonl",
        help="Path to write formatted tier 2 adversarial training data.",
    )
    parser.add_argument(
        "--max_exchanges",
        type=int,
        default=None,
        help="Maximum number of exchanges to process.",
    )
    parser.add_argument(
        "--max_connections_per_sample",
        type=int,
        default=2,
        help="Maximum number of connections to challenge per source sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    connections_path = Path(args.connections)
    if not connections_path.exists():
        logger.error("Connections file not found: %s", connections_path)
        return

    records = []
    with open(connections_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d connection records", len(records))

    adv_pipe = load_adversary_model(args.adversary_model)
    resolution_model_path = args.resolution_model or args.adversary_model
    res_pipe = (
        load_adversary_model(resolution_model_path)
        if resolution_model_path != args.adversary_model
        else adv_pipe
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_exchanges = 0
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for idx, record in enumerate(records):
            passage = record.get("source_text", "")
            observe_probe = record.get("observe_probe", "")
            connections = record.get("connections", [])

            if not connections:
                continue

            # Limit connections per sample
            connections = connections[: args.max_connections_per_sample]

            for connection in connections:
                if args.max_exchanges is not None and total_exchanges >= args.max_exchanges:
                    break

                # Step 1: adversary challenges the connection
                adv_prompt = build_adversary_prompt(passage, observe_probe, connection)
                try:
                    challenge = generate_adversary_response(adv_pipe, adv_prompt)
                except Exception as exc:
                    logger.warning(
                        "Adversary generation failed for record %d: %s", idx, exc
                    )
                    continue

                # Step 2: generate resolution (tier 2 OBSERVE/PROBE)
                res_prompt = build_resolution_prompt(passage, connection, challenge)
                try:
                    resolution = generate_adversary_response(
                        res_pipe,
                        res_prompt,
                        system=TIER2_SYSTEM,
                    )
                except Exception as exc:
                    logger.warning(
                        "Resolution generation failed for record %d: %s", idx, exc
                    )
                    continue

                sample = format_adversarial_exchange_as_tier2(
                    passage, connection, challenge, resolution
                )
                out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_exchanges += 1

            if (idx + 1) % 10 == 0:
                logger.info(
                    "  processed %d/%d records, %d exchanges so far",
                    idx + 1,
                    len(records),
                    total_exchanges,
                )

            if args.max_exchanges is not None and total_exchanges >= args.max_exchanges:
                break

    logger.info(
        "Wrote %d adversarial tier 2 samples to %s",
        total_exchanges,
        output_path,
    )


if __name__ == "__main__":
    main()
