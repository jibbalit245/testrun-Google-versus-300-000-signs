#!/usr/bin/env python3
"""
generate_training_data.py — Generate OBSERVE/PROBE assistant turns for
each source document, producing JSONL training data in the exact
ChatML format required by the pipeline.

A capable model (local 14B-Instruct, Claude, or GPT-4) is called to
produce the structured [OBSERVE]/[PROBE] reasoning output for each
extracted document chunk.

Usage:
    # Using Claude (set ANTHROPIC_API_KEY):
    python generate_training_data.py \\
        --source_dir data/sources \\
        --output_dir data \\
        --backend claude

    # Using OpenAI (set OPENAI_API_KEY):
    python generate_training_data.py \\
        --source_dir data/sources \\
        --output_dir data \\
        --backend openai --model gpt-4o

    # Using a local 14B-Instruct model:
    python generate_training_data.py \\
        --source_dir data/sources \\
        --output_dir data \\
        --backend local --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier system prompts (exact text from spec)
# ---------------------------------------------------------------------------

TIER_SYSTEM_PROMPTS: Dict[int, str] = {
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

# ---------------------------------------------------------------------------
# Source metadata: maps source name → (prior_name, domains, source_type)
# ---------------------------------------------------------------------------

SOURCE_METADATA: Dict[str, Tuple[str, str, str]] = {
    "darwin":        ("Darwin",        "biology,evolution,correspondence",    "letter"),
    "plato_meno":    ("Plato",         "philosophy,epistemology,dialogue",    "dialogue"),
    "plato_theaet":  ("Plato",         "philosophy,epistemology,dialogue",    "dialogue"),
    "plato_rep":     ("Plato",         "philosophy,politics,dialogue",        "dialogue"),
    "galileo":       ("Galileo",       "physics,mechanics,experiment",        "treatise"),
    "euclid":        ("Euclid",        "mathematics,geometry,proof",          "treatise"),
    "noether":       ("Noether",       "mathematics,physics,symmetry",        "paper"),
    "feynman":       ("Feynman",       "physics,pedagogy,lectures",           "lecture"),
    "clark_chalmers":("Clark+Chalmers","philosophy,cognition,mind",           "paper"),
    "thompson":      ("Thompson",      "biology,mathematics,morphology",      "treatise"),
}

# ---------------------------------------------------------------------------
# Generation prompt sent to the capable model
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = textwrap.dedent("""\
    You are an expert reasoning assistant. Your task is to generate the
    assistant turn of a training sample for a reasoning model.

    Given a passage, produce output in EXACTLY this format (no deviations):

    [TIER:{tier}] [PRIOR:{prior}] [DOMAIN:{domains}] [SOURCE:{source_type}]
    [OBSERVE] <2-4 sentences identifying structural features, constraints,
    relationships, and what the author is DOING not just saying>
    [PROBE] <2-4 questions that follow from the observation — at least one
    cross-domain, one identifying where the framework breaks, one suggesting
    a test or verification>

    Rules:
    - Do NOT summarize the passage. Identify structure and author intent.
    - [OBSERVE] must describe WHAT IS PRESENT structurally, not content summary.
    - [PROBE] questions must be generative — they open new inquiry.
    - At least one [PROBE] question must be cross-domain.
    - At least one [PROBE] question must identify where the framework breaks.
    - At least one [PROBE] question must suggest a test or verification.
    - Produce only the formatted output. No preamble. No explanation.
""")


def build_generation_prompt(
    passage: str,
    tier: int,
    prior: str,
    domains: str,
    source_type: str,
) -> str:
    return (
        f"Passage:\n{passage}\n\n"
        f"Generate the assistant turn for TIER:{tier}, "
        f"PRIOR:{prior}, DOMAIN:{domains}, SOURCE:{source_type}."
    )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def generate_with_claude(
    user_prompt: str,
    model: str = "claude-opus-4-5",
    max_tokens: int = 512,
) -> str:
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=GENERATION_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def generate_with_openai(
    user_prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 512,
) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


_local_pipeline = None


def generate_with_local(
    user_prompt: str,
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    max_tokens: int = 512,
) -> str:
    global _local_pipeline
    if _local_pipeline is None:
        import torch
        from transformers import pipeline
        logger.info("Loading local model %s …", model)
        _local_pipeline = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    out = _local_pipeline(
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    return out[0]["generated_text"][-1]["content"].strip()


# ---------------------------------------------------------------------------
# Passage chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 1500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            # keep last paragraph for overlap
            current = current[-1:] if overlap > 0 else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

def format_chatml_sample(
    tier: int,
    system_prompt: str,
    passage: str,
    assistant_turn: str,
) -> dict:
    """Return a sample dict with ``messages`` in ChatML structure."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage},
            {"role": "assistant", "content": assistant_turn},
        ],
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_source_file(
    filepath: Path,
    tier: int,
    generate_fn,
    retry_delay: float = 2.0,
    max_retries: int = 3,
) -> List[dict]:
    """Process one source file and return a list of formatted samples."""
    source_key = filepath.stem.lower()
    # Find matching metadata key (prefix match)
    meta_key = next(
        (k for k in SOURCE_METADATA if source_key.startswith(k)),
        None,
    )
    if meta_key is None:
        logger.warning(
            "No metadata for %s — using defaults.", filepath.name
        )
        prior, domains, source_type = "Unknown", "general", "text"
    else:
        prior, domains, source_type = SOURCE_METADATA[meta_key]

    text = filepath.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text)
    system_prompt = TIER_SYSTEM_PROMPTS[tier]
    samples: List[dict] = []

    for i, chunk in enumerate(chunks):
        user_prompt = build_generation_prompt(
            chunk, tier, prior, domains, source_type
        )
        assistant_turn = None
        for attempt in range(max_retries):
            try:
                assistant_turn = generate_fn(user_prompt)
                break
            except Exception as exc:
                logger.warning(
                    "Chunk %d/%d of %s — attempt %d failed: %s",
                    i + 1,
                    len(chunks),
                    filepath.name,
                    attempt + 1,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        if assistant_turn is None:
            logger.error(
                "Skipping chunk %d of %s after %d failures.",
                i + 1,
                filepath.name,
                max_retries,
            )
            continue

        sample = format_chatml_sample(tier, system_prompt, chunk, assistant_turn)
        samples.append(sample)
        logger.debug(
            "  chunk %d/%d → %d chars assistant turn",
            i + 1,
            len(chunks),
            len(assistant_turn),
        )

    logger.info(
        "Processed %s: %d/%d chunks succeeded.",
        filepath.name,
        len(samples),
        len(chunks),
    )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OBSERVE/PROBE training data from source documents."
    )
    parser.add_argument(
        "--source_dir",
        default="data/sources",
        help="Directory containing source .txt files, organised by tier "
             "(subdirs tier1/, tier2/, tier3/) or flat.",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Directory to write tierN.jsonl output files.",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "openai", "local"],
        default="claude",
        help="Which model backend to use for generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the chosen backend.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Which tiers to generate data for.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the generation function
    if args.backend == "claude":
        model = args.model or "claude-opus-4-5"
        def generate_fn(prompt):
            return generate_with_claude(prompt, model=model)
    elif args.backend == "openai":
        model = args.model or "gpt-4o"
        def generate_fn(prompt):
            return generate_with_openai(prompt, model=model)
    else:
        model = args.model or "Qwen/Qwen2.5-14B-Instruct"
        def generate_fn(prompt):
            return generate_with_local(prompt, model=model)

    for tier in args.tiers:
        tier_dir = source_dir / f"tier{tier}"
        if tier_dir.is_dir():
            files = sorted(tier_dir.glob("*.txt"))
        else:
            # flat layout: all txt files assigned to this tier
            files = sorted(source_dir.glob("*.txt"))

        if not files:
            logger.warning("No .txt files found for tier %d in %s", tier, source_dir)
            continue

        logger.info("Tier %d: processing %d files…", tier, len(files))
        all_samples: List[dict] = []
        for fp in files:
            samples = process_source_file(fp, tier, generate_fn)
            all_samples.extend(samples)

        out_path = output_dir / f"tier{tier}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for sample in all_samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(
            "Tier %d: wrote %d samples to %s", tier, len(all_samples), out_path
        )


if __name__ == "__main__":
    main()
