"""
generate_assistant_turns.py — Generate OBSERVE/PROBE assistant turns for raw
source documents using a capable LLM (local 14B-Instruct or API).

Usage
-----
  # Using the local Qwen2.5-14B-Instruct model:
  python generate_assistant_turns.py \\
      --source_dir data/raw/ \\
      --output data/tier1.jsonl \\
      --tier 1 \\
      --backend local \\
      --model_name Qwen/Qwen2.5-14B-Instruct

  # Using OpenAI API:
  python generate_assistant_turns.py \\
      --source_dir data/raw/ \\
      --output data/tier1.jsonl \\
      --tier 1 \\
      --backend openai \\
      --model_name gpt-4o

  # Using Anthropic Claude API:
  python generate_assistant_turns.py \\
      --source_dir data/raw/ \\
      --output data/tier1.jsonl \\
      --tier 1 \\
      --backend anthropic \\
      --model_name claude-opus-4-5

Each raw source document in *source_dir* must be a plain-text (.txt) file.
The script chunks long documents into segments ≤ max_chunk_tokens and
generates one training sample per chunk.

Output JSONL schema (one JSON object per line):
  {
    "tier": <int>,
    "prior": <str>,
    "domain": <str>,
    "source_type": <str>,
    "system": <tier_system_prompt>,
    "user": <raw_source_chunk>,
    "assistant": <generated_observe_probe_turn>
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from pathlib import Path

from data_utils import TIER_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

# ── Source metadata map ────────────────────────────────────────────────────────

SOURCE_METADATA: dict[str, dict[str, str]] = {
    "darwin": {"prior": "natural_selection", "domain": "biology,evolution", "source_type": "correspondence"},
    "plato": {"prior": "socratic_method", "domain": "philosophy,epistemology", "source_type": "dialogue"},
    "galileo": {"prior": "empirical_observation", "domain": "physics,mechanics", "source_type": "treatise"},
    "euclid": {"prior": "axiomatic_reasoning", "domain": "mathematics,geometry", "source_type": "proof"},
    "noether": {"prior": "symmetry_conservation", "domain": "mathematics,physics", "source_type": "paper"},
    "feynman": {"prior": "physical_intuition", "domain": "physics,pedagogy", "source_type": "lecture"},
    "clark_chalmers": {"prior": "extended_cognition", "domain": "philosophy,cognitive_science", "source_type": "paper"},
    "darcy_thompson": {"prior": "mathematical_form", "domain": "biology,mathematics", "source_type": "book"},
}

# ── Generation prompt ──────────────────────────────────────────────────────────

_GENERATION_SYSTEM = (
    "You are an expert at generating structured reasoning annotations. "
    "Given a passage of text and a tier context, produce EXACTLY this format "
    "for the assistant turn — no additional prose, no explanations:\n\n"
    "[TIER:{tier}] [PRIOR:{prior}] [DOMAIN:{domain}] [SOURCE:{source_type}]\n"
    "[OBSERVE] <2-4 sentences identifying structural features, constraints, "
    "relationships, and what the author is DOING — not just saying. Focus on "
    "the reasoning moves and logical architecture.>\n"
    "[PROBE] <2-4 questions that follow from observation. At least one must be "
    "cross-domain, one must identify where the framework breaks or has limits, "
    "and one must suggest a test or verification that could confirm or refute "
    "the observation.>"
)

_GENERATION_USER_TEMPLATE = (
    "TIER {tier} CONTEXT:\n{tier_system}\n\n"
    "PASSAGE:\n{passage}\n\n"
    "Generate the assistant turn now. Use exactly the format specified. "
    "PRIOR={prior}, DOMAIN={domain}, SOURCE={source_type}."
)


# ── Text chunking ─────────────────────────────────────────────────────────────


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Split *text* into overlapping chunks of at most *max_chars* characters."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            # Keep last paragraph for overlap
            tail = current[-1] if len(current[-1]) <= overlap else current[-1][-overlap:]
            current = [tail, para]
            current_len = len(tail) + para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ── Backend interfaces ─────────────────────────────────────────────────────────


def generate_with_local(
    passage: str,
    tier: int,
    metadata: dict[str, str],
    model_name: str,
    tokenizer=None,
    model=None,
    max_new_tokens: int = 512,
) -> str:
    """Generate assistant turn using a locally loaded model."""
    import torch

    system_prompt = _GENERATION_SYSTEM.format(**metadata, tier=tier)
    user_prompt = _GENERATION_USER_TEMPLATE.format(
        tier=tier,
        tier_system=TIER_SYSTEM_PROMPTS[tier],
        passage=passage,
        **metadata,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


def generate_with_openai(
    passage: str,
    tier: int,
    metadata: dict[str, str],
    model_name: str,
    client=None,
    max_tokens: int = 512,
) -> str:
    """Generate assistant turn using the OpenAI API."""
    system_prompt = _GENERATION_SYSTEM.format(**metadata, tier=tier)
    user_prompt = _GENERATION_USER_TEMPLATE.format(
        tier=tier,
        tier_system=TIER_SYSTEM_PROMPTS[tier],
        passage=passage,
        **metadata,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_with_anthropic(
    passage: str,
    tier: int,
    metadata: dict[str, str],
    model_name: str,
    client=None,
    max_tokens: int = 512,
) -> str:
    """Generate assistant turn using the Anthropic API."""
    system_prompt = _GENERATION_SYSTEM.format(**metadata, tier=tier)
    user_prompt = _GENERATION_USER_TEMPLATE.format(
        tier=tier,
        tier_system=TIER_SYSTEM_PROMPTS[tier],
        passage=passage,
        **metadata,
    )
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.3,
    )
    return response.content[0].text.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────


def infer_metadata(filename: str) -> dict[str, str]:
    """Infer source metadata from filename stem."""
    stem = Path(filename).stem.lower()
    for key, meta in SOURCE_METADATA.items():
        if key in stem:
            return meta
    return {"prior": "general_reasoning", "domain": "general", "source_type": "text"}


def process_file(
    source_path: Path,
    tier: int,
    generate_fn,
    max_chunk_chars: int = 6000,
) -> list[dict]:
    """Process a single source file and return list of training samples."""
    text = source_path.read_text(encoding="utf-8", errors="replace")
    metadata = infer_metadata(source_path.name)
    chunks = chunk_text(text, max_chars=max_chunk_chars)
    samples: list[dict] = []
    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d of %s", i + 1, len(chunks), source_path.name)
        try:
            assistant_turn = generate_fn(
                passage=chunk,
                tier=tier,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to generate for chunk %d of %s: %s", i + 1, source_path.name, exc)
            continue
        samples.append({
            "tier": tier,
            "prior": metadata["prior"],
            "domain": metadata["domain"],
            "source_type": metadata["source_type"],
            "system": TIER_SYSTEM_PROMPTS[tier],
            "user": chunk,
            "assistant": assistant_turn,
        })
    return samples


def main(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build generator function ───────────────────────────────────────────
    if args.backend == "local":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading local model %s", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        def generate_fn(passage, tier, metadata):
            return generate_with_local(
                passage, tier, metadata, args.model_name,
                tokenizer=tokenizer, model=model,
            )

    elif args.backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        def generate_fn(passage, tier, metadata):
            return generate_with_openai(passage, tier, metadata, args.model_name, client=client)

    elif args.backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def generate_fn(passage, tier, metadata):
            return generate_with_anthropic(passage, tier, metadata, args.model_name, client=client)

    else:
        logger.error("Unknown backend: %s", args.backend)
        sys.exit(1)

    # ── Process all source files ──────────────────────────────────────────
    source_files = sorted(source_dir.glob("*.txt"))
    if not source_files:
        logger.error("No .txt files found in %s", source_dir)
        sys.exit(1)

    logger.info("Found %d source files, tier=%d", len(source_files), args.tier)
    total_samples = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for src_path in source_files:
            logger.info("Processing %s", src_path.name)
            samples = process_file(src_path, args.tier, generate_fn, args.max_chunk_chars)
            for sample in samples:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total_samples += len(samples)
            logger.info("  → %d samples from %s", len(samples), src_path.name)

    logger.info("Done. Wrote %d samples to %s", total_samples, output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OBSERVE/PROBE assistant turns for training data"
    )
    parser.add_argument("--source_dir", required=True, help="Directory of .txt source files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1, help="Training tier")
    parser.add_argument(
        "--backend",
        choices=["local", "openai", "anthropic"],
        default="local",
        help="LLM backend to use for generation",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Model name/path for local backend, or API model name",
    )
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        default=6000,
        help="Maximum characters per text chunk",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    main(parse_args())
