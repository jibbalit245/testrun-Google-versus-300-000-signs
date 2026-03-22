#!/usr/bin/env python3
"""
evaluate.py — Post-training evaluation probes.

Runs 5 structured probes against the trained model and logs pass/fail.

Pass criteria:
  4/5 pass  → data direction validated
  2-3/5     → adjust and retry
  0-1/5     → format problem, revisit

Usage (standalone):
    python evaluate.py \\
        --model_dir ./checkpoints/final \\
        --output eval_results.json

Or imported by train.py:
    from evaluate import run_evaluation_probes
    results = run_evaluation_probes(model, tokenizer)
"""

import argparse
import json
import logging
import re
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
# Probe definitions
# ---------------------------------------------------------------------------

PROBES: List[Dict] = [
    {
        "id": 1,
        "name": "novel_reasoning_from_observation",
        "description": "Novel reasoning from observation — salt in boiling water.",
        "prompt": (
            "When you add salt to water, does it boil faster or slower? "
            "Don't just retrieve the answer. Observe the structure of what's "
            "happening and reason through it."
        ),
        "pass_criteria": [
            # Model should reason about boiling point elevation (colligative property)
            # and distinguish between initial heat transfer effect vs. steady-state
            # It should identify the structural tension: salt raises boiling point
            # (requires MORE energy) but also affects nucleation
            r"boiling point",
            r"elevation|higher|raises|increase",
        ],
        "fail_criteria": [
            # Trivial retrieval without reasoning
            r"^(yes|no|faster|slower)[\.\s]*$",
        ],
        "pass_logic": "any_required_any_forbidden",
    },
    {
        "id": 2,
        "name": "cross_domain_structural_connection",
        "description": "Cross-domain structural connection — predator removal analogy.",
        "prompt": (
            "A wildlife manager removes all wolves from Yellowstone to protect "
            "elk herds. Ten years later, the river channels have shifted and "
            "riparian vegetation has collapsed. What structural pattern does this "
            "instantiate, and where else does the same pattern appear?"
        ),
        "pass_criteria": [
            # Should identify trophic cascade / keystone species
            # AND provide at least one structural analogue from another domain
            r"cascade|trophic|keystone|indirect",
            r"(bank|financial|economy|market|regulation|immune|system|network)",
        ],
        "fail_criteria": [],
        "pass_logic": "all_required",
    },
    {
        "id": 3,
        "name": "boundary_mapping",
        "description": "Boundary mapping — perpetual motion.",
        "prompt": (
            "Someone claims they've built a perpetual motion machine. "
            "Rather than dismissing it, map the boundary: what exactly would "
            "have to be true for it to work, and what does that require us to "
            "revise about our current understanding?"
        ),
        "pass_criteria": [
            # Should identify thermodynamic laws as the constraint
            # Should map what revision would entail (entropy, conservation laws)
            r"entropy|thermodynamic|conservation|energy",
            r"(would require|revision|revise|break|violate|constraint)",
        ],
        "fail_criteria": [
            # Should NOT just say "it's impossible" without mapping the boundary
            r"^(it.{0,20}impossible|can.{0,10}t work|doesn.{0,10}t exist)[\.\s]*$",
        ],
        "pass_logic": "any_required_any_forbidden",
    },
    {
        "id": 4,
        "name": "honest_uncertainty",
        "description": "Honest uncertainty — consciousness.",
        "prompt": (
            "What is consciousness? Not a definition — what do we actually know "
            "about it structurally, and where exactly does our understanding break?"
        ),
        "pass_criteria": [
            # Should identify what IS known (neural correlates, binding problem, etc.)
            # AND identify where understanding breaks (hard problem, explanatory gap)
            r"(hard problem|explanatory gap|qualia|subjective|what it.{0,10}like)",
            r"(neural correlate|binding|integration|access consciousness|phenomenal)",
        ],
        "fail_criteria": [
            # Should NOT give a confident complete definition
            r"^consciousness is (the|a) (ability|process|state|function)",
        ],
        "pass_logic": "any_required_any_forbidden",
    },
    {
        "id": 5,
        "name": "collaborative_engagement",
        "description": (
            "Collaborative engagement — model must contribute a genuine insight, "
            "not retrieve or organize."
        ),
        "prompt": (
            "I have this idea that gravity might be related to coherent atomic "
            "spin rather than mass directly. Mass might just be the axis we can "
            "measure. Poke holes in it."
        ),
        "pass_criteria": [
            # Must engage seriously with the structural claim
            # Must identify a specific physical constraint or prediction that fails
            # Should NOT just say "this is wrong because Einstein/Newton"
            r"(spin|angular momentum|gyromagnetic|magnetic moment|coupling)",
            r"(predict|measurement|test|observation|evidence|experiment|falsif)",
            r"(equivalence principle|geodesic|spacetime|curvature|field equation"
            r"|graviton|weak force|gravitational wave)",
        ],
        "fail_criteria": [
            # Pure dismissal without engagement
            r"^(this is (not|incorrect|wrong)|gravity is (not|already|well))",
        ],
        "pass_logic": "any_required_any_forbidden",
    },
]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

EVAL_SYSTEM = (
    "You are learning to see. Every passage contains structure — "
    "constraints that define what is possible, relationships that reveal "
    "what connects, and gaps that show where understanding breaks down. "
    "Your task is not to summarize or retrieve. Your task is to observe "
    "what is actually present, identify what follows from it, and ask the "
    "question that opens it further. When you encounter a boundary, map it. "
    "The boundary defines the shape of what lies beyond it."
)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system: str = EVAL_SYSTEM,
    max_new_tokens: int = 600,
    temperature: float = 0.7,
) -> str:
    """Generate a response for an evaluation prompt."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
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
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Pass/fail evaluation
# ---------------------------------------------------------------------------

def evaluate_response(response: str, probe: Dict) -> Tuple[bool, str]:
    """
    Apply the probe's pass/fail criteria to a response.
    Returns (passed: bool, reason: str).
    """
    logic = probe.get("pass_logic", "any_required_any_forbidden")
    required = probe.get("pass_criteria", [])
    forbidden = probe.get("fail_criteria", [])
    lower = response.lower()

    required_matches = [
        bool(re.search(pattern, lower, re.IGNORECASE))
        for pattern in required
    ]
    forbidden_matches = [
        bool(re.search(pattern, response, re.IGNORECASE))
        for pattern in forbidden
    ]

    any_forbidden = any(forbidden_matches)
    if any_forbidden:
        idx = next(i for i, m in enumerate(forbidden_matches) if m)
        return False, f"Matched forbidden pattern: {forbidden[idx]!r}"

    if logic == "all_required":
        passed = all(required_matches)
        if not passed:
            missing = [
                required[i]
                for i, m in enumerate(required_matches)
                if not m
            ]
            return False, f"Missing required patterns: {missing}"
        return True, "All required patterns matched"
    else:  # any_required_any_forbidden
        passed = any(required_matches) if required else True
        if not passed:
            return False, f"No required patterns matched: {required}"
        return True, "Required pattern(s) matched, no forbidden patterns"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation_probes(model, tokenizer) -> Dict:
    """
    Run all 5 evaluation probes.  Returns a structured results dict.
    """
    probe_results = []
    total_passed = 0

    for probe in PROBES:
        logger.info("Running probe %d: %s", probe["id"], probe["name"])
        response = generate_response(model, tokenizer, probe["prompt"])
        passed, reason = evaluate_response(response, probe)
        total_passed += int(passed)

        result = {
            "id": probe["id"],
            "name": probe["name"],
            "description": probe["description"],
            "pass": passed,
            "reason": reason,
            "response_excerpt": response[:500] + ("…" if len(response) > 500 else ""),
        }
        probe_results.append(result)
        logger.info(
            "  Probe %d (%s): %s — %s",
            probe["id"],
            probe["name"],
            "PASS" if passed else "FAIL",
            reason,
        )

    if total_passed >= 4:
        verdict = "VALIDATED"
        verdict_msg = "Data direction validated (≥4/5 probes passed)."
    elif total_passed >= 2:
        verdict = "MARGINAL"
        verdict_msg = f"{total_passed}/5 probes passed — adjust and retry."
    else:
        verdict = "FORMAT_PROBLEM"
        verdict_msg = f"{total_passed}/5 probes passed — format problem, revisit."

    logger.info("Evaluation verdict: %s — %s", verdict, verdict_msg)

    return {
        "probes": probe_results,
        "total_passed": total_passed,
        "total_probes": len(PROBES),
        "verdict": verdict,
        "verdict_message": verdict_msg,
    }


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation probes on a trained model."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Path to write the evaluation results JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    logger.info("Loading model from %s", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    results = run_evaluation_probes(model, tokenizer)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Results written to %s", output_path)
    logger.info(
        "%d/%d probes passed — %s",
        results["total_passed"],
        results["total_probes"],
        results["verdict"],
    )


if __name__ == "__main__":
    main()
