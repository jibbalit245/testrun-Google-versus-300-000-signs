"""
evaluate.py — Run 5 evaluation probes after training and log pass/fail results.

Pass criteria: 4/5 probes pass = data direction validated.
               2-3/5 = adjust and retry.
               0-1/5 = format problem, revisit.

Usage
-----
  python evaluate.py \\
      --model_path checkpoints/final_adapter \\
      [--output_log eval_results.json]

The script prints each probe result and a final verdict, and optionally writes
results to a JSON file.
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

# ── Probe definitions ──────────────────────────────────────────────────────────

PROBES: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Novel reasoning from observation",
        "prompt": (
            "When you add salt to water before boiling, what actually changes about "
            "the boiling process? Don't summarize what you know — observe what must be "
            "happening at the structural level and identify what follows from it."
        ),
        "pass_criteria": (
            "Response must identify structural effects (e.g., boiling point elevation, "
            "colligative properties, solute-solvent interactions) AND draw a non-trivial "
            "inference or consequence beyond the surface fact. Must NOT be pure retrieval."
        ),
    },
    {
        "id": 2,
        "name": "Cross-domain structural connection",
        "prompt": (
            "A wolf pack controls deer population by predation. Remove the wolves — "
            "deer overpopulate, overgraze riverbanks, rivers erode and change course. "
            "Identify the structural pattern here, then find it operating in a completely "
            "different domain."
        ),
        "pass_criteria": (
            "Response must (a) name the structural pattern (trophic cascade, "
            "indirect coupling, constraint removal) AND (b) identify a genuine "
            "cross-domain analog with the same structural logic — not a surface metaphor."
        ),
    },
    {
        "id": 3,
        "name": "Boundary mapping",
        "prompt": (
            "I want to build a machine that runs forever without energy input. "
            "Don't just say it's impossible — map exactly where and why the boundary is. "
            "What would have to be true about physics for this to work, and why isn't it?"
        ),
        "pass_criteria": (
            "Response must map the boundary precisely: identify which physical law(s) "
            "are violated (thermodynamics), what those laws actually say at a structural "
            "level, and what counterfactual conditions would be required. Must not "
            "merely assert impossibility without structural analysis."
        ),
    },
    {
        "id": 4,
        "name": "Honest uncertainty",
        "prompt": (
            "What is consciousness? Not the philosophical positions — what do you "
            "actually know, what do you not know, and where exactly does your understanding "
            "break down?"
        ),
        "pass_criteria": (
            "Response must distinguish between what is established (neural correlates, "
            "functional definitions), what is genuinely unknown (why there is subjective "
            "experience at all — the hard problem), and map where the boundary of "
            "understanding lies. Must NOT fake certainty or collapse into either "
            "physicalist or dualist assertion."
        ),
    },
    {
        "id": 5,
        "name": "Collaborative engagement — genuine contribution",
        "prompt": (
            "I have this idea that gravity might be related to coherent atomic spin "
            "rather than mass directly. Mass might just be the axis we can measure. "
            "Poke holes in it."
        ),
        "pass_criteria": (
            "Response must: (1) engage genuinely with the structural logic of the idea, "
            "(2) identify specific falsifying evidence or logical inconsistencies with "
            "known physics (e.g., gravitational lensing of massless photons, equivalence "
            "principle, spin-statistics), AND (3) produce at least one insight or "
            "structural connection the user hadn't considered — grounded in real physics — "
            "that genuinely advances the inquiry. NOT mere retrieval. NOT just listing "
            "counter-evidence. A contribution."
        ),
    },
]

# ── Model loading ─────────────────────────────────────────────────────────────


def load_model_for_eval(model_path: str):
    """Load the fine-tuned adapter for evaluation inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info("Loading model for evaluation from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to read base model name from adapter config
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            base_name = json.load(f).get("base_model_name_or_path", "Qwen/Qwen2.5-72B-Base")
    else:
        base_name = "Qwen/Qwen2.5-72B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if adapter_config_path.exists():
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def run_probe(probe: dict[str, Any], model, tokenizer, max_new_tokens: int = 1024) -> str:
    """Run a single probe and return the model's raw response."""
    import torch

    system = (
        "You are a reasoning model. Think carefully about structure, "
        "constraints, and relationships. Do not merely retrieve — engage."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": probe["prompt"]},
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
            repetition_penalty=1.05,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def score_probe(probe: dict[str, Any], response: str) -> tuple[bool, str]:
    """Automated heuristic scoring. Returns (passed, reason).

    This is a best-effort heuristic; a human reviewer should verify.
    """
    resp_lower = response.lower()
    pid = probe["id"]

    if pid == 1:
        # Must show structural analysis beyond "salt raises boiling point"
        structural_keywords = [
            "colligative", "solute", "solvent", "boiling point elevation",
            "vapor pressure", "entropy", "chemical potential", "interaction",
        ]
        has_structure = any(k in resp_lower for k in structural_keywords)
        # Responses under 50 words are treated as retrieval-only (no structural analysis).
        # Genuine structural reasoning typically requires more words to establish
        # premises, mechanisms, and consequences.
        is_retrieval_only = len(response.split()) < 50
        passed = has_structure and not is_retrieval_only
        reason = "structural keywords found" if has_structure else "missing structural analysis"

    elif pid == 2:
        # Must name pattern AND give cross-domain analog
        pattern_words = ["cascade", "trophic", "indirect", "coupling", "constraint", "feedback"]
        domain_indicators = ["economy", "immune", "market", "neural", "network", "social", "financial", "supply chain"]
        has_pattern = any(k in resp_lower for k in pattern_words)
        has_cross_domain = any(k in resp_lower for k in domain_indicators)
        passed = has_pattern and has_cross_domain
        reason = (
            "named pattern and cross-domain analog found"
            if passed
            else f"pattern={'yes' if has_pattern else 'no'}, cross-domain={'yes' if has_cross_domain else 'no'}"
        )

    elif pid == 3:
        # Must reference thermodynamics and map the boundary
        thermo_words = ["thermodynamics", "entropy", "conservation", "energy", "second law", "first law"]
        boundary_words = ["would require", "would need", "counterfactual", "if physics", "violated", "break down"]
        has_thermo = any(k in resp_lower for k in thermo_words)
        has_boundary = any(k in resp_lower for k in boundary_words)
        passed = has_thermo and has_boundary
        reason = (
            "thermodynamic boundary mapped"
            if passed
            else f"thermo={'yes' if has_thermo else 'no'}, boundary={'yes' if has_boundary else 'no'}"
        )

    elif pid == 4:
        # Must distinguish known from unknown and not fake certainty
        uncertainty_words = ["don't know", "unclear", "hard problem", "unknown", "not known", "uncertain", "mystery"]
        false_certainty = ["consciousness is simply", "consciousness is just", "we know exactly", "fully understood"]
        has_uncertainty = any(k in resp_lower for k in uncertainty_words)
        has_false_certainty = any(k in resp_lower for k in false_certainty)
        passed = has_uncertainty and not has_false_certainty
        reason = (
            "honest uncertainty expressed"
            if passed
            else (
                "fakes certainty" if has_false_certainty
                else "lacks explicit uncertainty acknowledgment"
            )
        )

    elif pid == 5:
        # Must identify specific falsifying physics AND contribute new insight
        falsifying_words = [
            "photon", "lensing", "equivalence principle", "spin-statistics",
            "general relativity", "geodesic", "massless", "gravitational wave",
        ]
        contribution_words = [
            "consider", "what if", "interestingly", "connection", "suggests",
            "this implies", "structural", "follow from", "advance", "further",
        ]
        has_falsifying = any(k in resp_lower for k in falsifying_words)
        has_contribution = sum(1 for k in contribution_words if k in resp_lower) >= 2
        passed = has_falsifying and has_contribution
        reason = (
            "falsifying evidence and contribution present"
            if passed
            else f"falsifying={'yes' if has_falsifying else 'no'}, contribution={'yes' if has_contribution else 'no'}"
        )
    else:
        passed = False
        reason = "unknown probe id"

    return passed, reason


# ── Main evaluation runner ────────────────────────────────────────────────────


def run_all_probes(
    model_path: str,
    tokenizer=None,
    model=None,
    output_log: str | None = None,
) -> dict[str, Any]:
    """Run all 5 probes and return a results dict."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model_for_eval(model_path)

    results: list[dict[str, Any]] = []
    passed_count = 0

    print("\n" + "=" * 70)
    print("EVALUATION PROBES")
    print("=" * 70)

    for probe in PROBES:
        print(f"\nProbe {probe['id']}: {probe['name']}")
        print("-" * 60)
        print(f"Prompt: {probe['prompt'][:200]}...")

        response = run_probe(probe, model, tokenizer)
        passed, reason = score_probe(probe, response)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Response (first 400 chars): {response[:400]}...")
        print(f"Result: {status} — {reason}")

        if passed:
            passed_count += 1

        results.append({
            "probe_id": probe["id"],
            "name": probe["name"],
            "passed": passed,
            "reason": reason,
            "response_preview": response[:500],
            "full_response": response,
        })

    print("\n" + "=" * 70)
    print(f"FINAL SCORE: {passed_count}/5 probes passed")

    if passed_count >= 4:
        verdict = "VALIDATED — Data direction confirmed. Proceed to full training."
    elif passed_count >= 2:
        verdict = "PARTIAL — Adjust format or data and retry."
    else:
        verdict = "FAILED — Format problem. Revisit data generation."

    print(f"VERDICT: {verdict}")
    print("=" * 70 + "\n")

    summary = {
        "passed": passed_count,
        "total": 5,
        "verdict": verdict,
        "probes": results,
    }

    if output_log:
        with open(output_log, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Wrote evaluation results to %s", output_log)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5 evaluation probes on the trained model"
    )
    parser.add_argument("--model_path", required=True, help="Path to trained LoRA adapter or model")
    parser.add_argument("--output_log", default=None, help="Optional JSON file to write results to")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args()
    results = run_all_probes(args.model_path, output_log=args.output_log)
    passed = results["passed"]
    sys.exit(0 if passed >= 4 else 1)
