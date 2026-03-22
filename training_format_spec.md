# Training Format Specification

## Overview

This document specifies the exact training data format for the LoRA validation
run on Qwen 2.5 72B Base. All training samples **must** conform to this format.

---

## ChatML Token Convention

Use **Qwen native ChatML**:

```
<|im_start|>  and  <|im_end|>
```

**Not** `<|system|>` / `<|end|>` — these are incorrect for Qwen2.

---

## Sample Structure

Every training sample has exactly this structure:

```
<|im_start|>system
{tier_system_prompt}<|im_end|>
<|im_start|>user
{raw_source_content}<|im_end|>
<|im_start|>assistant
[TIER:{n}] [PRIOR:{prior_name}] [DOMAIN:{domains}] [SOURCE:{type}]
[OBSERVE] {2-4 sentences identifying structural features, constraints,
relationships, and what the author is DOING not just saying}
[PROBE] {2-4 questions that follow from observation — at least one
cross-domain, one identifying where framework breaks, one suggesting
a test or verification}<|im_end|>
```

---

## Loss Masking

**Loss is computed ONLY on the assistant turn.**

Set `train_on_input: false`. The model learns to PRODUCE the OBSERVE/PROBE
reasoning engagement, not to memorize the source content.

In `data_utils.py`, `tokenise_and_mask()` sets `labels = -100` for all tokens
in the system and user turns.

---

## Tier System Prompts (Tiers 1–3, Validation Run)

### Tier 1

> You are learning to see. Every passage contains structure — constraints that
> define what is possible, relationships that reveal what connects, and gaps
> that show where understanding breaks down. Your task is not to summarize or
> retrieve. Your task is to observe what is actually present, identify what
> follows from it, and ask the question that opens it further. When you
> encounter a boundary, map it. The boundary defines the shape of what lies
> beyond it.

### Tier 2

> You have axioms. Now derive from them. Every passage builds on principles
> you already hold. Your task is to follow the chain of reasoning — from
> premise through logic to conclusion — and verify that each step holds. When
> a derivation skips a step, identify what was assumed. When a proof reaches
> its conclusion, ask what else follows from the same premises that the author
> did not pursue.

### Tier 3

> You have principles and derivations. Now apply them. Every passage puts
> theory into contact with reality. Your task is to observe where the model
> fits the world and where it doesn't. When application succeeds, note what
> made the mapping work. When it fails, map exactly where the abstraction
> breaks against the concrete. The gap between model and reality is where the
> next insight lives.

*(The full 10-tier prompt set is used for production training beyond this
validation run.)*

---

## Assistant Turn Tags

All tags are **required** and must appear in order:

| Tag | Format | Description |
|-----|--------|-------------|
| `[TIER:n]` | `[TIER:1]`, `[TIER:2]`, `[TIER:3]` | Training tier |
| `[PRIOR:name]` | e.g., `[PRIOR:natural_selection]` | Knowledge prior being activated |
| `[DOMAIN:...]` | e.g., `[DOMAIN:biology,evolution]` | One or more domains |
| `[SOURCE:type]` | e.g., `[SOURCE:correspondence]` | Type of source material |
| `[OBSERVE]` | 2–4 sentences | Structural observation (what the author is DOING) |
| `[PROBE]` | 2–4 questions | At least one cross-domain, one boundary, one test/verification |

---

## OBSERVE Requirements

- Identify **structural features**: constraints, axioms, logical dependencies
- Identify **relationships**: what connects to what, and how
- Identify **gaps**: where understanding breaks down or is assumed
- Describe what the author is **DOING** (logical moves) — not just saying
- **Not** a summary or retrieval

## PROBE Requirements

Minimum 3 of 4 question types must be present:

1. **Cross-domain**: "What other domain exhibits this same structural pattern?"
2. **Boundary**: "Where does this framework break down or fail?"
3. **Test/verification**: "What experiment or observation would confirm or
   refute this?"
4. **Follow-on derivation**: "What else follows from the same premises that
   the author did not pursue?"

---

## Source Data for Validation Run

Each source maps to metadata used in assistant turn tags:

| Source | `prior` | `domain` | `source_type` |
|--------|---------|----------|---------------|
| Darwin correspondence | `natural_selection` | `biology,evolution` | `correspondence` |
| Plato (Meno, Theaetetus, Republic) | `socratic_method` | `philosophy,epistemology` | `dialogue` |
| Galileo Two New Sciences | `empirical_observation` | `physics,mechanics` | `treatise` |
| Euclid Elements | `axiomatic_reasoning` | `mathematics,geometry` | `proof` |
| Noether's theorem papers | `symmetry_conservation` | `mathematics,physics` | `paper` |
| Feynman Lectures (selected) | `physical_intuition` | `physics,pedagogy` | `lecture` |
| Clark & Chalmers "Extended Mind" | `extended_cognition` | `philosophy,cognitive_science` | `paper` |
| D'Arcy Thompson "On Growth and Form" | `mathematical_form` | `biology,mathematics` | `book` |

---

## JSONL Schema

Each line in a training JSONL file must be a valid JSON object with these keys:

```json
{
  "tier": 1,
  "prior": "natural_selection",
  "domain": "biology,evolution",
  "source_type": "correspondence",
  "system": "<exact tier system prompt from above>",
  "user": "<raw source passage>",
  "assistant": "[TIER:1] [PRIOR:natural_selection] ..."
}
```

---

## Adversarial Exchange Format

Adversarial samples (tier 2, formatted by `adversary_challenge.py`) use
`source_type: adversarial_exchange` and have the user turn structured as:

```
ORIGINAL PASSAGE:
{passage}

ADVERSARY CHALLENGE:
{challenge question}
```

These are merged with standard tier-2 training data before the tier-2 training
run.

---

## Validation

`data_utils.validate_sample()` checks:
1. Required keys present (`system`, `user`, `assistant`, `tier`)
2. Tier is 1, 2, or 3
3. System prompt matches the canonical tier prompt exactly
4. Assistant turn contains all required tags in the correct order

Run validation before training:
```bash
python -c "
from data_utils import load_and_validate_jsonl
samples = load_and_validate_jsonl('data/tier1.jsonl', strict=True)
print(f'{len(samples)} valid samples')
"
```
