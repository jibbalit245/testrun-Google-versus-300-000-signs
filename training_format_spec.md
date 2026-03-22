# Training Format Specification

## Overview

This document specifies the exact data format used by the LoRA validation
training pipeline for Qwen 2.5 72B (base model).

The objective of this training run is to validate that a specific data format
produces **genuine reasoning behaviour** rather than retrieval or
summarization.

---

## Sample Structure

Every training sample follows this exact structure using Qwen native ChatML
tokens (`<|im_start|>` / `<|im_end|>`):

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

**Critical**: Loss is computed **only** on the assistant turn
(`train_on_input: false`).  The model learns to **produce** the
OBSERVE/PROBE reasoning engagement — it does not memorize source content.

---

## Tokens

Use **Qwen native ChatML** tokens:

| Token | Value |
|-------|-------|
| Turn start | `<\|im_start\|>` (literal: `<|im_start|>`) |
| Turn end   | `<\|im_end\|>` (literal: `<|im_end|>`) |

Do **not** use `<|system|>` / `<|end|>` or other token styles.

---

## Tier System Prompts

### Tier 1 — Observation

> You are learning to see. Every passage contains structure —
> constraints that define what is possible, relationships that reveal
> what connects, and gaps that show where understanding breaks down.
> Your task is not to summarize or retrieve. Your task is to observe
> what is actually present, identify what follows from it, and ask the
> question that opens it further. When you encounter a boundary, map it.
> The boundary defines the shape of what lies beyond it.

### Tier 2 — Derivation

> You have axioms. Now derive from them. Every passage builds
> on principles you already hold. Your task is to follow the chain of
> reasoning — from premise through logic to conclusion — and verify that
> each step holds. When a derivation skips a step, identify what was
> assumed. When a proof reaches its conclusion, ask what else follows
> from the same premises that the author did not pursue.

### Tier 3 — Application

> You have principles and derivations. Now apply them. Every
> passage puts theory into contact with reality. Your task is to observe
> where the model fits the world and where it doesn't. When application
> succeeds, note what made the mapping work. When it fails, map exactly
> where the abstraction breaks against the concrete. The gap between
> model and reality is where the next insight lives.

*(Tiers 1–3 are used for the validation run.  Full 10-tier prompts are
defined in the pipeline implementation.)*

---

## Assistant Turn Fields

| Field | Description |
|-------|-------------|
| `[TIER:{n}]` | Tier number (1–10) |
| `[PRIOR:{prior_name}]` | Author or intellectual tradition (e.g. `Darwin`, `Euclid`, `Adversarial`) |
| `[DOMAIN:{domains}]` | Comma-separated domain tags (e.g. `biology,evolution`) |
| `[SOURCE:{type}]` | Document type: `letter`, `dialogue`, `treatise`, `paper`, `lecture`, `exchange` |
| `[OBSERVE]` | 2–4 sentences: structural features, constraints, relationships, and what the author **does** |
| `[PROBE]` | 2–4 questions: at minimum one cross-domain, one boundary-locating, one test/verification |

---

## JSONL Format

Training data is stored as newline-delimited JSON.  Each line is a dict
in the `messages` format understood by the Qwen tokenizer's
`apply_chat_template`:

```json
{
  "messages": [
    {"role": "system",    "content": "<tier_system_prompt>"},
    {"role": "user",      "content": "<raw_source_passage>"},
    {"role": "assistant", "content": "[TIER:1] [PRIOR:Darwin] ..."}
  ],
  "tier": 1
}
```

Alternatively, samples may use the pre-formatted `text` field containing
the full ChatML string — this is also accepted by the pipeline.

---

## Data Sources (Validation Run)

The ~3–5 M token validation subset draws from:

| Source | Domain(s) | Type |
|--------|-----------|------|
| Darwin correspondence (reasoning passages) | biology, evolution | letter |
| Plato — Meno, Theaetetus, Republic | philosophy, epistemology | dialogue |
| Galileo — Two New Sciences | physics, mechanics | treatise |
| Euclid — Elements | mathematics, geometry | treatise |
| Noether's theorem papers | mathematics, physics | paper |
| Feynman Lectures (selected chapters) | physics, pedagogy | lecture |
| Clark & Chalmers — "Extended Mind" | philosophy, cognition | paper |
| D'Arcy Thompson — "On Growth and Form" (selected) | biology, mathematics | treatise |

---

## Adversarial Data Format

Adversarial exchange samples are formatted identically to standard
training samples but use:

- `[TIER:2]` (always tier 2)
- `[PRIOR:Adversarial]`
- `[DOMAIN:cross-domain]`
- `[SOURCE:exchange]`
- User turn contains original passage + `[ADVERSARIAL CHALLENGE]` block
- Assistant turn is the resolution OBSERVE/PROBE

Adversarial JSONL files are merged with standard JSONL at training time
via `--adversarial_data` flag.

---

## Pipeline Files

| File | Purpose |
|------|---------|
| `generate_training_data.py` | Generate OBSERVE/PROBE from source documents |
| `dream_state.py` | Extract stated connections from tier 1 data |
| `adversary_challenge.py` | Run adversarial exchanges; output tier 2 data |
| `train.py` | Main training script |
| `fsdp_config.yaml` | Accelerate FSDP configuration |
| `evaluate.py` | 5 post-training evaluation probes |
| `launch.sh` | Venv + install + training launch |

---

## QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| FSDP strategy | Full shard |
| FSDP wrap unit | `Qwen2DecoderLayer` |
| Sequence packing | Enabled |
| Training epochs | 3 |
| LR schedule | Cosine |
| Warmup ratio | 3% |
| Checkpoint interval | 200 steps |

---

## Evaluation Pass Criteria

| Score | Verdict |
|-------|---------|
| 4–5 / 5 probes pass | **Validated** — data direction confirmed |
| 2–3 / 5 probes pass | **Marginal** — adjust training and retry |
| 0–1 / 5 probes pass | **Format problem** — revisit data format |
