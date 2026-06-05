# agents.py

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from retriever import build_context_prompt
import json
import asyncio
import re
import random   # ← ADD THIS
import ollama

def call_ollama(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=messages,
        options={
            "temperature": temperature,
            "num_ctx": 12228,
            "num_predict": -1,
        },
        keep_alive="2h",
    )

    try:
        content = response.message.content
    except AttributeError:
        content = response["message"]["content"]

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


ollama_model = LiteLlm(model="ollama_chat/qwen2.5:7b")

_SCHEMA_CACHE: dict = {}

def _load_schema(schema_path: str = "schema.json") -> dict:
    if not _SCHEMA_CACHE:
        with open(schema_path) as f:
            _SCHEMA_CACHE.update(json.load(f))
    return _SCHEMA_CACHE


def _schema_to_constraints(schema: dict) -> str:
    lines = []
    for field, rules in schema.items():
        t = rules.get("type", "unknown")
        if rules.get("values") == [0, 1]:
            lines.append(f"  {field}: binary, must be 0 or 1")
        elif t == "categorical":
            vals = ", ".join(str(v) for v in rules.get("values", []))
            lines.append(f"  {field}: one of [{vals}]")
        else:
            lo, hi   = rules.get("min"), rules.get("max")
            nullable = " (nullable)" if rules.get("nullable") else ""
            # ↓ CHANGED: include p25/p75/mean so the model sees the distribution
            mean = rules.get("mean", "")
            p25  = rules.get("p25", "")
            p75  = rules.get("p75", "")
            lines.append(
                f"  {field}: {t}, MUST be between {lo} and {hi} (inclusive){nullable}"
                f" — mean={mean}, p25={p25}, p75={p75}"
            )
    return "\n".join(lines)


def build_generator_prompt(partial_row: dict, feedback: str = "", schema: dict | None = None) -> str:
    if schema is None:
        schema = _load_schema()

    remaining_fields = [f for f in schema if f not in partial_row]
    constraints      = _schema_to_constraints({f: schema[f] for f in remaining_fields})
    required_list    = json.dumps(remaining_fields)

    # ── DIVERSITY HINT ───────────────────────────────────────────────────────
    # Only hint at fields that haven't already been fixed by the caller.
    # If partial_row already pins vehicle_age/customer_age/subscription_length
    # (e.g. from a scenario seed) we must NOT suggest different random values.
    _hint_parts = []
    if "vehicle_age" not in partial_row:
        _hint_parts.append(f"vehicle_age around {random.randint(2, 60)} months")
    if "customer_age" not in partial_row:
        _hint_parts.append(f"customer_age around {random.randint(22, 70)}")
    if "subscription_length" not in partial_row:
        _hint_parts.append(f"subscription_length around {round(random.uniform(1, 24), 1)} months")
    diversity_hint = (
        ("For this specific row, bias toward: " + ", ".join(_hint_parts) + ".")
        if _hint_parts else ""
    )
    # ────────────────────────────────────────────────────────────────────────

    feedback_block = ""
    if feedback:
        feedback_block = f"""
=== Validator Feedback (Previous Attempt Rejected) ===
{feedback}
You MUST include ALL fields listed and fix these issues.
=== End Feedback ===
"""

    # ── FULL PROMPT WITH DIVERSITY INSTRUCTIONS ──────────────────────────────
    return f"""You are generating one row of an insurance claims dataset.
You MUST produce realistic variation — do NOT repeat the same values across rows.
Use the full valid range, not just the midpoint or mean.

Fields already set (do NOT include these in your output):
{json.dumps(partial_row, indent=2)}

You MUST generate ALL of these {len(remaining_fields)} fields — no omissions:
{required_list}

Constraints and realistic ranges for each field:
{constraints}

IMPORTANT SAMPLING RULES:
- For numeric fields: sample across the full [min, max] range, not just the mean.
- Use p25/p75 as a guide — roughly half your values should fall outside the mean.
- For categorical fields: vary your choices; do not always pick the most common value.
- Each row must feel like a DIFFERENT real vehicle/customer, not a clone of the last.

{diversity_hint}
{feedback_block}
Return a single JSON object containing ONLY the {len(remaining_fields)} required fields above.
No explanation, no markdown, no extra text."""
    # ────────────────────────────────────────────────────────────────────────