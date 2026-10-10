"""Local generation with Hugging Face transformers: plain text and JSON replies.

A model is loaded on first use and kept for the life of the process, on Apple's GPU (MPS) when there
is one. Decoding is greedy, so the same prompt always gets the same reply.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from . import CodeIndexError

MAX_NEW_TOKENS = 2000
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=None)
def load(model: str):
    """(tokenizer, model) for a Hugging Face model id, downloaded on first use."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    except OSError as e:
        raise CodeIndexError(f"cannot load model '{model}': {e}") from None
    return tokenizer, lm.to(device()).eval()


def generate(model: str, prompt: str) -> str:
    """The model's reply to prompt sent as one user message."""
    import torch

    tokenizer, lm = load(model)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(lm.device)
    with torch.inference_mode():
        out = lm.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                          top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def conforms(value, schema: dict) -> bool:
    """Whether value matches schema: types, required keys, array items (the JSON Schema subset used here)."""
    kind = schema.get("type")
    if kind == "object":
        return (isinstance(value, dict) and set(schema.get("required", [])) <= value.keys()
                and all(conforms(value[k], s) for k, s in schema.get("properties", {}).items() if k in value))
    if kind == "array":
        return isinstance(value, list) and all(conforms(v, schema.get("items", {})) for v in value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def parse_json(model: str, reply: str, schema: dict) -> dict:
    """The JSON object in reply, checked against schema.

    transformers, unlike Ollama, does not constrain decoding to a schema, so the object is taken from
    the first "{" to the last "}" (models like to wrap it in ```json fences) and checked here.
    """
    match = JSON_RE.search(reply)
    try:
        parsed = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict) or not conforms(parsed, schema):
        raise CodeIndexError(f"{model} replied without the expected JSON: {reply[:200]!r}")
    return parsed


def generate_json(model: str, prompt: str, schema: dict) -> dict:
    return parse_json(model, generate(model, prompt), schema)
