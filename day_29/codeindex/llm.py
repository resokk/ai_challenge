"""Local generation: plain text and JSON replies, with transformers or, on Apple silicon, MLX.

A model is loaded on first use and kept for the life of the process. A model id from mlx-community
(pre-quantized weights, e.g. "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit") runs on MLX; any other id
runs on transformers in bf16, on Apple's GPU (MPS) when there is one. The generation settings
(temperature, reply and context length) are process-wide: see configure().
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from functools import lru_cache

from . import CodeIndexError

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Settings:
    temperature: float = 0.0  # 0 is greedy decoding: the same prompt always gets the same reply
    top_p: float = 1.0  # with temperature > 0: sample only from the tokens making up this much probability
    # Answers take ~100 tokens, the longest JSON reply (task memory) a few hundred: the limit only cuts off
    # a model stuck repeating itself. Prompts take ~1.6k tokens; the window caps what sources may add.
    max_new_tokens: int = 768  # the reply's length limit
    context_window: int = 4096  # prompt and reply together, in tokens; a longer prompt is an error


settings = Settings()
usage = {"calls": 0, "prompt_tokens": 0, "new_tokens": 0, "seconds": 0.0, "peak_memory_gb": 0.0}


def configure(**changes) -> Settings:
    """Change the generation settings of every later call, e.g. configure(max_new_tokens=768)."""
    global settings
    settings = replace(settings, **changes)
    return settings


def is_mlx(model: str) -> bool:
    return model.startswith("mlx-community/")


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=None)
def load(model: str):
    """(tokenizer, model) for a Hugging Face model id, downloaded on first use."""
    try:
        if is_mlx(model):
            from mlx_lm import load as mlx_load

            lm, tokenizer = mlx_load(model)
            return tokenizer, lm
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    except (OSError, ImportError) as e:
        raise CodeIndexError(f"cannot load model '{model}': {e}") from None
    return tokenizer, lm.to(device()).eval()


def generate(model: str, prompt: str) -> str:
    """The model's reply to prompt sent as one user message."""
    tokenizer, lm = load(model)
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) + settings.max_new_tokens > settings.context_window:
        raise CodeIndexError(f"prompt of {len(ids)} tokens and a reply of up to {settings.max_new_tokens} "
                             f"do not fit the context window of {settings.context_window}")
    start = time.perf_counter()
    reply, new_tokens, memory = (generate_mlx if is_mlx(model) else generate_torch)(tokenizer, lm, ids)
    usage["calls"] += 1
    usage["prompt_tokens"] += len(ids)
    usage["new_tokens"] += new_tokens
    usage["seconds"] += time.perf_counter() - start
    usage["peak_memory_gb"] = max(usage["peak_memory_gb"], memory)
    return reply.strip()


def generate_torch(tokenizer, lm, ids: list[int]) -> tuple[str, int, float]:
    """(reply, its length in tokens, GPU memory in GB) from a transformers model."""
    import torch

    sampled = settings.temperature > 0
    with torch.inference_mode():
        out = lm.generate(torch.tensor([ids], device=lm.device), max_new_tokens=settings.max_new_tokens,
                          do_sample=sampled, temperature=settings.temperature if sampled else None,
                          top_p=settings.top_p if sampled else None, top_k=None,
                          pad_token_id=tokenizer.eos_token_id)
    new = out[0, len(ids):]
    memory = torch.mps.driver_allocated_memory() if lm.device.type == "mps" else 0
    return tokenizer.decode(new, skip_special_tokens=True), len(new), memory / 1e9


def generate_mlx(tokenizer, lm, ids: list[int]) -> tuple[str, int, float]:
    """(reply, its length in tokens, peak memory in GB) from an MLX model."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(settings.temperature, settings.top_p if settings.temperature > 0 else 0.0)
    text, last = "", None
    for last in stream_generate(lm, tokenizer, ids, max_tokens=settings.max_new_tokens, sampler=sampler):
        text += last.text
    return text, last.generation_tokens if last else 0, mx.get_peak_memory() / 1e9


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
