"""The e5 embedding model through transformers: mean pooling over the
attention mask, L2-normalized float32 vectors."""

from __future__ import annotations

import functools
import os

import numpy as np

MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32
MAX_LENGTH = 512


class EmbedderError(RuntimeError):
    """The model could not be loaded or run."""


def model_name() -> str:
    return os.environ.get("HELP_EMBED_MODEL", MODEL)


@functools.cache
def _load(name: str):
    # Imported here so that code which never embeds (tests with a fake
    # embedder, `--help`) does not pay for loading torch.
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
    except Exception as error:
        raise EmbedderError(
            f"не удалось загрузить модель эмбеддингов {name}: {type(error).__name__}: {error}. "
            "При первом запуске модель скачивается с Hugging Face — нужна сеть; дальше она берётся из кэша."
        ) from error
    model.eval()
    return torch, tokenizer, model


def _embed(texts: list[str]) -> np.ndarray:
    torch, tokenizer, model = _load(model_name())
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(texts[start:start + BATCH_SIZE], max_length=MAX_LENGTH, padding=True,
                              truncation=True, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
    if not batches:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(batches).astype(np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    return _embed([f"passage: {text}" for text in texts])


def embed_query(text: str) -> np.ndarray:
    return _embed([f"query: {text}"])[0]
