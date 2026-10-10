"""Embeddings with sentence-transformers: task prefixes, batching and L2 normalization."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

from . import CodeIndexError
from .llm import device

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 32
MAX_TEXT_CHARS = 8000

# (document prefix, query prefix) per model; models not listed get none.
PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "mixedbread-ai/mxbai-embed-large-v1": ("", "Represent this sentence for searching relevant passages: "),
}


def default_model() -> str:
    return os.environ.get("CODEINDEX_EMBED_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=None)
def load(model: str):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model, device=device())
    except OSError as e:
        raise CodeIndexError(f"cannot load embedding model '{model}': {e}") from None


class Embedder:
    def __init__(self, model: str):
        self.model = model
        self.dim: int | None = None  # read from the model once it is loaded
        self.doc_prefix, self.query_prefix = PREFIXES.get(model, ("", ""))

    def _embed(self, texts: list[str]) -> np.ndarray:
        st = load(self.model)
        self.dim = st.get_embedding_dimension()
        return st.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        batches = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [self.doc_prefix + t[:MAX_TEXT_CHARS] for t in texts[i:i + BATCH_SIZE]]
            batches.append(self._embed(batch))
            print(f"\rembedding {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="", file=sys.stderr)
        if batches:
            print(file=sys.stderr)
        return np.vstack(batches) if batches else np.empty((0, self.dim or 0), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text[:MAX_TEXT_CHARS]])[0]
