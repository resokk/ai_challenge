"""Ollama /api/embed client: task prefixes, batching and L2 normalization."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

from . import CodeIndexError

DEFAULT_MODEL = "nomic-embed-text"
BATCH_SIZE = 32
MAX_TEXT_CHARS = 8000

# (document prefix, query prefix) per model; models not listed get none.
PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def post_json(path: str, payload: dict, timeout: float = 600) -> dict:
    """POST to Ollama; connection failures and a missing model become CodeIndexError."""
    host, model = ollama_host(), payload.get("model")
    request = urllib.request.Request(f"{host}{path}", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise CodeIndexError(f"Ollama model '{model}' is not available; run: ollama pull {model}") from None
        detail = e.read().decode(errors="replace").strip()
        raise CodeIndexError(f"Ollama {path} failed with HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise CodeIndexError(f"Ollama is not reachable at {host} ({reason}); run: ollama serve") from None


def default_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_MODEL)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Embedder:
    def __init__(self, model: str):
        self.model = model
        self.dim: int | None = None  # read from the first response
        self.doc_prefix, self.query_prefix = PREFIXES.get(model.split(":")[0], ("", ""))

    def _embed(self, texts: list[str]) -> np.ndarray:
        data = post_json("/api/embed", {"model": self.model, "input": texts, "truncate": True})
        vectors = np.asarray(data["embeddings"], dtype="float32")
        self.dim = vectors.shape[1]
        return _normalize(vectors)

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
