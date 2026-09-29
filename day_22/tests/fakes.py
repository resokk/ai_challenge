"""Ollama stand-ins so the tests run offline."""

import hashlib
import re

import numpy as np

from codeindex import CodeIndexError


class FakeEmbedder:
    """Bag-of-words vectors: each word adds to a hashed dimension."""

    def __init__(self, model="fake-embed", dim=64):
        self.model, self.dim, self.calls = model, dim, []

    def _vector(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


class DownEmbedder(FakeEmbedder):
    def embed_documents(self, texts):
        raise CodeIndexError("Ollama is not reachable at http://localhost:11434; run: ollama serve")


class FakeGenerate:
    def __init__(self):
        self.prompts = []

    def __call__(self, model, prompt):
        self.prompts.append(prompt)
        return f"summary {len(self.prompts)}"
