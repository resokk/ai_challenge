"""LLM summaries of chunks via Ollama /api/generate, cached in meta.db by content_hash."""

from __future__ import annotations

import sqlite3
import sys
from typing import Callable

from .chunker import Chunk
from .embedder import post_json

DEFAULT_MODEL = "qwen2.5-coder:1.5b"
PROMPT = "Describe in 1–3 sentences what this code does.\n\n{code}"


def ollama_generate(model: str, prompt: str, temperature: float | None = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    return post_json("/api/generate", payload)["response"].strip()


class Summarizer:
    def __init__(self, model: str = DEFAULT_MODEL, generate: Callable[[str, str], str] = ollama_generate):
        self.model = model
        self.generate = generate

    def embed_texts(self, conn: sqlite3.Connection, chunks: list[Chunk]) -> list[str]:
        """Header, summary and code per chunk; only uncached content reaches the LLM.

        New summaries join conn's open transaction, so they persist only with the index.
        """
        texts, generated = [], 0
        for chunk in chunks:
            row = conn.execute("SELECT summary FROM summaries WHERE content_hash = ? AND model = ?",
                               (chunk.content_hash, self.model)).fetchone()
            if row:
                summary = row[0]
            else:
                summary = self.generate(self.model, PROMPT.format(code=chunk.content))
                conn.execute("INSERT OR REPLACE INTO summaries VALUES (?, ?, ?)",
                             (chunk.content_hash, self.model, summary))
                generated += 1
                print(f"\rsummarizing: {generated} generated", end="", file=sys.stderr)
            texts.append(f"{chunk.header()}# summary: {summary}\n{chunk.content}")
        print(f"\rsummaries: {generated} generated, {len(chunks) - generated} cached", file=sys.stderr)
        return texts
