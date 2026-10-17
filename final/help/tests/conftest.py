"""Fakes for the embedding model and DeepSeek, and a helper writing articles."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import embedder

DIM = 256


def bag_of_words(text: str) -> np.ndarray:
    """A deterministic stand-in for e5: hashed word counts, L2-normalized."""
    vector = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


@pytest.fixture(autouse=True)
def fake_embedder(request, monkeypatch):
    """Every test but `@pytest.mark.model` ones runs on the fake; `calls`
    counts the passages embedded."""
    if request.node.get_closest_marker("model"):
        return None
    calls = SimpleNamespace(passages=0)

    def passages(texts):
        calls.passages += len(texts)
        return np.array([bag_of_words(text) for text in texts], dtype=np.float32).reshape(len(texts), DIM)

    monkeypatch.setenv("HELP_EMBED_MODEL", "fake-bag-of-words")
    monkeypatch.setattr(embedder, "embed_passages", passages)
    monkeypatch.setattr(embedder, "embed_query", bag_of_words)
    return calls


def write_article(directory: Path, number: int, article_id: str, body: str, *, title: str | None = None,
                  category: str = "start", commands: str = "[]", related: str = "[]") -> Path:
    path = directory / f"{number:02d}-{article_id}.md"
    title = title or article_id.replace("-", " ").capitalize()
    path.write_text(
        f"---\nid: {article_id}\ntitle: {title}\ncategory: {category}\ncommands: {commands}\n"
        f"related: {related}\n---\n\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def mini_articles(tmp_path) -> Path:
    """Three small articles: SSH keys (git), commits (git), running code (run)."""
    directory = tmp_path / "articles"
    directory.mkdir()
    write_article(directory, 1, "ssh-key", "SSH ключ для GitHub.\n\n## Как добавить ключ\n/git ssh показывает "
                  "ключ, добавьте ключ в GitHub.\n\n## Ключ не работает\nключ не добавлен в GitHub.\n\n"
                  "## Deploy key\nключ репозитория с правом записи.",
                  title="SSH-ключ", category="git", commands="[/git ssh]", related="[commit]")
    write_article(directory, 2, "commit", "Коммит изменений.\n\n## Как закоммитить\n/git commit сообщение "
                  "коммита.", title="Коммит", category="git", commands="[/git commit]")
    write_article(directory, 3, "run", "Запуск кода.\n\n## Как запустить скрипт\n/run main.py запускает "
                  "скрипт.", title="Запуск", category="run", commands="[/run]")
    return directory


class FakeClient:
    """An AsyncOpenAI stand-in: records each request, then replies with
    `reply` or raises `error`."""

    def __init__(self, reply: str = "Ответ модели.", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])
