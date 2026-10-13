"""Shared fixtures: a miniature Camunda-like git repository and a deterministic fake embedder."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from camunda_rag.config import EmbeddingSettings, Settings
from camunda_rag.embeddings import Chunk, Embedder

FIXTURE = Path(__file__).parent / "fixtures" / "minirepo"


class FakeEmbedder(Embedder):
    """Whitespace tokens, hashed bag-of-words vectors: deterministic and model-free."""

    def __init__(self, max_tokens: int = 64) -> None:  # noqa: D107 - no model loading
        self.cfg = EmbeddingSettings(max_tokens=max_tokens)
        self.name, self.device, self.dim = "fake", "cpu", 64
        self.model = object()

    def count(self, text: str) -> int:
        return len(text.split()) + 2

    def _split_unit(self, unit: str, budget: int) -> list[Chunk]:
        words = unit.split()
        step = budget - 2
        return [Chunk(" ".join(words[i:i + step]), len(words[i:i + step]) + 2, "split") for i in range(0, len(words), step)]

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out


@pytest.fixture(scope="session")
def mini_repo(tmp_path_factory) -> Path:
    repo = tmp_path_factory.mktemp("repo") / "camunda"
    shutil.copytree(FIXTURE, repo)
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "fixture"]):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


@pytest.fixture(scope="session")
def settings(mini_repo, tmp_path_factory) -> Settings:
    return Settings(repo_path=mini_repo, data_dir=tmp_path_factory.mktemp("data"), dialects=["postgresql", "oracle", "h2"],
                    default_dialect="postgresql")


@pytest.fixture(scope="session")
def indexed(settings) -> Settings:
    from camunda_rag.indexer import Indexer

    Indexer(settings, progress=lambda m: None).build(skip_java=True, embed=True, embedder=FakeEmbedder())
    return settings


@pytest.fixture(scope="session")
def service(indexed):
    from camunda_rag.service import CatalogService

    svc = CatalogService(indexed, load_embedder=False)
    return svc
