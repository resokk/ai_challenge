"""Building, reusing and rebuilding the index; search filters."""

from __future__ import annotations

import json

import numpy as np
import pytest

import index


def test_build_writes_vectors_and_chunks(mini_articles, tmp_path):
    built = index.build(mini_articles, tmp_path / "index")
    vectors = np.load(tmp_path / "index" / "vectors.npy")
    meta = json.loads((tmp_path / "index" / "chunks.json").read_text(encoding="utf-8"))
    assert vectors.shape == (len(built.chunks), 256) and vectors.dtype == np.float32
    assert meta["model"] == "fake-bag-of-words"
    assert meta["articles_sha256"] == index.articles_hash(mini_articles)
    assert [c["article_id"] for c in meta["chunks"]].count("ssh-key") == 4
    assert meta["categories"] == {"ssh-key": "git", "commit": "git", "run": "run"}


def test_load_reuses_index_without_rebuilding(mini_articles, tmp_path, fake_embedder):
    index.build(mini_articles, tmp_path / "index")
    embedded = fake_embedder.passages
    loaded = index.load_or_build(mini_articles, tmp_path / "index")
    assert fake_embedder.passages == embedded
    assert loaded.search("как закоммитить", k=1)[0].article_id == "commit"


def test_rebuild_when_an_article_changes(mini_articles, tmp_path, fake_embedder):
    index.build(mini_articles, tmp_path / "index")
    path = mini_articles / "03-run.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n## Модули\n/run -m pytest запускает тесты.\n",
                    encoding="utf-8")
    assert index.load(mini_articles, tmp_path / "index") is None
    embedded = fake_embedder.passages
    rebuilt = index.load_or_build(mini_articles, tmp_path / "index")
    assert fake_embedder.passages > embedded
    assert "Модули" in {item.section for item in rebuilt.chunks}
    assert index.load(mini_articles, tmp_path / "index") is not None


def test_rebuild_when_the_model_changes(mini_articles, tmp_path, monkeypatch):
    index.build(mini_articles, tmp_path / "index")
    monkeypatch.setenv("HELP_EMBED_MODEL", "another-model")
    assert index.load(mini_articles, tmp_path / "index") is None
    index.load_or_build(mini_articles, tmp_path / "index")
    meta = json.loads((tmp_path / "index" / "chunks.json").read_text(encoding="utf-8"))
    assert meta["model"] == "another-model"


def test_damaged_index_is_rebuilt(mini_articles, tmp_path):
    index.build(mini_articles, tmp_path / "index")
    (tmp_path / "index" / "chunks.json").write_text("{broken", encoding="utf-8")
    assert index.load(mini_articles, tmp_path / "index") is None
    assert index.load_or_build(mini_articles, tmp_path / "index").chunks


def test_search_ranks_and_rounds(mini_articles, tmp_path):
    hits = index.build(mini_articles, tmp_path / "index").search("ключ GitHub", k=5)
    assert hits[0].article_id == "ssh-key"
    assert hits == sorted(hits, key=lambda hit: -hit.score)
    assert all(hit.score == round(hit.score, 3) for hit in hits)
    assert hits[0].text.startswith("SSH-ключ. ")


def test_no_more_than_two_chunks_per_article(mini_articles, tmp_path):
    hits = index.build(mini_articles, tmp_path / "index").search("ключ ключ GitHub", k=10)
    ids = [hit.article_id for hit in hits]
    assert ids.count("ssh-key") == 2
    assert set(ids) == {"ssh-key", "commit", "run"}


def test_category_filter(mini_articles, tmp_path):
    built = index.build(mini_articles, tmp_path / "index")
    assert {hit.article_id for hit in built.search("ключ", k=5, category="run")} == {"run"}
    assert {hit.article_id for hit in built.search("ключ", k=5, category="git")} == {"ssh-key", "commit"}
    assert built.search("ключ", category="tools") == []
    with pytest.raises(ValueError, match="категория"):
        built.search("ключ", category="nope")


def test_cli_search(mini_articles, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(index, "ARTICLES_DIR", mini_articles)
    monkeypatch.setattr(index, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(index.build, "__defaults__", (mini_articles, tmp_path / "index"))
    monkeypatch.setattr(index.load_or_build, "__defaults__", (mini_articles, tmp_path / "index"))
    assert index.main(["build"]) == 0
    assert index.main(["search", "как закоммитить", "-k", "1"]) == 0
    assert "[commit] Коммит — Как закоммитить" in capsys.readouterr().out
