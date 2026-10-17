"""The semantic index over the help articles: built once, rebuilt when the
articles or the model change, searched by brute-force cosine similarity.

    python index.py build
    python index.py search "вопрос" [-k 5] [--category git]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import embedder
from articles import ARTICLES_DIR, CATEGORIES, HELP_DIR, Chunk, chunk, load_articles

INDEX_DIR = HELP_DIR / "index"
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
# At most this many chunks of one article in a result.
PER_ARTICLE = 2


@dataclass(frozen=True)
class Hit:
    article_id: str
    title: str
    section: str
    score: float
    text: str


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, categories: dict[str, str]):
        self.chunks = chunks
        self.vectors = vectors
        self.categories = categories

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[Hit]:
        """The `k` chunks closest to `query`, no more than two per article."""
        if category is not None and category not in CATEGORIES:
            raise ValueError(f"неизвестная категория {category!r}; есть: {', '.join(CATEGORIES)}")
        if not self.chunks:
            return []
        scores = self.vectors @ embedder.embed_query(query)
        hits: list[Hit] = []
        per_article: dict[str, int] = {}
        for i in np.argsort(-scores, kind="stable"):
            item = self.chunks[i]
            if category is not None and self.categories[item.article_id] != category:
                continue
            if per_article.get(item.article_id, 0) >= PER_ARTICLE:
                continue
            per_article[item.article_id] = per_article.get(item.article_id, 0) + 1
            hits.append(Hit(item.article_id, item.title, item.section, round(float(scores[i]), 3), item.text))
            if len(hits) == k:
                break
        return hits


def articles_hash(articles_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(articles_dir).glob("*.md")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def build(articles_dir: Path = ARTICLES_DIR, index_dir: Path = INDEX_DIR) -> Index:
    """Chunk and embed every article and write the index to `index_dir`."""
    articles = load_articles(articles_dir)
    chunks = [item for article in articles for item in chunk(article)]
    vectors = embedder.embed_passages([item.text for item in chunks])
    categories = {article.id: article.category for article in articles}
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / VECTORS_FILE, vectors)
    # Written last: its hash is what marks the index as complete and current.
    meta = {
        "model": embedder.model_name(),
        "articles_sha256": articles_hash(articles_dir),
        "categories": categories,
        "chunks": [asdict(item) for item in chunks],
    }
    (index_dir / CHUNKS_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return Index(chunks, vectors, categories)


def load(articles_dir: Path = ARTICLES_DIR, index_dir: Path = INDEX_DIR) -> Index | None:
    """The stored index, or None when it is missing, damaged or out of date."""
    index_dir = Path(index_dir)
    try:
        meta = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        vectors = np.load(index_dir / VECTORS_FILE)
    except (OSError, ValueError):
        return None
    if meta.get("model") != embedder.model_name() or meta.get("articles_sha256") != articles_hash(articles_dir):
        return None
    chunks = [Chunk(**item) for item in meta["chunks"]]
    if len(chunks) != len(vectors):
        return None
    return Index(chunks, vectors, meta["categories"])


def load_or_build(articles_dir: Path = ARTICLES_DIR, index_dir: Path = INDEX_DIR) -> Index:
    return load(articles_dir, index_dir) or build(articles_dir, index_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Индекс справки по боту")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="перестроить индекс")
    search = commands.add_parser("search", help="найти фрагменты по вопросу")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--category", choices=CATEGORIES)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            index = build()
            print(f"Индекс построен: {len(index.chunks)} фрагментов из {len(index.categories)} статей → {INDEX_DIR}")
            return 0
        for hit in load_or_build().search(args.query, args.k, args.category):
            print(f"{hit.score:.3f}  [{hit.article_id}] {hit.title} — {hit.section}")
        return 0
    except embedder.EmbedderError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
