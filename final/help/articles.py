"""Help articles: loading, validation and splitting into chunks for the index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HELP_DIR = Path(__file__).resolve().parent
ARTICLES_DIR = HELP_DIR / "articles"

CATEGORIES = ("start", "tasks", "git", "files", "run", "chat", "tools", "admin", "troubleshooting", "examples")
FIELDS = ("id", "title", "category", "commands", "related")
LIST_FIELDS = ("commands", "related")
FILE_PATTERN = re.compile(r"(\d{2})-([a-z0-9-]+)\.md")
# A section longer than this is cut at paragraph boundaries.
CHUNK_MAX_CHARS = 1200
INTRO_SECTION = "Коротко"


class ArticleError(ValueError):
    """An article that cannot be indexed; the message names the file."""


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    commands: tuple[str, ...]
    related: tuple[str, ...]
    body: str
    path: Path

    def word_count(self) -> int:
        return len(self.body.split())


@dataclass(frozen=True)
class Chunk:
    article_id: str
    title: str
    section: str
    text: str


def _value(raw: str) -> str | tuple[str, ...]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return tuple(item.strip() for item in raw[1:-1].split(",") if item.strip())
    return raw


def parse(path: Path) -> Article:
    """One article: `key: value` / `key: [a, b]` front matter, then the body."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[3:]:
        raise ArticleError(f"{path.name}: нет front matter между строками ---")
    header, body = text[4:].split("\n---\n", 1)
    meta: dict[str, str | tuple[str, ...]] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        key, sep, raw = line.partition(":")
        if not sep:
            raise ArticleError(f"{path.name}: строка front matter без «:»: {line!r}")
        meta[key.strip()] = _value(raw)
    for field in FIELDS:
        if field not in meta:
            raise ArticleError(f"{path.name}: нет поля {field}")
        if (field in LIST_FIELDS) != isinstance(meta[field], tuple):
            kind = "список [a, b]" if field in LIST_FIELDS else "строка"
            raise ArticleError(f"{path.name}: поле {field} должно быть: {kind}")
    match = FILE_PATTERN.fullmatch(path.name)
    if not match or match[2] != meta["id"]:
        raise ArticleError(f"{path.name}: имя файла должно быть NN-{meta['id']}.md")
    if meta["category"] not in CATEGORIES:
        raise ArticleError(f"{path.name}: неизвестная категория {meta['category']!r}")
    return Article(
        id=meta["id"],
        title=meta["title"],
        category=meta["category"],
        commands=meta["commands"],
        related=meta["related"],
        body=body.strip(),
        path=path,
    )


def load_articles(directory: Path = ARTICLES_DIR) -> list[Article]:
    """All articles in reading order, with ids unique and `related` resolvable."""
    articles = [parse(path) for path in sorted(Path(directory).glob("*.md"))]
    seen: dict[str, Path] = {}
    for article in articles:
        if article.id in seen:
            raise ArticleError(f"{article.path.name}: id {article.id} уже есть в {seen[article.id].name}")
        seen[article.id] = article.path
    for article in articles:
        for related in article.related:
            if related not in seen:
                raise ArticleError(f"{article.path.name}: related ссылается на несуществующую статью {related}")
    return articles


def _sections(body: str) -> list[tuple[str, str]]:
    """(heading, text) per `##` section; the part before the first one is the
    intro. `#` lines and headings inside code blocks are left alone."""
    sections: list[tuple[str, list[str]]] = [(INTRO_SECTION, [])]
    fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif not fence and line.startswith("# "):
            continue
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _pieces(text: str, limit: int) -> list[str]:
    """`text` cut at paragraph boundaries into pieces of about `limit` chars;
    a single paragraph longer than that stays whole."""
    pieces: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        if current and len(current) + len(paragraph) + 2 > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk(article: Article, limit: int = CHUNK_MAX_CHARS) -> list[Chunk]:
    """One chunk per section, each starting with "{title}. {section}" so its
    embedding knows what the article is about."""
    return [
        Chunk(article.id, article.title, heading, f"{article.title}. {heading}\n{piece}")
        for heading, text in _sections(article.body)
        for piece in _pieces(text, limit)
    ]
