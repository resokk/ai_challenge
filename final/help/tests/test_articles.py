"""The 50 articles, their command coverage, front matter checks and chunking."""

from __future__ import annotations

import re

import pytest

import articles
from articles import ARTICLES_DIR, ArticleError, chunk, load_articles, parse
from conftest import write_article

BOT_PY = articles.HELP_DIR.parent / "bot" / "bot.py"
GIT_SUBCOMMANDS = ("commit", "push", "diff", "reset", "ssh")


@pytest.fixture(scope="module")
def all_articles():
    return load_articles()


def test_fifty_articles_numbered_in_order(all_articles):
    names = sorted(path.name for path in ARTICLES_DIR.glob("*.md"))
    assert len(names) == 50
    assert [name[:3] for name in names] == [f"{n:02d}-" for n in range(1, 51)]
    assert len(all_articles) == 50


def test_article_size_and_links(all_articles):
    ids = {article.id for article in all_articles}
    for article in all_articles:
        assert 150 <= article.word_count() <= 500, (article.id, article.word_count())
        assert set(article.related) <= ids and article.id not in article.related, article.id
        assert article.body.startswith(f"# {article.title}\n"), article.id
        assert "\n## " in article.body, article.id


def test_every_category_has_articles(all_articles):
    assert {article.category for article in all_articles} == set(articles.CATEGORIES)


def test_every_bot_command_has_an_article(all_articles):
    """Fails as soon as the bot gets a command (or /git subcommand) without help."""
    commands = set(re.findall(r'CommandHandler\("(\w+)"', BOT_PY.read_text(encoding="utf-8")))
    assert {"start", "help", "git", "tasks"} <= commands
    covered = {command for article in all_articles for command in article.commands}
    missing = [f"/{name}" for name in sorted(commands) if f"/{name}" not in covered]
    missing += [f"/git {sub}" for sub in GIT_SUBCOMMANDS if f"/git {sub}" not in covered]
    assert not missing, f"нет статьи для команд: {missing}"


# front matter


def test_parse_reads_lists_and_strings(tmp_path):
    path = write_article(tmp_path, 7, "demo", "Текст.\n\n## Раздел\nещё", title="Демо: тест",
                         commands="[/git ssh, /tasks]", related="[]")
    article = parse(path)
    assert article.title == "Демо: тест"
    assert article.commands == ("/git ssh", "/tasks") and article.related == ()


@pytest.mark.parametrize("header, error", [
    ("id: demo\ntitle: T\ncategory: start\ncommands: []", "нет поля related"),
    ("id: other\ntitle: T\ncategory: start\ncommands: []\nrelated: []", "NN-other.md"),
    ("id: demo\ntitle: T\ncategory: misc\ncommands: []\nrelated: []", "неизвестная категория"),
    ("id: demo\ntitle: T\ncategory: start\ncommands: /tasks\nrelated: []", "commands"),
])
def test_parse_rejects_bad_front_matter(tmp_path, header, error):
    path = tmp_path / "01-demo.md"
    path.write_text(f"---\n{header}\n---\n\n# T\n\ntext\n", encoding="utf-8")
    with pytest.raises(ArticleError, match=error) as caught:
        parse(path)
    assert "01-demo.md" in str(caught.value)


def test_load_rejects_duplicate_id_and_broken_related(tmp_path):
    write_article(tmp_path, 1, "a", "text")
    write_article(tmp_path, 2, "b", "text", related="[missing]")
    with pytest.raises(ArticleError, match="02-b.md.*missing"):
        load_articles(tmp_path)
    (tmp_path / "02-b.md").unlink()
    write_article(tmp_path, 3, "a", "text")
    with pytest.raises(ArticleError, match="03-a.md.*уже есть"):
        load_articles(tmp_path)


# chunking


def test_chunks_follow_sections_and_start_with_title(tmp_path):
    body = "Вступление.\n\n## Как сделать\nшаг 1\n\n```\n## не заголовок\n```\n\n## Что может пойти не так\nошибка"
    article = parse(write_article(tmp_path, 1, "demo", body, title="Демо"))
    chunks = chunk(article)
    assert [item.section for item in chunks] == ["Коротко", "Как сделать", "Что может пойти не так"]
    assert chunks[0].text == "Демо. Коротко\nВступление."
    assert chunks[1].text.startswith("Демо. Как сделать\nшаг 1") and "## не заголовок" in chunks[1].text
    assert all(item.article_id == "demo" and item.title == "Демо" for item in chunks)
    assert "# Демо" not in chunks[0].text


def test_long_section_is_cut_at_paragraphs(tmp_path):
    paragraphs = [f"Абзац {i}. " + "слово " * 60 for i in range(8)]
    article = parse(write_article(tmp_path, 1, "demo", "## Длинный\n" + "\n\n".join(paragraphs), title="Демо"))
    chunks = chunk(article, limit=1200)
    assert len(chunks) > 1
    assert all(item.section == "Длинный" and item.text.startswith("Демо. Длинный\n") for item in chunks)
    assert all(len(item.text) <= 1200 + len("Демо. Длинный\n") for item in chunks)
    joined = "\n\n".join(item.text.split("\n", 1)[1] for item in chunks)
    assert joined == "\n\n".join(paragraphs).strip()


def test_real_articles_chunk_within_limit(all_articles):
    chunks = [item for article in all_articles for item in chunk(article)]
    assert len(chunks) > 150
    for item in chunks:
        assert item.text.startswith(f"{item.title}. {item.section}\n")
