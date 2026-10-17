"""The real e5 model on the real articles: user-worded questions find the
right article. Skipped when the model is not in the Hugging Face cache."""

from __future__ import annotations

import pytest

import embedder
import index

try:
    from huggingface_hub import try_to_load_from_cache
except ImportError:  # transformers not installed
    try_to_load_from_cache = None

CACHED = try_to_load_from_cache is not None and isinstance(
    try_to_load_from_cache(embedder.MODEL, "config.json"), str
)

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not CACHED, reason=f"модели {embedder.MODEL} нет в кэше Hugging Face"),
]

QUESTIONS = [
    ("как подключить приватный репо", "private-repo"),
    ("бот не пушит", "git-push"),
    ("как запустить тесты", "run-module"),
    ("как удалить историю", "reset"),
    ("как добавить ssh ключ в github", "ssh-key"),
    ("как закоммитить изменения", "git-commit"),
    ("как поставить пакет requests", "pip"),
    ("как откатить все изменения в файлах", "git-reset"),
    ("как получить доступ к боту", "register"),
    ("сколько может работать мой скрипт", "run-limits"),
    ("где посмотреть заполнение контекста", "context"),
    ("как переключиться на другой проект", "task-switch"),
]


@pytest.fixture(scope="module")
def real_index(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("HELP_EMBED_MODEL", raising=False)
        patch.setenv("HF_HUB_OFFLINE", "1")
        yield index.build(index.ARTICLES_DIR, tmp_path_factory.mktemp("index"))


@pytest.mark.parametrize("question, article_id", QUESTIONS)
def test_article_in_top_three(real_index, monkeypatch, question, article_id):
    monkeypatch.delenv("HELP_EMBED_MODEL", raising=False)
    found = [hit.article_id for hit in real_index.search(question, k=3)]
    assert article_id in found, (question, found)
