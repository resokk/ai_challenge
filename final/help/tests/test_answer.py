"""answer(): what DeepSeek is sent, the sources line, the fallback."""

from __future__ import annotations

import pytest

import answer
import index
from conftest import FakeClient


@pytest.fixture
def help_index(mini_articles, tmp_path, monkeypatch):
    built = index.build(mini_articles, tmp_path / "index")
    monkeypatch.setattr(answer, "_index", built)
    return built


@pytest.fixture
def deepseek(monkeypatch):
    def install(**kwargs):
        fake = FakeClient(**kwargs)
        monkeypatch.setattr(answer, "client", lambda: fake)
        return fake

    return install


async def test_request_has_prompt_question_and_fragments(help_index, deepseek):
    fake = deepseek(reply="Отправьте `/git ssh` и добавьте ключ в GitHub.")
    text = await answer.answer("как добавить ключ в GitHub?")
    [request] = fake.requests
    system, user = request["messages"]
    assert system == {"role": "system", "content": answer.PROMPT_PATH.read_text(encoding="utf-8").strip()}
    assert "только по приведённым фрагментам" in system["content"]
    assert user["role"] == "user" and "как добавить ключ в GitHub?" in user["content"]
    assert "[ssh-key] SSH-ключ — Как добавить ключ\nSSH-ключ. Как добавить ключ\n/git ssh" in user["content"]
    assert request["temperature"] == 0.2 and request["model"] == answer.model()
    assert text.startswith("Отправьте `/git ssh`")
    assert text.splitlines()[-1].startswith("Статьи: SSH-ключ (ssh-key)")


async def test_sources_line_lists_each_article_once(help_index, deepseek):
    deepseek(reply="ответ")
    text = await answer.answer("ключ ключ GitHub")
    sources = text.splitlines()[-1]
    assert sources.count("(ssh-key)") == 1
    assert sources.startswith("Статьи: SSH-ключ (ssh-key), ")
    assert set(sources.removeprefix("Статьи: ").split(", ")) == {"SSH-ключ (ssh-key)", "Коммит (commit)", "Запуск (run)"}


@pytest.mark.parametrize("error", [TimeoutError("timed out"), RuntimeError("401 invalid key")])
async def test_deepseek_failure_returns_fragments(help_index, deepseek, error, caplog):
    fake = deepseek(error=error)
    text = await answer.answer("как закоммитить")
    assert len(fake.requests) == 1
    assert text.startswith("Суммаризация недоступна, найденные фрагменты справки:\n\n[commit] Коммит — Как закоммитить")
    assert text.splitlines()[-1].startswith("Статьи: Коммит (commit)")
    assert type(error).__name__ in caplog.text


async def test_empty_model_reply_counts_as_failure(help_index, deepseek):
    deepseek(reply="  ")
    assert (await answer.answer("как закоммитить")).startswith(answer.FALLBACK_HEADING)


@pytest.mark.parametrize("question, message", [("", "пустой"), ("   ", "пустой"), ("x" * 1001, "слишком длинный")])
async def test_bad_questions_never_reach_deepseek(help_index, deepseek, question, message):
    fake = deepseek()
    with pytest.raises(answer.QuestionError, match=message):
        await answer.answer(question)
    assert fake.requests == []


async def test_question_at_the_limit_is_accepted(help_index, deepseek):
    deepseek()
    assert await answer.answer("ключ " + "x" * 995)
