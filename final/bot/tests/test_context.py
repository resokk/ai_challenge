"""Context order, /summary, and history summarization."""

from __future__ import annotations

import json

import pytest

import bot
import config
import context
import git_ops
import llm
from fake_llm import ScriptedLLM, response


@pytest.fixture
def no_clone(monkeypatch):
    async def switch(ctx):
        return git_ops.SwitchResult(False)

    monkeypatch.setattr(git_ops, "switch_workspace", switch)


def add_tool_group(store, telegram_id, call_id="c1"):
    calls = [{"id": call_id, "type": "function",
              "function": {"name": "call_tool", "arguments": json.dumps({"name": "list_files", "arguments": {}})}}]
    store.add_message(telegram_id, "assistant", "", tool_calls=calls)
    store.add_message(telegram_id, "tool", "main.py (10 B)", tool_call_id=call_id)


# context order


def test_context_order(store, approved, registry):
    approved(1)
    store.set_agents_md(1, "use ruff")
    store.upsert_task(1, "a", "https://github.com/acme/demo")
    store.set_current_task(1, "a")
    store.set_memory(1, "a", "используем FastAPI")
    store.add_message(1, "user", "hi")
    add_tool_group(store, 1)
    store.add_message(1, "assistant", "hello")
    messages = context.build(store, 1, registry.catalog())
    assert [m["role"] for m in messages] == ["system", "system", "system", "user", "assistant", "tool", "assistant"]
    first = messages[0]["content"]
    assert first.startswith(context.system_prompt())
    assert "Текущая задача: `a`, репозиторий: https://github.com/acme/demo" in first
    assert "Память задачи:\nиспользуем FastAPI" in first
    assert messages[1]["content"] == f"{context.AGENTS_HEADING}\n\nuse ruff"
    assert "call_tool" in messages[2]["content"] and "read_file" in messages[2]["content"]
    assert messages[4]["tool_calls"][0]["id"] == "c1" and messages[5]["tool_call_id"] == "c1"


def test_optional_parts(store, approved, registry):
    approved(1)
    messages = context.build(store, 1, registry.catalog())
    assert len(messages) == 2  # no AGENTS.md, no history
    assert "Текущая задача" not in messages[0]["content"]
    store.upsert_task(1, "a", "https://github.com/acme/demo")
    store.set_current_task(1, "a")
    first = context.build(store, 1, registry.catalog())[0]["content"]
    assert "Текущая задача" in first and "Память задачи:\n" not in first


async def test_switching_task_keeps_history_changes_prefix(tg, store, approved, registry, no_clone):
    approved(1)
    await tg(bot.tasks, "/tasks a acme/demo")
    await tg(bot.memory, "/memory используем FastAPI")
    store.add_message(1, "user", "hi")
    await tg(bot.tasks, "/tasks b acme/other")
    messages = context.build(store, 1, registry.catalog())
    assert "`b`" in messages[0]["content"] and "FastAPI" not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hi"}
    await tg(bot.tasks, "/tasks a")
    messages = context.build(store, 1, registry.catalog())
    assert "`a`" in messages[0]["content"] and "FastAPI" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_incomplete_tool_group_is_dropped(store, approved, registry):
    approved(1)
    store.add_message(1, "user", "hi")
    store.add_message(1, "assistant", "", tool_calls=[{"id": "x", "type": "function",
                                                        "function": {"name": "call_tool", "arguments": "{}"}}])
    store.add_message(1, "user", "again")
    roles = [m["role"] for m in context.build(store, 1, registry.catalog())[2:]]
    assert roles == ["user", "user"]


# /summary


async def test_summary_request_has_no_tools(tg, store, approved, monkeypatch):
    approved(1)
    store.set_agents_md(1, "use ruff")
    store.set_last_tokens(1, 1234)
    store.add_message(1, "user", "посмотри проект")
    add_tool_group(store, 1)
    store.add_message(1, "assistant", "в проекте main.py")
    before = [dict(row) for row in store.history(1)]
    fake = ScriptedLLM(response("Цель: разобрать проект."))
    monkeypatch.setattr(llm, "complete", fake)

    message = await tg(bot.summary_command, "/summary")
    assert message.replies[-1] == "Цель: разобрать проект."
    request = fake.requests[0]
    assert request["tools"] is None
    messages = request["messages"]
    assert messages[0]["content"].startswith(context.system_prompt())
    assert messages[1]["content"].endswith("use ruff")
    text = json.dumps(messages, ensure_ascii=False)
    assert "### read_file" not in text and "tool_calls" not in text
    assert all(m["role"] != "tool" for m in messages)
    assert [m["role"] for m in messages[2:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == context.DIALOG_SUMMARY_TASK
    assert [dict(row) for row in store.history(1)] == before
    assert store.get_user(1)["last_tokens"] == 1234


async def test_summary_of_empty_history_skips_llm(tg, approved, monkeypatch):
    approved(1)
    fake = ScriptedLLM()
    monkeypatch.setattr(llm, "complete", fake)
    assert "пуста" in (await tg(bot.summary_command, "/summary")).replies[-1]
    assert fake.requests == []


# summarization


def fill(store, count, telegram_id=1):
    for i in range(count):
        store.add_message(telegram_id, "user" if i % 2 == 0 else "assistant", f"m{i}")


async def test_no_summary_below_threshold(tg, store, approved, monkeypatch):
    approved(1)
    fill(store, 9)
    store.set_last_tokens(1, int(config.CONTEXT_LIMIT_TOKENS * 0.59))
    fake = ScriptedLLM(response("ok"))
    monkeypatch.setattr(llm, "complete", fake)
    await tg(bot.chat, "next")
    assert len(fake.requests) == 1  # just the turn
    assert all(row["role"] != "summary" for row in store.history(1))


async def test_summary_compresses_oldest_two_thirds(tg, store, approved, registry, monkeypatch):
    approved(1)
    store.set_agents_md(1, "use ruff")
    store.upsert_task(1, "a", "https://github.com/acme/demo")
    store.set_current_task(1, "a")
    store.set_memory(1, "a", "FastAPI")
    fill(store, 9)
    prefix_before = context.build(store, 1, registry.catalog())[:3]
    store.set_last_tokens(1, int(config.CONTEXT_LIMIT_TOKENS * 0.6))
    fake = ScriptedLLM(response("СВОДКА-1"), response("ответ", prompt=300, completion=5))
    monkeypatch.setattr(llm, "complete", fake)

    await tg(bot.chat, "next")
    summary_request = fake.requests[0]
    assert summary_request["tools"] is None
    transcript = summary_request["messages"][1]["content"]
    assert "m5" in transcript and "m6" not in transcript

    rows = store.history(1)
    assert [row["role"] for row in rows][:1] == ["summary"]
    assert [row["content"] for row in rows] == ["СВОДКА-1", "m6", "m7", "m8", "next", "ответ"]
    turn = fake.requests[1]["messages"]
    assert turn[:3] == prefix_before
    assert turn[3] == {"role": "system", "content": f"{context.SUMMARY_HEADING}\nСВОДКА-1"}
    assert store.get_task(1, "a")["memory"] == "FastAPI"
    assert store.get_user(1)["last_tokens"] == 305


async def test_old_summary_is_absorbed(store, approved, monkeypatch):
    approved(1)
    fill(store, 6)
    monkeypatch.setattr(llm, "complete", ScriptedLLM(response("S1"), response("S2")))
    assert await context.summarize(store, 1)
    assert [row["content"] for row in store.history(1)] == ["S1", "m4", "m5"]
    fill(store, 3)
    assert await context.summarize(store, 1)
    rows = store.history(1)
    assert [row["content"] for row in rows] == ["S2", "m1", "m2"]
    assert sum(row["role"] == "summary" for row in rows) == 1


async def test_summary_never_splits_tool_group(store, approved, monkeypatch):
    approved(1)
    store.add_message(1, "user", "u0")
    store.add_message(1, "assistant", "a1")
    store.add_message(1, "user", "u2")
    add_tool_group(store, 1, "c1")  # rows 3, 4: the 2/3 boundary falls between them
    store.add_message(1, "assistant", "a5")
    monkeypatch.setattr(llm, "complete", ScriptedLLM(response("S")))
    assert context.split_point(store.history(1)) == 5
    await context.summarize(store, 1)
    assert [row["content"] for row in store.history(1)] == ["S", "a5"]


async def test_failed_summary_keeps_history(tg, store, approved, monkeypatch):
    approved(1)
    fill(store, 6)
    before = [dict(row) for row in store.history(1)]
    store.set_last_tokens(1, config.CONTEXT_LIMIT_TOKENS)
    monkeypatch.setattr(llm, "complete", ScriptedLLM(RuntimeError("down"), response("всё равно отвечаю")))
    message = await tg(bot.chat, "next")
    assert message.replies[-1] == "всё равно отвечаю"
    assert [dict(row) for row in store.history(1)][:6] == before


async def test_context_command(tg, store, approved, no_clone):
    approved(1)
    await tg(bot.tasks, "/tasks a acme/demo")
    store.set_last_tokens(1, config.CONTEXT_LIMIT_TOKENS // 4)
    fill(store, 3)
    store.replace_with_summary(1, "S", [store.history(1)[0]["id"]])
    reply = (await tg(bot.context_command, "/context")).replies[-1]
    assert "a — https://github.com/acme/demo" in reply and "Память задачи: нет" in reply
    assert "(25%)" in reply and "Сообщений в истории: 2" in reply and "Сводка: есть" in reply
