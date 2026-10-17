"""Handlers: access, tool-call routing, /tasks, /memory, AGENTS.md."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram.ext import ApplicationHandlerStop

import bot
import config
import git_ops
import llm
from conftest import FakeQuery
from fake_llm import FakeCall, ScriptedLLM, response
from tools.registry import Tool


@pytest.fixture
def no_clone(monkeypatch):
    async def switch(ctx):
        return git_ops.SwitchResult(False)

    monkeypatch.setattr(git_ops, "switch_workspace", switch)


@pytest.fixture
def scripted(monkeypatch):
    def install(*responses):
        fake = ScriptedLLM(*responses)
        monkeypatch.setattr(llm, "complete", fake)
        return fake

    return install


# registration and access


async def test_start_registers_pending_once(tg, store):
    message = await tg(bot.start, "/start")
    assert "ждёт одобрения" in message.replies[-1]
    await tg(bot.start, "/start")
    assert len(store.list_users()) == 1
    assert store.get_user(1)["status"] == "pending"


@pytest.mark.parametrize("status", [None, "pending", "blocked"])
async def test_gate_stops_unapproved(tg, store, scripted, status):
    fake = scripted()
    if status:
        store.register(1, "u", "U")
        store.set_status(1, status)
    with pytest.raises(ApplicationHandlerStop):
        await tg(bot.gate, "hello")
    with pytest.raises(ApplicationHandlerStop):
        await tg(bot.gate, query=FakeQuery("gitreset:1:a"))
    assert fake.requests == [] and store.history(1) == []


async def test_gate_lets_start_and_approved_through(tg, approved):
    await tg(bot.gate, "/start")
    await tg(bot.gate, "/start@devbot")
    approved(1)
    await tg(bot.gate, "hello")


# who gets a tool result


async def test_model_tool_result_goes_to_history_not_user(tg, approved, store, scripted):
    approved(1)
    fake = scripted(
        response(None, [FakeCall("c1", "current_datetime", {})]),
        response("Сейчас полдень.", prompt=500, completion=20),
    )
    message = await tg(bot.chat, "который час?")
    assert message.replies[-1] == "Сейчас полдень."
    assert not any("UTC" in reply for reply in message.replies)
    assert any(reply == "вызываю current_datetime…" for reply in message.replies)
    assert all(sent.deleted for sent in message.sent if sent.text.startswith("вызываю"))
    roles = [(row["role"], row["tool_call_id"]) for row in store.history(1)]
    assert roles == [("user", None), ("assistant", None), ("tool", "c1"), ("assistant", None)]
    assert "UTC" in store.history(1)[2]["content"]
    assert store.get_user(1)["last_tokens"] == 520
    assert fake.requests[0]["tools"][0]["function"]["name"] == "call_tool"
    assert fake.requests[1]["messages"][-1]["role"] == "tool"


async def test_tool_loop_is_bounded(tg, approved, scripted, monkeypatch):
    approved(1)
    monkeypatch.setattr(config, "MAX_TOOL_ITERATIONS", 2)
    fake = scripted(
        response(None, [FakeCall("c1", "current_datetime", {})]),
        response(None, [FakeCall("c2", "current_datetime", {})]),
        response("Хватит."),
    )
    message = await tg(bot.chat, "loop")
    assert message.replies[-1] == "Хватит."
    assert fake.requests[-1]["tools"] is None


async def test_user_tool_call_answers_user_only(tg, approved, store, scripted):
    approved(1)
    fake = scripted()
    message = await tg(bot.tool_command, "/tool current_datetime {}")
    assert "UTC" in message.replies[-1]
    assert fake.requests == [] and store.history(1) == []
    message = await tg(bot.tool_command, "/tool python_docs_link {\"topic\": \"json.loads\"}")
    assert message.replies[-1].endswith("json.html#json.loads")


@pytest.mark.parametrize("text", ["/tool current_datetime {oops", "/tool current_datetime [1]"])
async def test_user_tool_invalid_json(tg, approved, text):
    approved(1)
    assert "JSON-объект" in (await tg(bot.tool_command, text)).replies[-1]


async def test_tool_without_arguments_shows_help(tg, approved):
    approved(1)
    assert "/tools" in (await tg(bot.tool_command, "/tool")).replies[-1]


async def test_help_without_question_lists_commands(tg, approved):
    approved(1)
    assert "/tasks" in (await tg(bot.help_command, "/help")).replies[-1]


async def test_help_question_goes_to_help_server(tg, approved, registry, store, scripted):
    approved(1)
    fake = scripted()
    asked = []

    async def answer(ctx, arguments):
        asked.append(arguments)
        return "Ответ справки"

    registry.register(Tool(bot.HELP_TOOL, "help", {"type": "object", "properties": {"question": {"type": "string"}},
                                                   "required": ["question"]}, answer))
    message = await tg(bot.help_command, "/help как создать задачу?")
    assert asked == [{"question": "как создать задачу?"}]
    assert message.replies[-1] == "Ответ справки"
    assert fake.requests == [] and store.history(1) == []


async def test_help_question_without_help_server(tg, approved):
    approved(1)
    reply = (await tg(bot.help_command, "/help как создать задачу?")).replies[-1]
    assert "недоступен" in reply and "/tasks" in reply


async def test_llm_error_keeps_user_message(tg, approved, store, scripted):
    approved(1)
    scripted(RuntimeError("network down"))
    message = await tg(bot.chat, "hello")
    assert "Не удалось" in message.replies[-1]
    assert [row["role"] for row in store.history(1)] == ["user"]


# /tasks


async def test_tasks_empty(tg, approved):
    approved(1)
    assert "Задач нет" in (await tg(bot.tasks, "/tasks")).replies[-1]


async def test_tasks_create_update_switch(tg, approved, store, no_clone):
    approved(1)
    await tg(bot.tasks, "/tasks a acme/demo")
    await tg(bot.tasks, "/tasks b https://github.com/acme/other.git")
    assert store.get_user(1)["current_task_id"] == "b"
    assert store.get_task(1, "b")["repo_url"] == "https://github.com/acme/other"
    await tg(bot.tasks, "/tasks a https://github.com/acme/renamed")
    assert len(store.list_tasks(1)) == 2
    assert store.get_task(1, "a")["repo_url"] == "https://github.com/acme/renamed"
    await tg(bot.tasks, "/tasks b")
    assert store.get_user(1)["current_task_id"] == "b"
    listing = (await tg(bot.tasks, "/tasks")).replies[-1]
    assert "▶ b — https://github.com/acme/other" in listing and "• a — https://github.com/acme/renamed" in listing


async def test_tasks_errors(tg, approved, store, no_clone):
    approved(1)
    await tg(bot.tasks, "/tasks a acme/demo")
    reply = (await tg(bot.tasks, "/tasks zzz")).replies[-1]
    assert "Задачи zzz нет" in reply and "a" in reply
    assert "id задачи" in (await tg(bot.tasks, "/tasks bad/id acme/demo")).replies[-1]
    for repo in ("acme", "http://github.com/a/b", "https://gitlab.com/a/b", "a/b/c", "git@github.com:a/b.git"):
        assert "owner/name" in (await tg(bot.tasks, f"/tasks c {repo}")).replies[-1], repo
    assert "/tasks <id>" in (await tg(bot.tasks, "/tasks a b c")).replies[-1]
    assert [row["task_id"] for row in store.list_tasks(1)] == ["a"]


async def test_tasks_are_per_user(tg, approved, store, no_clone):
    approved(1)
    approved(2)
    await tg(bot.tasks, "/tasks a acme/demo", user=1)
    assert "Задач нет" in (await tg(bot.tasks, "/tasks", user=2)).replies[-1]
    assert "нет" in (await tg(bot.tasks, "/tasks a", user=2)).replies[-1]


@pytest.mark.parametrize("text, expected", [
    ("acme/demo", "https://github.com/acme/demo"),
    ("https://github.com/acme/demo", "https://github.com/acme/demo"),
    ("https://github.com/acme/demo.git", "https://github.com/acme/demo"),
    ("https://github.com/Acme-1/my.repo_x/", "https://github.com/Acme-1/my.repo_x"),
    ("acme/..", None),
    ("../etc", None),
])
def test_normalize_repo(text, expected):
    assert git_ops.normalize_repo(text) == expected


# /memory


async def test_memory(tg, approved, store, no_clone):
    approved(1)
    assert "/tasks" in (await tg(bot.memory, "/memory")).replies[-1]
    await tg(bot.tasks, "/tasks a acme/demo")
    assert "пуста" in (await tg(bot.memory, "/memory")).replies[-1]
    await tg(bot.memory, "/memory используем FastAPI\nи pytest")
    assert store.get_task(1, "a")["memory"] == "используем FastAPI\nи pytest"
    assert "используем FastAPI" in (await tg(bot.memory, "/memory")).replies[-1]
    await tg(bot.memory, "/memory только pytest")
    assert store.get_task(1, "a")["memory"] == "только pytest"

    await tg(bot.tasks, "/tasks b acme/other")
    assert "пуста" in (await tg(bot.memory, "/memory")).replies[-1]
    assert "[память]" in (await tg(bot.tasks, "/tasks")).replies[-1].split("\n")[1]

    await tg(bot.tasks, "/tasks a")
    reply = (await tg(bot.memory, "/memory " + "x" * (config.MEMORY_MAX_CHARS + 1))).replies[-1]
    assert "не изменена" in reply
    assert store.get_task(1, "a")["memory"] == "только pytest"

    await tg(bot.reset_command, "/reset")
    assert store.get_task(1, "a")["memory"] == "только pytest"


# AGENTS.md


class FakeDocument:
    def __init__(self, name: str, data: bytes):
        self.file_name = name
        self.file_size = len(data)
        self._data = data

    async def get_file(self):
        return SimpleNamespace(download_as_bytearray=self._download)

    async def _download(self):
        return bytearray(self._data)


async def test_agents_md(tg, approved, store, monkeypatch):
    approved(1)
    assert "не задан" in (await tg(bot.agents, "/agents")).replies[-1]
    await tg(bot.agents_document, document=FakeDocument("AGENTS.md", "# Rules\nuse ruff".encode()))
    assert store.get_user(1)["agents_md"] == "# Rules\nuse ruff"
    assert "use ruff" in (await tg(bot.agents, "/agents")).replies[-1]
    assert ".md" in (await tg(bot.agents_document, document=FakeDocument("x.txt", b"x"))).replies[-1]
    monkeypatch.setattr(config, "AGENTS_MD_MAX_CHARS", 5)
    reply = (await tg(bot.agents_document, document=FakeDocument("notes.md", b"123456"))).replies[-1]
    assert "лимит" in reply and store.get_user(1)["agents_md"] == "# Rules\nuse ruff"
    await tg(bot.agents_clear, "/agents_clear")
    assert store.get_user(1)["agents_md"] is None


# sending


def test_split_message_keeps_code_blocks_whole():
    text = "intro\n```python\n" + "\n".join(f"x{i} = {i}" for i in range(2000)) + "\n```\noutro"
    parts = bot.split_message(text, limit=1000)
    assert len(parts) > 2
    assert all(len(part) <= 1000 for part in parts)
    for part in parts:
        assert part.count("```") % 2 == 0, part[:80]
    assert parts[1].startswith("```python")
    joined = "\n".join(parts)
    assert "x1999 = 1999" in joined and joined.endswith("outro")


def test_to_html_escapes_and_formats():
    assert bot.to_html("a < b **bold** `x<y`\n```\nif a<b:\n```") == (
        "a &lt; b <b>bold</b> <code>x&lt;y</code>\n<pre>if a&lt;b:</pre>"
    )


async def test_send_falls_back_to_plain_text():
    from telegram.error import BadRequest

    sent = []

    class Message:
        async def reply_text(self, text, parse_mode=None):
            if parse_mode:
                raise BadRequest("can't parse entities")
            sent.append(text)

    await bot.send_text(Message(), "**x**")
    assert sent == ["**x**"]
