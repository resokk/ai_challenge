"""What the model is sent: the fixed prefix and the chat history, and the
summarization that keeps the history within the context limit."""

from __future__ import annotations

import json
import logging

import config
import llm
from storage import Storage

logger = logging.getLogger(__name__)

AGENTS_HEADING = "Инструкции проекта пользователя (AGENTS.md)"
SUMMARY_HEADING = "Сводка предыдущей части диалога:"

SUMMARIZE_PROMPT = (
    "Ты сжимаешь начало диалога пользователя с ассистентом Python-разработчика, чтобы освободить "
    "контекст. Напиши краткую сводку. Сохрани: цели пользователя, принятые решения, имена файлов, "
    "функций, классов и пакетов, версии, команды, нерешённые вопросы и ключевые фрагменты кода. "
    "Если в начале есть предыдущая сводка — включи её содержание. Пиши кратко, без вступлений, "
    "ничего не добавляй от себя."
)

DIALOG_SUMMARY_TASK = (
    "Кратко изложи этот диалог: цель, что сделано и решено, ключевые файлы, код и команды, "
    "открытые вопросы и следующие шаги."
)


def system_prompt() -> str:
    return (config.PROMPTS_DIR / "system.md").read_text(encoding="utf-8").strip()


def prefix(user, task, catalog: str | None) -> list[dict]:
    """Parts 1-3: rebuilt for every request, never stored or summarized.
    Without a catalog (for requests without tools) part 3 is left out."""
    text = system_prompt()
    if task is not None:
        text += f"\n\nТекущая задача: `{task['task_id']}`, репозиторий: {task['repo_url']}"
        if task["memory"]:
            text += f"\n\nПамять задачи:\n{task['memory']}"
    messages = [{"role": "system", "content": text}]
    if user["agents_md"]:
        messages.append({"role": "system", "content": f"{AGENTS_HEADING}\n\n{user['agents_md']}"})
    if catalog is not None:
        messages.append({"role": "system", "content": catalog})
    return messages


def history_messages(rows, with_tools: bool = True) -> list[dict]:
    """Part 4 as API messages. Without tools, `tool` messages and tool calls
    are dropped, and so is an assistant message that was only a tool call."""
    messages = []
    for row in _complete_groups(rows):
        if row["role"] == "summary":
            messages.append({"role": "system", "content": f"{SUMMARY_HEADING}\n{row['content']}"})
        elif row["role"] == "tool":
            if with_tools:
                messages.append({"role": "tool", "tool_call_id": row["tool_call_id"], "content": row["content"]})
        elif row["role"] == "assistant" and row["tool_calls"]:
            if with_tools:
                messages.append({"role": "assistant", "content": row["content"] or "",
                                 "tool_calls": json.loads(row["tool_calls"])})
            elif row["content"]:
                messages.append({"role": "assistant", "content": row["content"]})
        else:
            messages.append({"role": row["role"], "content": row["content"]})
    return messages


def _complete_groups(rows) -> list:
    """Drop a tool-call group whose results never all arrived (the turn died
    mid-way): the API rejects a call without its answer."""
    rows = list(rows)
    keep = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row["role"] == "assistant" and row["tool_calls"]:
            expected = {call["id"] for call in json.loads(row["tool_calls"])}
            j = i + 1
            while j < len(rows) and rows[j]["role"] == "tool":
                j += 1
            if {r["tool_call_id"] for r in rows[i + 1:j]} >= expected:
                keep += rows[i:j]
            i = j
        elif row["role"] == "tool":
            i += 1  # orphaned result
        else:
            keep.append(row)
            i += 1
    return keep


def build(store: Storage, telegram_id: int, catalog: str) -> list[dict]:
    """The full request: system prompt, AGENTS.md, tool catalog, history."""
    user = store.get_user(telegram_id)
    return prefix(user, store.current_task(telegram_id), catalog) + history_messages(store.history(telegram_id))


def dialog_summary_request(store: Storage, telegram_id: int) -> list[dict] | None:
    """The `/summary` request: no catalog, no tool traffic. None when there is
    nothing to summarize."""
    rows = store.history(telegram_id)
    if not rows:
        return None
    user = store.get_user(telegram_id)
    return (
        prefix(user, store.current_task(telegram_id), None)
        + history_messages(rows, with_tools=False)
        + [{"role": "user", "content": DIALOG_SUMMARY_TASK}]
    )


def needs_summary(user) -> bool:
    return user["last_tokens"] >= config.SUMMARIZE_AT * config.CONTEXT_LIMIT_TOKENS


def split_point(rows) -> int:
    """How many of the oldest rows to compress: two thirds, moved forward so a
    tool-call group is never cut - the whole group goes into the summary."""
    k = len(rows) * 2 // 3
    while 0 < k < len(rows) and rows[k]["role"] == "tool":
        k += 1
    return k


def _transcript(rows) -> str:
    lines = []
    for row in rows:
        if row["role"] == "summary":
            lines.append(f"[Предыдущая сводка]\n{row['content']}")
        elif row["role"] == "tool":
            lines.append(f"[Результат инструмента]\n{(row['content'] or '')[:2000]}")
        else:
            text = row["content"] or ""
            if row["tool_calls"]:
                calls = [call["function"]["arguments"] for call in json.loads(row["tool_calls"])]
                text += "".join(f"\n[вызов call_tool {args[:500]}]" for args in calls)
            lines.append(f"[{row['role']}]\n{text}")
    return "\n\n".join(lines)


async def summarize(store: Storage, telegram_id: int) -> bool:
    """Fold the oldest two thirds of the history (old summary included) into
    one summary row. On any failure the history stays as it was."""
    rows = store.history(telegram_id)
    k = split_point(rows)
    if k < 2:
        return False
    covered = rows[:k]
    try:
        summary = await llm.text_reply(
            [{"role": "system", "content": SUMMARIZE_PROMPT}, {"role": "user", "content": _transcript(covered)}],
            temperature=0.2,
        )
    except Exception:
        logger.exception("Summarization failed for %s; history kept", telegram_id)
        return False
    if not summary:
        logger.warning("Empty summary for %s; history kept", telegram_id)
        return False
    store.replace_with_summary(telegram_id, summary, [row["id"] for row in covered])
    return True
