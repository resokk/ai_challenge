"""One registry for built-in and MCP tools, and the one dispatcher both the
model and the user go through to call them."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import jsonschema

import config
from run_as import RunResult, UserContext

logger = logging.getLogger(__name__)

# The only function the model is given; the catalog itself is text in a
# system message.
CALL_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "call_tool",
        "description": "Вызвать инструмент из каталога в системном сообщении.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name", "arguments"],
        },
    },
}


class ToolError(Exception):
    """A failure the caller can act on; its text becomes the result."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[[UserContext, dict], Awaitable[str | RunResult]]
    # For tools whose own work is bounded by a longer limit (pip, user code).
    timeout: float | None = None


@dataclass
class ToolReply:
    """What the caller gets: the text, and for a process run its full result,
    which a user-facing caller shows stream by stream."""

    text: str
    run: RunResult | None = None


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[обрезано: показано {limit} из {len(text)} символов]"


def clip_middle(text: str, limit: int) -> str:
    """Keep the start and the end, which is where output says what happened."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... вырезано {len(text) - 2 * half} символов ...]\n{text[-half:]}"


def format_run(run: RunResult) -> str:
    limit = config.RUN_OUTPUT_MAX_CHARS
    return (
        f"exit_code: {run.exit_code}{' (таймаут)' if run.timed_out else ''}\n"
        f"duration: {run.duration:.1f} s\n"
        f"stdout:\n{clip_middle(run.stdout, limit) or '(пусто)'}\n"
        f"stderr:\n{clip_middle(run.stderr, limit) or '(пусто)'}"
    )


class Registry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"инструмент {tool.name} уже зарегистрирован")
        jsonschema.Draft202012Validator.check_schema(tool.parameters)
        self.tools[tool.name] = tool

    def catalog(self) -> str:
        """The tool list as the model and `/tools` see it."""
        entries = [
            f"### {tool.name}\n{tool.description}\nПараметры (JSON Schema): "
            f"{json.dumps(tool.parameters, ensure_ascii=False)}"
            for tool in self.tools.values()
        ]
        return (
            "Каталог инструментов. Вызывай их только через функцию `call_tool` с полями "
            "`name` (точное имя из каталога) и `arguments` (объект по схеме параметров).\n\n"
            + "\n\n".join(entries)
        )

    async def call_tool(self, ctx: UserContext, name: str, arguments: dict) -> ToolReply:
        """Run a tool for whoever asked. Nothing is sent anywhere, and nothing
        raises: every failure comes back as text the caller can act on."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolReply(f"Ошибка: инструмента «{name}» нет. Доступны: {', '.join(self.tools)}")
        try:
            jsonschema.validate(arguments, tool.parameters)
        except jsonschema.ValidationError as error:
            where = "/".join(str(part) for part in error.absolute_path) or "аргументы"
            return ToolReply(f"Ошибка в аргументах {name} ({where}): {error.message}")
        timeout = tool.timeout or config.TOOL_TIMEOUT_SEC
        try:
            result = await asyncio.wait_for(tool.func(ctx, arguments), timeout)
        except asyncio.TimeoutError:
            return ToolReply(f"Ошибка: {name} не ответил за {timeout:g} с")
        except ToolError as error:
            return ToolReply(f"Ошибка: {error}")
        except Exception as error:
            logger.exception("Tool %s failed", name)
            return ToolReply(f"Ошибка при вызове {name}: {type(error).__name__}: {error}")
        if isinstance(result, RunResult):
            return ToolReply(truncate(format_run(result), config.TOOL_RESULT_MAX_CHARS), result)
        return ToolReply(truncate(str(result), config.TOOL_RESULT_MAX_CHARS))
