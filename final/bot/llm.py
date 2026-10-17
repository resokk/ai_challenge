"""The DeepSeek client and the model -> tool -> model loop."""

from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

import config
from tools.registry import CALL_TOOL_SPEC

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            max_retries=1,
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
    return _client


async def complete(messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None):
    """One chat completion. `tools` is left out of the request entirely when
    there are none, so a request without tools really has none."""
    request = {"model": config.DEEPSEEK_MODEL, "messages": messages}
    if tools:
        request["tools"] = tools
    if temperature is not None:
        request["temperature"] = temperature
    return await client().chat.completions.create(**request)


def used_tokens(response) -> int:
    usage = response.usage
    return (usage.prompt_tokens + usage.completion_tokens) if usage else 0


async def text_reply(messages: list[dict], temperature: float | None = None) -> str:
    """A request with no tools at all, for summaries, diff reviews and commit messages."""
    response = await complete(messages, temperature=temperature)
    return (response.choices[0].message.content or "").strip()


async def run_turn(
    messages: list[dict],
    call_tool: Callable[[str, dict], Awaitable[str]],
    save: Callable[[dict], None],
    on_tool: Callable[[str], Awaitable[None]],
) -> tuple[str, int]:
    """Answer the conversation in `messages`, letting the model call tools
    through `call_tool`. Every message the turn adds - tool calls, their
    results, the answer - goes to `save` as it happens. Returns the answer and
    the tokens of the last request."""
    messages = list(messages)
    tokens = 0
    for _ in range(config.MAX_TOOL_ITERATIONS):
        response = await complete(messages, tools=[CALL_TOOL_SPEC])
        tokens = used_tokens(response)
        message = response.choices[0].message
        if not message.tool_calls:
            return _finish(messages, message.content, save), tokens
        assistant = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [call.model_dump(include={"id", "type", "function"}) for call in message.tool_calls],
        }
        messages.append(assistant)
        save(assistant)
        for call in message.tool_calls:
            result = await _run_call(call, call_tool, on_tool)
            reply = {"role": "tool", "tool_call_id": call.id, "content": result}
            messages.append(reply)
            save(reply)

    logger.warning("Tool loop hit MAX_TOOL_ITERATIONS=%d", config.MAX_TOOL_ITERATIONS)
    response = await complete(messages)
    return _finish(messages, response.choices[0].message.content, save), used_tokens(response)


def _finish(messages: list[dict], content: str | None, save: Callable[[dict], None]) -> str:
    text = (content or "").strip() or "(пустой ответ модели)"
    save({"role": "assistant", "content": text})
    return text


async def _run_call(call, call_tool, on_tool) -> str:
    if call.function.name != CALL_TOOL_SPEC["function"]["name"]:
        return f"Ошибка: функции {call.function.name} нет, используй call_tool."
    try:
        request = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return "Ошибка: аргументы call_tool — не валидный JSON."
    name = request.get("name")
    arguments = request.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return "Ошибка: call_tool ожидает {\"name\": строка, \"arguments\": объект}."
    await on_tool(name)
    return await call_tool(name, arguments)
