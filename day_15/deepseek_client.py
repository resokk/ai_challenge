from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

import protocol_log
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Chat completions against the DeepSeek API, over a conversation history
    the caller owns."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant chatting with a user over Telegram. Keep replies concise."

    # A summary is a fidelity job, not a creative one, so it runs cold. The
    # prefix labels the summary inside the history it stands in for, so the
    # model reads it as notes on an earlier conversation rather than as a turn.
    SUMMARY_TEMPERATURE = 0.2
    SUMMARY_PREFIX = "Summary of the earlier part of our conversation:\n"
    SUMMARY_SYSTEM_PROMPT = (
        "You are compressing a chat conversation so it can be dropped from the context window. "
        "Write a compact set of notes that preserves the facts, decisions, names, numbers, "
        "preferences, and open questions a later reply would still need. Write notes, not "
        "dialogue, and never add anything that was not said."
    )

    # HTTP/2, because HTTP/1.1 response bodies from api.deepseek.com stall
    # indefinitely on some networks: the headers arrive, the body never does.
    # The timeout is the backstop for any other stall - a reply the user never
    # gets is worse than an error they do.
    _client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=1,
        http_client=httpx.AsyncClient(http2=True, timeout=httpx.Timeout(60.0, connect=10.0)),
    )

    # A run of tool calls is the model working, not a conversation, so it is
    # bounded: whatever it has after this many rounds is what the user gets.
    MAX_TOOL_ROUNDS = 4

    @classmethod
    async def get_reply(
        cls,
        history: list[dict],
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        call_tool: Callable[[str, dict], Awaitable[str]] | None = None,
    ) -> str:
        """The model's answer to the conversation. Given `tools` and a `call_tool`
        to run them, the model may act before answering: each round its calls are
        run and their results handed back, until it replies with words.

        The rounds are local to this call - only the answer goes back to the
        caller - so a conversation stays what was said, not how it was served."""
        system_prompt = system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        logger.info("System prompt (%d chars, %d messages): %s", len(system_prompt), len(history), system_prompt)

        messages = [{"role": "system", "content": system_prompt}, *history]
        options = {"tools": tools} if tools and call_tool else {}

        for _ in range(cls.MAX_TOOL_ROUNDS):
            message = await cls._complete(messages, **options)
            if not message.tool_calls:
                return message.content or ""
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": await cls._run_tool(call, call_tool),
                    }
                )

        logger.warning("Gave up after %d rounds of tool calls", cls.MAX_TOOL_ROUNDS)
        return message.content or ""

    @staticmethod
    async def _run_tool(call, call_tool: Callable[[str, dict], Awaitable[str]]) -> str:
        """Run one call the model asked for. Whatever goes wrong - arguments that
        are not JSON, a tool that raises - is reported back to the model as the
        result, which is the one thing it can do something about."""
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            return "The arguments were not valid JSON. Send them again."
        try:
            return await call_tool(call.function.name, arguments)
        except Exception as error:
            logger.exception("Tool %s failed", call.function.name)
            return f"The call failed: {error!r}"

    @classmethod
    async def _complete(cls, messages: list[dict], **kwargs):
        """Every call to the model goes through here, so both halves of it reach
        the protocol log without either caller remembering to write them.

        The request is logged as the whole body that goes over the wire - the
        history being sent included, verbatim - rather than a rendering of it,
        so the log answers what was actually asked and not just roughly what.
        The reply is logged whole for the same reason: what the model did is in
        the tool calls it made, not only in the words it wrote."""
        request = {"model": DEEPSEEK_MODEL, "messages": messages, **kwargs}
        # ensure_ascii=False: a log of the conversation should be readable in
        # whatever language it was held in.
        protocol_log.record("bot -> deepseek", json.dumps(request, ensure_ascii=False, indent=2))
        try:
            response = await cls._client.chat.completions.create(**request)
        except Exception as error:
            # So a request in the log is never left without an outcome.
            protocol_log.record("deepseek -> bot", f"<call failed: {error!r}>")
            raise
        message = response.choices[0].message
        protocol_log.record(
            "deepseek -> bot", json.dumps(message.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
        )
        return message

    @staticmethod
    def _transcript(messages: list[dict]) -> str:
        """A conversation as flat text, for a model being asked to read one
        rather than continue it."""
        return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)

    @classmethod
    async def summarize(cls, history: list[dict]) -> str:
        """Notes standing in for a whole conversation, ready to be the history a
        later one resumes from. The transcript may already contain an earlier
        summary, which this one folds in - so the heading it may copy is stripped."""
        message = await cls._complete(
            [
                {"role": "system", "content": cls.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": cls._transcript(history)},
            ],
            temperature=cls.SUMMARY_TEMPERATURE,
        )
        summary = (message.content or "").strip()
        heading = cls.SUMMARY_PREFIX.strip()
        return summary[len(heading) :].lstrip() if summary.startswith(heading) else summary
