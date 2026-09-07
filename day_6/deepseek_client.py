from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import NamedTuple

from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


class BroadcastResult(NamedTuple):
    model: str
    content: str
    usage: dict | None
    elapsed_seconds: float


class DeepSeekClientInterface(ABC):
    """Per-chat interface to a DeepSeek-backed assistant. An implementation owns
    that chat's message history and exposes it read-only via `history`."""

    @property
    @abstractmethod
    def history(self) -> list[dict]: ...

    @abstractmethod
    def add_user_message(self, content: str) -> None: ...

    @abstractmethod
    def add_assistant_message(self, content: str) -> None: ...

    @abstractmethod
    def discard_last_user_message(self) -> None: ...

    @abstractmethod
    def clear_history(self) -> None: ...

    @abstractmethod
    async def get_reply(
        self,
        system_prompt: str,
        response_format: str,
        max_tokens: int | None,
        stop: list[str] | None,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    async def get_step_by_step_reply(
        self,
        system_prompt: str,
        response_format: str,
        max_tokens: int | None,
        stop: list[str] | None,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    async def generate_smart_prompt(self, temperature: float) -> str: ...

    @abstractmethod
    async def get_reply_with_replacement(
        self,
        replacement_content: str,
        system_prompt: str,
        response_format: str,
        max_tokens: int | None,
        stop: list[str] | None,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    async def get_team_reply(
        self,
        system_prompt: str,
        response_format: str,
        max_tokens: int | None,
        stop: list[str] | None,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    async def get_broadcast_replies(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> list[BroadcastResult]: ...

    @abstractmethod
    async def get_broadcast_and_compare(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> tuple[list[BroadcastResult], list[BroadcastResult]]: ...


class DeepSeekClient(DeepSeekClientInterface):
    """Per-chat interface to the DeepSeek API. Each instance owns one chat's
    message history, so callers never pass history around themselves."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant chatting with a user over Telegram. Keep replies concise."

    VALID_RESPONSE_FORMATS = ("text", "markdown", "json", "xml")
    DEFAULT_RESPONSE_FORMAT = "text"

    # DeepSeek's API only knows "text"/"json_object" - "markdown" is a Telegram-side
    # rendering choice and "xml" is requested as JSON then converted client-side
    # (see json_to_xml below), so both still hit the API as one of those two.
    API_FORMAT_TYPES = {"text": "text", "markdown": "text", "json": "json_object", "xml": "json_object"}

    PRO_MODEL = "deepseek-v4-pro"
    BROADCAST_MODELS = ("deepseek-v4-flash", PRO_MODEL, "deepseek-v4-flash-vision-exp")

    MAX_STOP_SEQUENCES = 4
    MAX_HISTORY_MESSAGES = 20

    VALID_MODES = ("DIRECT", "STEP_BY_STEP", "SMART_PROMPT", "TEAM")
    DEFAULT_MODE = "DIRECT"

    VALID_TEMPERATURES = (0, 0.2, 0.5, 0.7, 1, 1.3, 1.5, 1.8, 2)
    DEFAULT_TEMPERATURE = 1

    SMART_PROMPT_SYSTEM_PROMPT = (
        "You are a prompt engineer. Rewrite the user's latest message into a clear, detailed, "
        "well-structured prompt that another AI assistant can follow to give the best possible "
        "answer. Preserve the original intent and any constraints implied by the conversation. "
        "Output only the rewritten prompt, with no extra commentary."
    )

    TEAM_ROLE_PROMPTS = {
        "analyst": (
            "You are the team's analyst. Break the user's request down into clear requirements, "
            "constraints, and edge cases, and outline your recommended approach."
        ),
        "scientist": (
            "You are the team's scientist. Ground your answer in evidence and first principles, "
            "and clearly flag assumptions or uncertainty."
        ),
        "engineer": (
            "You are the team's engineer. Focus on a concrete, practical, efficient solution or "
            "implementation for the user's request."
        ),
    }

    TEAM_CRITIC_PROMPT = (
        "You are the team's critic and final voice. Review the drafts below from the analyst, "
        "scientist, and engineer, resolve any disagreements between them, and write one clear, "
        "complete, final answer for the user. Do not mention the team, the drafts, or the review "
        "process - just answer as yourself."
    )

    COMPARE_BASE_PROMPT = (
        "You are comparing responses several AI models gave to the same user prompt. Note key "
        "differences, strengths, and weaknesses, and state which response best answers the prompt."
    )

    COMPARE_ROLE_PROMPTS = {
        "caveman": "Speak like a caveman: short, blunt, simple words. No jargon.",
        "engineer": "Speak like a pragmatic engineer: focus on correctness, efficiency, and practical usefulness.",
        "humanitarian": "Speak like a humanitarian: focus on empathy, ethics, and human impact.",
    }

    _client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def __init__(self) -> None:
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return self._history

    # ---- history management ----

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self.history[:] = self.history[-self.MAX_HISTORY_MESSAGES :]

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def discard_last_user_message(self) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()

    def clear_history(self) -> None:
        self._history = []

    # ---- XML conversion helpers ----

    @staticmethod
    def _sanitize_tag(key: str) -> str:
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
        if not tag or not (tag[0].isalpha() or tag[0] == "_"):
            tag = f"_{tag}"
        return tag

    @classmethod
    def _append_xml(cls, parent: ET.Element, tag: str, value: object) -> None:
        if isinstance(value, dict):
            child = ET.SubElement(parent, tag)
            for key, item in value.items():
                cls._append_xml(child, cls._sanitize_tag(key), item)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    # A bare list-of-lists would otherwise flatten into siblings
                    # under the same tag, losing the nested grouping. Wrap each
                    # nested list in its own container element instead.
                    child = ET.SubElement(parent, tag)
                    for sub_item in item:
                        cls._append_xml(child, "item", sub_item)
                else:
                    cls._append_xml(parent, tag, item)
        else:
            child = ET.SubElement(parent, tag)
            child.text = "" if value is None else str(value)

    @classmethod
    def json_to_xml(cls, data: object, root_tag: str = "response") -> str:
        root = ET.Element(root_tag)
        if isinstance(data, dict):
            for key, value in data.items():
                cls._append_xml(root, cls._sanitize_tag(key), value)
        elif isinstance(data, list):
            for item in data:
                cls._append_xml(root, "item", item)
        else:
            root.text = str(data)
        return ET.tostring(root, encoding="unicode")

    # ---- request helpers ----

    @classmethod
    def _augment_system_prompt(cls, system_prompt: str, response_format: str) -> str:
        if response_format in ("json", "xml") and "json" not in system_prompt.lower():
            return f"{system_prompt}\nRespond only with a valid JSON object."
        if response_format == "markdown":
            return f"{system_prompt}\nFormat your response using Telegram-compatible Markdown."
        return system_prompt

    @staticmethod
    def _extract_usage(response) -> dict | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    @classmethod
    async def _complete(
        cls,
        messages: list[dict],
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        model: str = DEEPSEEK_MODEL,
        return_usage: bool = False,
    ) -> str | tuple[str, dict | None]:
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        response = await cls._client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": cls.API_FORMAT_TYPES[response_format]},
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content

        if response_format == "xml" and content:
            try:
                content = cls.json_to_xml(json.loads(content))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to convert JSON reply to XML, returning raw JSON")

        if return_usage:
            return content, cls._extract_usage(response)
        return content

    # ---- public chat interface (reads/writes self.history) ----

    async def get_reply(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        messages = [
            {"role": "system", "content": self._augment_system_prompt(system_prompt, response_format)},
            *self.history,
        ]
        return await self._complete(messages, response_format, max_tokens, stop, temperature)

    async def get_step_by_step_reply(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Like get_reply, but appends an instruction to the last message for
        this request only - the stored history keeps the original text."""
        last = self.history[-1]
        outgoing = [*self.history[:-1], {**last, "content": f"{last['content']}\nProceed step by step."}]
        messages = [
            {"role": "system", "content": self._augment_system_prompt(system_prompt, response_format)},
            *outgoing,
        ]
        return await self._complete(messages, response_format, max_tokens, stop, temperature)

    async def generate_smart_prompt(self, temperature: float = DEFAULT_TEMPERATURE) -> str:
        messages = [{"role": "system", "content": self.SMART_PROMPT_SYSTEM_PROMPT}, *self.history]
        return await self._complete(messages, response_format="text", temperature=temperature)

    async def get_reply_with_replacement(
        self,
        replacement_content: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Like get_reply, but with the last user message swapped out (e.g. for
        a SMART_PROMPT-generated prompt) - stored history is untouched."""
        outgoing = [*self.history[:-1], {"role": "user", "content": replacement_content}]
        messages = [
            {"role": "system", "content": self._augment_system_prompt(system_prompt, response_format)},
            *outgoing,
        ]
        return await self._complete(messages, response_format, max_tokens, stop, temperature)

    async def _get_role_draft(self, role_prompt: str, system_prompt: str, temperature: float) -> str:
        messages = [{"role": "system", "content": f"{system_prompt}\n\n{role_prompt}"}, *self.history]
        return await self._complete(messages, response_format="text", temperature=temperature)

    async def get_team_reply(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        roles = list(self.TEAM_ROLE_PROMPTS)
        drafts = await asyncio.gather(
            *(self._get_role_draft(self.TEAM_ROLE_PROMPTS[role], system_prompt, temperature) for role in roles)
        )
        draft_summary = "\n\n".join(f"{role.capitalize()} draft:\n{draft}" for role, draft in zip(roles, drafts))

        last = self.history[-1]
        critic_history = [
            *self.history[:-1],
            {"role": "user", "content": f"{last['content']}\n\n---\nTeam drafts to synthesize:\n\n{draft_summary}"},
        ]
        critic_system_prompt = self._augment_system_prompt(
            f"{system_prompt}\n\n{self.TEAM_CRITIC_PROMPT}", response_format
        )
        messages = [{"role": "system", "content": critic_system_prompt}, *critic_history]
        return await self._complete(messages, response_format, max_tokens, stop, temperature)

    # ---- broadcast / compare (independent of self.history) ----

    async def _broadcast_one(self, model: str, system_prompt: str, temperature: float, prompt: str) -> BroadcastResult:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        start = time.monotonic()
        try:
            content, usage = await self._complete(
                messages, response_format="text", temperature=temperature, model=model, return_usage=True
            )
        except Exception:
            logger.exception("Broadcast call to %s failed", model)
            content, usage = "(request failed)", None
        return BroadcastResult(model, content, usage, time.monotonic() - start)

    async def get_broadcast_replies(
        self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, temperature: float = DEFAULT_TEMPERATURE
    ) -> list[BroadcastResult]:
        return list(
            await asyncio.gather(
                *(self._broadcast_one(model, system_prompt, temperature, prompt) for model in self.BROADCAST_MODELS)
            )
        )

    async def _compare_one(self, role: str, prompt: str, replies_block: str, temperature: float) -> BroadcastResult:
        system_prompt = f"{self.COMPARE_BASE_PROMPT}\n\n{self.COMPARE_ROLE_PROMPTS[role]}"
        compare_prompt = f"User prompt:\n{prompt}\n\nModel responses:\n\n{replies_block}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": compare_prompt}]
        start = time.monotonic()
        try:
            content, usage = await self._complete(
                messages, response_format="text", temperature=temperature, model=self.PRO_MODEL, return_usage=True
            )
        except Exception:
            logger.exception("Comparison call (%s) failed", role)
            content, usage = "(request failed)", None
        return BroadcastResult(f"{self.PRO_MODEL} comparison, {role} role", content, usage, time.monotonic() - start)

    async def get_broadcast_and_compare(
        self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, temperature: float = DEFAULT_TEMPERATURE
    ) -> tuple[list[BroadcastResult], list[BroadcastResult]]:
        replies = await self.get_broadcast_replies(prompt, system_prompt, temperature)
        replies_block = "\n\n".join(f"{result.model}:\n{result.content}" for result in replies)
        comparisons = await asyncio.gather(
            *(self._compare_one(role, prompt, replies_block, temperature) for role in self.COMPARE_ROLE_PROMPTS)
        )
        return replies, list(comparisons)
