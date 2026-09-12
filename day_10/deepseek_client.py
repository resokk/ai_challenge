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
from storage import DEFAULT_BRANCH, MessageStore, count_tokens

logger = logging.getLogger(__name__)


class BroadcastResult(NamedTuple):
    model: str
    content: str
    usage: dict | None
    elapsed_seconds: float


class TokenUsage(NamedTuple):
    """Token cost of one exchange. `request` and `history` are counted locally
    with the same tokenizer the store uses, so they are estimates; `reply` and
    `total` come from the API's own usage report when it sends one."""

    request: int  # the user's latest message on its own
    history: int  # every message in the dialog history sent with it
    reply: int  # the model's reply
    total: int  # the whole exchange, as billed


class CompressionResult(NamedTuple):
    """What one round of history compression cost and saved."""

    mode: str  # the strategy that ran
    folded: int  # messages taken out of the context
    tokens_before: int  # tokens the dialog history held before
    tokens_after: int  # tokens it holds now, whatever replaced them included
    detail: str = ""  # what that mode did, in the mode's own terms


class DeepSeekClientInterface(ABC):
    """Per-chat interface to a DeepSeek-backed assistant. An implementation owns
    that chat's message history and exposes it read-only via `history`."""

    @property
    @abstractmethod
    def history(self) -> list[dict]: ...

    @property
    @abstractmethod
    def last_token_usage(self) -> TokenUsage | None: ...

    @abstractmethod
    def add_user_message(self, content: str) -> None: ...

    @abstractmethod
    def add_assistant_message(self, content: str) -> None: ...

    @abstractmethod
    def discard_last_user_message(self) -> None: ...

    @abstractmethod
    def clear_history(self) -> None: ...

    @abstractmethod
    async def compress_history(self, threshold: int, mode: str) -> CompressionResult | None: ...

    @property
    @abstractmethod
    def branch(self) -> str: ...

    @property
    @abstractmethod
    def facts(self) -> list[str]: ...

    @abstractmethod
    def switch_branch(self, branch: str) -> None: ...

    @abstractmethod
    def list_branches(self) -> list[str]: ...

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
    MAX_HISTORY_MESSAGES = 1000

    VALID_MODES = ("DIRECT", "STEP_BY_STEP", "SMART_PROMPT", "TEAM")
    DEFAULT_MODE = "DIRECT"

    VALID_TEMPERATURES = (0, 0.2, 0.5, 0.7, 1, 1.3, 1.5, 1.8, 2)
    DEFAULT_TEMPERATURE = 1

    # How many messages the history may reach before its oldest half is folded
    # into a summary. 0 disables compression.
    VALID_COMPRESSIONS = (0, 10, 20, 50, 100)
    DEFAULT_COMPRESSION = 0

    # What to do when a conversation reaches the compression threshold.
    VALID_COMPRESSION_MODES = ("SUMMARIZE", "WINDOW", "BRANCH", "FACTS")
    DEFAULT_COMPRESSION_MODE = "SUMMARIZE"

    MAX_BRANCH_NAME_LENGTH = 32

    # Summaries and fact extraction are fidelity jobs, not creative ones, so
    # they ignore the chat's temperature and always run cold.
    SUMMARY_TEMPERATURE = 0.2
    SUMMARY_PREFIX = "Summary of the earlier part of our conversation:\n"
    SUMMARY_SYSTEM_PROMPT = (
        "You are compressing the older part of a chat conversation so it can be dropped from "
        "the context window. Write a compact set of notes that preserves the facts, decisions, "
        "names, numbers, preferences, and open questions a later reply would still need. Write "
        "notes, not dialogue, and never add anything that was not said."
    )

    SMART_PROMPT_SYSTEM_PROMPT = (
        "You are a prompt engineer. Rewrite the user's latest message into a clear, detailed, "
        "well-structured prompt that another AI assistant can follow to give the best possible "
        "answer. Preserve the original intent and any constraints implied by the conversation. "
        "Output only the rewritten prompt, with no extra commentary."
    )

    FACTS_PREFIX = "Known facts about this conversation:\n"
    FACTS_SYSTEM_PROMPT = (
        "You extract durable facts from one exchange of a chat. A durable fact is something "
        "that stays true afterwards and a later reply would need: names, preferences, "
        "decisions, numbers, constraints, commitments. Give one fact per line, each a short "
        "standalone statement with no bullet or numbering. If the exchange contains no such "
        "fact, reply with nothing at all."
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

    def __init__(self, store: MessageStore, branch: str = DEFAULT_BRANCH) -> None:
        self._store = store
        self._branch = branch
        self._history: list[dict] = store.load_history(self.MAX_HISTORY_MESSAGES, branch)
        self._facts: list[str] = store.load_facts(branch)
        # Only set once this process has answered a message: token counts
        # describe an exchange, and reloaded history has none behind it.
        self._last_usage: TokenUsage | None = None

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def branch(self) -> str:
        return self._branch

    @property
    def facts(self) -> list[str]:
        return self._facts

    def switch_branch(self, branch: str) -> None:
        """Point this client at another conversation. Branches are independent:
        each has its own messages, summary and facts, and nothing is copied
        between them."""
        self._branch = branch
        self._history = self._store.load_history(self.MAX_HISTORY_MESSAGES, branch)
        self._facts = self._store.load_facts(branch)
        self._last_usage = None

    def list_branches(self) -> list[str]:
        return self._store.list_branches()

    @property
    def last_token_usage(self) -> TokenUsage | None:
        return self._last_usage

    # ---- history management ----
    # The store is the durable copy of this history, so every change here is
    # mirrored to it: what survives a restart is exactly what is in memory.

    def add_user_message(self, content: str) -> None:
        self._store.record("user", content, self._branch)
        self.history.append({"role": "user", "content": content})
        self.history[:] = self.history[-self.MAX_HISTORY_MESSAGES :]

    def add_assistant_message(self, content: str) -> None:
        self._store.record("assistant", content, self._branch)
        self.history.append({"role": "assistant", "content": content})

    def discard_last_user_message(self) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
            self._store.discard_last_incoming(self._branch)

    def clear_history(self) -> None:
        self._history = []
        self._facts = []
        self._last_usage = None
        self._store.clear(self._branch)

    async def compress_history(
        self, threshold: int, mode: str = DEFAULT_COMPRESSION_MODE
    ) -> CompressionResult | None:
        """Once the conversation has grown to `threshold` messages, shrink it
        the way `mode` says. Returns what the round cost and saved, or None if
        nothing needed doing.

        Compression is maintenance, not part of answering the user, so a mode
        that fails is logged and skipped rather than raised: the conversation
        simply stays as it is until the next message."""
        if not threshold:
            return None

        # FACTS learns from every exchange, so it runs before the size check;
        # the others only have something to do once the conversation is long.
        if mode == "FACTS":
            return await self._compress_facts(threshold)
        if len(self.history) < threshold:
            return None
        if mode == "WINDOW":
            return self._compress_window(threshold)
        if mode == "BRANCH":
            return self._compress_branch()
        return await self._compress_summarize(threshold)

    async def _compress_summarize(self, threshold: int) -> CompressionResult | None:
        """Replace the oldest half of the conversation with a summary of it."""
        half = len(self.history) // 2
        # The summary is an assistant note, so let the surviving half start on a
        # user turn to keep the conversation alternating.
        while half < len(self.history) and self.history[half]["role"] != "user":
            half += 1
        if half >= len(self.history):
            return None

        try:
            summary = await self._summarize(self.history[:half])
        except Exception:
            logger.exception("Failed to summarize the oldest %s messages", half)
            return None

        summary = self._strip_summary_heading(summary)
        if not summary:
            logger.warning("Summarization returned an empty summary, leaving the conversation uncompressed")
            return None

        tokens_before = self._history_tokens()
        # Everything before the split is folded away, so any earlier summary
        # (always at index 0) goes with it and `kept` is purely logged messages
        # - which is what the store counts back from.
        kept = self.history[half:]
        summary_message = {"role": "assistant", "content": f"{self.SUMMARY_PREFIX}{summary}"}
        self._history = [summary_message, *kept]
        self._store.record_summary(summary_message["content"], len(kept), self._branch)
        return CompressionResult(
            "SUMMARIZE", half, tokens_before, self._history_tokens(), "folded the oldest messages into a summary"
        )

    def _compress_window(self, threshold: int) -> CompressionResult | None:
        """Keep only the newest `threshold` messages and drop the rest. No model
        call, no summary: what falls out of the window is simply forgotten."""
        dropped = len(self.history) - threshold
        if dropped <= 0:
            return None
        tokens_before = self._history_tokens()
        self._history = self.history[-threshold:]
        self._store.trim_to_last(threshold, self._branch)
        return CompressionResult(
            "WINDOW", dropped, tokens_before, self._history_tokens(), f"keeping the newest {threshold} messages"
        )

    def _compress_branch(self) -> CompressionResult:
        """Leave the conversation where it is and continue in a fresh branch.

        Nothing is summarized or deleted: the context is empty again, and the
        full conversation is still there under its own name to switch back to."""
        previous, tokens_before, folded = self._branch, self._history_tokens(), len(self.history)
        self.switch_branch(self._next_branch_name())
        return CompressionResult(
            "BRANCH", folded, tokens_before, 0, f"continuing in '{self._branch}'; '{previous}' is kept as it was"
        )

    def _next_branch_name(self) -> str:
        """The current branch's name with the lowest free -N suffix."""
        stem = self._branch.rsplit("-", 1)[0] if self._branch.rsplit("-", 1)[-1].isdigit() else self._branch
        taken = set(self.list_branches())
        number = 2
        while f"{stem}-{number}" in taken:
            number += 1
        return f"{stem}-{number}"

    async def _compress_facts(self, threshold: int) -> CompressionResult | None:
        """Learn facts from the latest exchange, then keep only the newest
        `threshold` messages. The messages are forgotten like WINDOW, but what
        they established is carried forward as facts instead."""
        tokens_before = self._history_tokens()

        learned = []
        if len(self.history) >= 2:
            try:
                learned = self._parse_facts(await self._extract_facts(self.history[-2:]))
            except Exception:
                logger.exception("Failed to extract facts from the latest exchange")
        for fact in learned:
            self._facts.append(fact)
            self._store.record_fact(fact, self._branch)

        dropped = max(len(self.history) - threshold, 0)
        if dropped:
            self._history = self.history[-threshold:]
            self._store.trim_to_last(threshold, self._branch)
        if not dropped and not learned:
            return None

        detail = f"learned {len(learned)} fact(s), {len(self._facts)} known"
        return CompressionResult("FACTS", dropped, tokens_before, self._history_tokens(), detail)

    def _parse_facts(self, extracted: str) -> list[str]:
        """One fact per line, minus any bullet the model added, minus anything
        already known - facts are a set, and a repeated one is not new."""
        known = {fact.lower() for fact in self._facts}
        facts = []
        for line in (extracted or "").splitlines():
            fact = line.strip().lstrip("-*•").strip()
            if fact and fact.lower() not in known:
                known.add(fact.lower())
                facts.append(fact)
        return facts

    @classmethod
    def _strip_summary_heading(cls, summary: str) -> str:
        """Drop any heading the model copied from the summary it was given.

        Each round summarizes a transcript that contains the previous summary,
        heading line and all, and the model tends to reproduce that line - so
        without this the headings stack up, one per round, forever."""
        heading = cls.SUMMARY_PREFIX.strip()
        text = summary.strip()
        while text.startswith(heading):
            text = text[len(heading) :].lstrip()
        return text

    async def _summarize(self, messages: list[dict]) -> str:
        transcript = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)
        request = [
            {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        return await self._complete(request, response_format="text", temperature=self.SUMMARY_TEMPERATURE)

    async def _extract_facts(self, messages: list[dict]) -> str:
        exchange = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)
        request = [
            {"role": "system", "content": self.FACTS_SYSTEM_PROMPT},
            {"role": "user", "content": exchange},
        ]
        return await self._complete(request, response_format="text", temperature=self.SUMMARY_TEMPERATURE)

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

    def _history_tokens(self) -> int:
        return sum(count_tokens(message["content"]) for message in self.history)

    def _measure_usage(self, reply: str, usage: dict | None) -> TokenUsage:
        last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
        request = count_tokens(last_user)
        # self.history still ends with the message being answered, so this is
        # the full dialog context the model just saw.
        history = self._history_tokens()
        reply_tokens = usage["completion_tokens"] if usage else count_tokens(reply or "")
        total = usage["total_tokens"] if usage else history + reply_tokens
        return TokenUsage(request, history, reply_tokens, total)

    async def _complete_and_record(
        self,
        messages: list[dict],
        response_format: str,
        max_tokens: int | None,
        stop: list[str] | None,
        temperature: float,
    ) -> str:
        """Complete a turn of this conversation and remember what it cost, so
        /stats can report the tokens behind the last message."""
        content, usage = await self._complete(
            messages, response_format, max_tokens, stop, temperature, return_usage=True
        )
        self._last_usage = self._measure_usage(content, usage)
        return content

    def _context(self, messages: list[dict] | None = None) -> list[dict]:
        """The conversation as the model should see it: what is known first,
        then the messages. In FACTS mode the messages are a short window, and
        this is what carries everything the dropped ones established."""
        conversation = self.history if messages is None else messages
        if not self._facts:
            return list(conversation)
        facts = {"role": "assistant", "content": self.FACTS_PREFIX + "\n".join(self._facts)}
        return [facts, *conversation]

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
            *self._context(),
        ]
        return await self._complete_and_record(messages, response_format, max_tokens, stop, temperature)

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
            *self._context(outgoing),
        ]
        return await self._complete_and_record(messages, response_format, max_tokens, stop, temperature)

    async def generate_smart_prompt(self, temperature: float = DEFAULT_TEMPERATURE) -> str:
        messages = [{"role": "system", "content": self.SMART_PROMPT_SYSTEM_PROMPT}, *self._context()]
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
            *self._context(outgoing),
        ]
        return await self._complete_and_record(messages, response_format, max_tokens, stop, temperature)

    async def _get_role_draft(self, role_prompt: str, system_prompt: str, temperature: float) -> str:
        messages = [{"role": "system", "content": f"{system_prompt}\n\n{role_prompt}"}, *self._context()]
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
        messages = [{"role": "system", "content": critic_system_prompt}, *self._context(critic_history)]
        return await self._complete_and_record(messages, response_format, max_tokens, stop, temperature)

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
