from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET

from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant chatting with a user over Telegram. Keep replies concise."

VALID_RESPONSE_FORMATS = ("text", "markdown", "json", "xml")
DEFAULT_RESPONSE_FORMAT = "text"

# DeepSeek's API only knows "text"/"json_object" - "markdown" is a Telegram-side
# rendering choice and "xml" is requested as JSON then converted client-side
# (see bot.py / json_to_xml below), so both still hit the API as one of those two.
API_FORMAT_TYPES = {"text": "text", "markdown": "text", "json": "json_object", "xml": "json_object"}

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


MAX_STOP_SEQUENCES = 4

VALID_MODES = ("DIRECT", "STEP_BY_STEP", "SMART_PROMPT", "TEAM")
DEFAULT_MODE = "DIRECT"


def _sanitize_tag(key: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
    if not tag or not (tag[0].isalpha() or tag[0] == "_"):
        tag = f"_{tag}"
    return tag


def _append_xml(parent: ET.Element, tag: str, value: object) -> None:
    if isinstance(value, dict):
        child = ET.SubElement(parent, tag)
        for key, item in value.items():
            _append_xml(child, _sanitize_tag(key), item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                # A bare list-of-lists would otherwise flatten into siblings
                # under the same tag, losing the nested grouping. Wrap each
                # nested list in its own container element instead.
                child = ET.SubElement(parent, tag)
                for sub_item in item:
                    _append_xml(child, "item", sub_item)
            else:
                _append_xml(parent, tag, item)
    else:
        child = ET.SubElement(parent, tag)
        child.text = "" if value is None else str(value)


def json_to_xml(data: object, root_tag: str = "response") -> str:
    root = ET.Element(root_tag)
    if isinstance(data, dict):
        for key, value in data.items():
            _append_xml(root, _sanitize_tag(key), value)
    elif isinstance(data, list):
        for item in data:
            _append_xml(root, "item", item)
    else:
        root.text = str(data)
    return ET.tostring(root, encoding="unicode")


def _augment_system_prompt(system_prompt: str, response_format: str) -> str:
    if response_format in ("json", "xml") and "json" not in system_prompt.lower():
        return f"{system_prompt}\nRespond only with a valid JSON object."
    if response_format == "markdown":
        return f"{system_prompt}\nFormat your response using Telegram-compatible Markdown."
    return system_prompt


async def _complete(
    messages: list[dict],
    response_format: str = DEFAULT_RESPONSE_FORMAT,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> str:
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop:
        kwargs["stop"] = stop
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        response_format={"type": API_FORMAT_TYPES[response_format]},
        **kwargs,
    )
    content = response.choices[0].message.content

    if response_format == "xml" and content:
        try:
            content = json_to_xml(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to convert JSON reply to XML, returning raw JSON")

    return content


async def get_reply(
    history: list[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> str:
    messages = [{"role": "system", "content": _augment_system_prompt(system_prompt, response_format)}, *history]
    return await _complete(messages, response_format, max_tokens, stop)


SMART_PROMPT_SYSTEM_PROMPT = (
    "You are a prompt engineer. Rewrite the user's latest message into a clear, detailed, "
    "well-structured prompt that another AI assistant can follow to give the best possible "
    "answer. Preserve the original intent and any constraints implied by the conversation. "
    "Output only the rewritten prompt, with no extra commentary."
)


async def generate_smart_prompt(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SMART_PROMPT_SYSTEM_PROMPT}, *history]
    return await _complete(messages, response_format="text")


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


async def _get_role_draft(history: list[dict], system_prompt: str, role_prompt: str) -> str:
    messages = [{"role": "system", "content": f"{system_prompt}\n\n{role_prompt}"}, *history]
    return await _complete(messages, response_format="text")


async def get_team_reply(
    history: list[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> str:
    roles = list(TEAM_ROLE_PROMPTS)
    drafts = await asyncio.gather(
        *(_get_role_draft(history, system_prompt, TEAM_ROLE_PROMPTS[role]) for role in roles)
    )
    draft_summary = "\n\n".join(f"{role.capitalize()} draft:\n{draft}" for role, draft in zip(roles, drafts))

    critic_history = [
        *history[:-1],
        {"role": "user", "content": f"{history[-1]['content']}\n\n---\nTeam drafts to synthesize:\n\n{draft_summary}"},
    ]
    critic_system_prompt = _augment_system_prompt(f"{system_prompt}\n\n{TEAM_CRITIC_PROMPT}", response_format)
    messages = [{"role": "system", "content": critic_system_prompt}, *critic_history]
    return await _complete(messages, response_format, max_tokens, stop)
