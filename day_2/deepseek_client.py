from __future__ import annotations

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


async def get_reply(
    history: list[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> str:
    if response_format in ("json", "xml") and "json" not in system_prompt.lower():
        system_prompt = f"{system_prompt}\nRespond only with a valid JSON object."
    elif response_format == "markdown":
        system_prompt = f"{system_prompt}\nFormat your response using Telegram-compatible Markdown."

    messages = [{"role": "system", "content": system_prompt}, *history]
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
