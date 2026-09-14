from __future__ import annotations

import logging

import httpx
from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Chat completions against the DeepSeek API, over a conversation history
    the caller owns."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant chatting with a user over Telegram. Keep replies concise."

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

    @classmethod
    async def get_reply(cls, history: list[dict], system_prompt: str | None = None) -> str | None:
        system_prompt = system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        logger.info("System prompt (%d chars, %d messages): %s", len(system_prompt), len(history), system_prompt)
        response = await cls._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": system_prompt}, *history],
        )
        return response.choices[0].message.content
