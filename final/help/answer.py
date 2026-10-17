"""An answer to a question about the bot: the local index finds the
fragments, DeepSeek turns them into a short answer.

    python answer.py "вопрос"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

import embedder
import index
from articles import HELP_DIR
from index import Hit, Index

load_dotenv(HELP_DIR / ".env")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PROMPT_PATH = HELP_DIR / "prompts" / "answer.md"
MAX_QUESTION_CHARS = 1000
TOP_K = 5
TIMEOUT_SEC = 60
FALLBACK_HEADING = "Суммаризация недоступна, найденные фрагменты справки:"

logger = logging.getLogger(__name__)

_index: Index | None = None
_client: AsyncOpenAI | None = None


class QuestionError(ValueError):
    """A question the help cannot take; the message says why."""


def api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "")


def model() -> str:
    return os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"


def get_index() -> Index:
    global _index
    if _index is None:
        _index = index.load_or_build()
    return _index


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key(), base_url=DEEPSEEK_BASE_URL, timeout=TIMEOUT_SEC, max_retries=1)
    return _client


def check_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise QuestionError("Вопрос пустой: напишите, что нужно сделать в боте.")
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionError(
            f"Вопрос слишком длинный: {len(question)} символов при лимите {MAX_QUESTION_CHARS}. Сформулируйте короче."
        )
    return question


def fragments(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{hit.article_id}] {hit.title} — {hit.section}\n{hit.text}" for hit in hits)


def sources(hits: list[Hit]) -> str:
    seen = dict.fromkeys((hit.title, hit.article_id) for hit in hits)
    return "Статьи: " + ", ".join(f"{title} ({article_id})" for title, article_id in seen)


async def summarize(question: str, hits: list[Hit]) -> str:
    response = await client().chat.completions.create(
        model=model(),
        temperature=0.2,
        messages=[
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8").strip()},
            {"role": "user", "content": f"Вопрос: {question}\n\nФрагменты справки:\n\n{fragments(hits)}"},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("пустой ответ модели")
    return text


async def answer(question: str) -> str:
    """A short answer with the articles it is based on. A DeepSeek failure
    does not fail the answer: the fragments come back as they are."""
    question = check_question(question)
    hits = get_index().search(question, k=TOP_K)
    if not hits:
        return "В справке нет статей."
    try:
        text = await summarize(question, hits)
    except Exception as error:
        logger.warning("DeepSeek summarization failed: %s: %s", type(error).__name__, error)
        text = f"{FALLBACK_HEADING}\n\n{fragments(hits)}"
    return f"{text}\n\n{sources(hits)}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.WARNING)
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print('Использование: python answer.py "вопрос"', file=sys.stderr)
        return 2
    try:
        print(asyncio.run(answer(args[0])))
    except (QuestionError, embedder.EmbedderError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
