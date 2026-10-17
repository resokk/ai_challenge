"""The help MCP server: one tool, `help`, answering questions about the bot
from its help articles. Streamable HTTP on 127.0.0.1:8770/mcp.

    python server.py
"""

from __future__ import annotations

import logging
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import answer
import embedder

HOST = "127.0.0.1"
PORT = 8770
PATH = "/mcp"

DESCRIPTION = (
    "Справка по этому Telegram-боту. Вызывай на любой вопрос о самом боте: «как в этом боте сделать X», "
    "как получить доступ, команды и их параметры, задачи и репозитории GitHub, git и SSH-ключи, приватные "
    "репозитории, запуск кода (/run) и установка пакетов (/pip), venv, история, /context, /summary, /reset, "
    "память задачи и AGENTS.md, инструменты (/tools, /tool), администрирование, ошибки и сообщения бота, "
    "примеры сценариев. `question` — вопрос пользователя его словами. Возвращает готовый короткий ответ "
    "со ссылками на статьи справки."
)

logger = logging.getLogger(__name__)

mcp = MCPServer("help", instructions="Справка по Telegram-боту — ассистенту Python-разработчика.")


@mcp.tool(name="help", description=DESCRIPTION, structured_output=False)
async def help_tool(question: str) -> str:
    try:
        return await answer.answer(question)
    except answer.QuestionError as error:
        raise ToolError(str(error)) from error


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
    if not answer.api_key():
        print("Ошибка: в help/.env не задан DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)
    # Index and model are ready before the first request is accepted.
    try:
        index = answer.get_index()
        embedder.embed_query("проверка")
    except (embedder.EmbedderError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
    logger.info("Help index: %d chunks from %d articles", len(index.chunks), len(index.categories))
    mcp.run("streamable-http", host=HOST, port=PORT, streamable_http_path=PATH, json_response=True,
            stateless_http=True)


if __name__ == "__main__":
    main()
