"""Reference tools: no files, no user code."""

from __future__ import annotations

import builtins
import sys
from datetime import datetime, timezone

import httpx

from run_as import UserContext
from tools.registry import Tool, ToolError

PYPI_URL = "https://pypi.org/pypi/{name}/json"
DOCS_URL = "https://docs.python.org/3/library/"


async def pypi_package_info(ctx: UserContext, arguments: dict) -> str:
    name = arguments["name"].strip()
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(PYPI_URL.format(name=name))
    if response.status_code == 404:
        raise ToolError(f"пакет {name} на PyPI не найден")
    response.raise_for_status()
    info = response.json()["info"]
    links = {"PyPI": info.get("package_url"), "Home": info.get("home_page"), **(info.get("project_urls") or {})}
    lines = [
        f"{info['name']} {info['version']}",
        info.get("summary") or "(нет описания)",
        f"Требуемый Python: {info.get('requires_python') or 'не указан'}",
        "Ссылки:",
        *(f"- {label}: {url}" for label, url in links.items() if url),
    ]
    return "\n".join(lines)


async def python_docs_link(ctx: UserContext, arguments: dict) -> str:
    topic = arguments["topic"].strip().removesuffix("()")
    module = topic.split(".")[0]
    if module in sys.stdlib_module_names:
        anchor = f"#{topic}" if "." in topic else ""
        return f"{DOCS_URL}{module}.html{anchor}"
    if hasattr(builtins, topic):
        page = "exceptions" if isinstance(getattr(builtins, topic), type) and issubclass(
            getattr(builtins, topic), BaseException) else "functions"
        return f"{DOCS_URL}{page}.html#{topic}"
    raise ToolError(f"«{topic}» не найден в стандартной библиотеке")


async def current_datetime(ctx: UserContext, arguments: dict) -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC (%A)")


TOOLS = [
    Tool(
        "pypi_package_info",
        "Сведения о пакете с PyPI: последняя версия, описание, требуемый Python, ссылки.",
        {"type": "object", "properties": {"name": {"type": "string", "minLength": 1}},
         "required": ["name"], "additionalProperties": False},
        pypi_package_info,
    ),
    Tool(
        "python_docs_link",
        "Ссылка на раздел docs.python.org для модуля или функции стандартной библиотеки "
        "(например, `pathlib`, `json.loads`, `len`).",
        {"type": "object", "properties": {"topic": {"type": "string", "minLength": 1}},
         "required": ["topic"], "additionalProperties": False},
        python_docs_link,
    ),
    Tool(
        "current_datetime",
        "Текущие дата и время UTC.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        current_datetime,
    ),
]
