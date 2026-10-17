"""File tools: arguments are checked here, the work is done by fs_worker
running as the user's OS account."""

from __future__ import annotations

import json

import config
import run_as
from run_as import UserContext
from tools.registry import Tool, ToolError

NO_TASK = "нет активной задачи с рабочей копией — выберите её: /tasks <id>"


async def worker(ctx: UserContext, op: str, args: dict | None = None):
    """Run one fs_worker operation on the user's working copy."""
    if ctx.task_id is None and op != "reset_workspace":
        raise ToolError(NO_TASK)
    request = {"op": op, "root": str(ctx.workspace), "args": args or {}}
    result = await run_as.run_as(
        ctx,
        [config.SYSTEM_PYTHON, "-I", str(config.FS_WORKER_PATH)],
        ctx.home,
        config.TOOL_TIMEOUT_SEC,
        input=json.dumps(request).encode(),
    )
    try:
        reply = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ToolError(f"файловая операция не удалась: {result.stderr.strip()[-500:]}") from None
    if not reply["ok"]:
        raise ToolError(reply["error"])
    return reply["result"]


def _tool(name: str, description: str, properties: dict, required: list[str], limits: dict | None = None) -> Tool:
    """A tool backed by the fs_worker operation of the same name. `limits` maps
    worker arguments to the config settings the worker cannot read itself."""
    async def call(ctx: UserContext, arguments: dict) -> str:
        settings = {arg: getattr(config, setting) for arg, setting in (limits or {}).items()}
        return await worker(ctx, name, {**arguments, **settings})

    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    return Tool(name, description, schema, call)


PATH = {"type": "string", "minLength": 1, "description": "Путь относительно корня рабочей копии"}

TOOLS = [
    _tool(
        "list_files",
        "Файлы проекта с размерами (учитывает .gitignore, без .git/).",
        {
            "path": {**PATH, "default": "."},
            "glob": {"type": "string", "default": "**/*", "description": "Шаблон: `**/*.py`, `*.md`"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 200},
        },
        [],
    ),
    _tool(
        "read_file",
        "Содержимое текстового файла с номерами строк; можно запросить диапазон строк.",
        {
            "path": PATH,
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "end_line": {"type": ["integer", "null"], "minimum": 1, "default": None},
        },
        ["path"],
        {"max_chars": "FILE_READ_MAX_CHARS"},
    ),
    _tool(
        "search_files",
        "Поиск строки или регулярного выражения по текстовым файлам проекта (с учётом .gitignore). "
        "Результат — `path:line: строка`, сгруппированные по файлам.",
        {
            "query": {"type": "string", "minLength": 1},
            "regex": {"type": "boolean", "default": False},
            "glob": {"type": "string", "default": "**/*"},
            "case_sensitive": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
        },
        ["query"],
    ),
    _tool(
        "python_outline",
        "Структура Python-файла: импорты, классы, методы и функции с номерами строк, сигнатурами "
        "и первой строкой docstring.",
        {"path": PATH},
        ["path"],
    ),
    _tool(
        "write_file",
        "Создать файл или полностью перезаписать его (недостающие каталоги создаются).",
        {"path": PATH, "content": {"type": "string"}},
        ["path", "content"],
        {"max_bytes": "FILE_WRITE_MAX_BYTES"},
    ),
    _tool(
        "edit_file",
        "Точечная правка: заменить `old_str` на `new_str`. `old_str` должен встречаться в файле ровно один раз.",
        {"path": PATH, "old_str": {"type": "string", "minLength": 1}, "new_str": {"type": "string"}},
        ["path", "old_str", "new_str"],
    ),
]
