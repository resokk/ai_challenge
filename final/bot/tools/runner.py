"""The user's venv, installing packages into it and running their code."""

from __future__ import annotations

import logging
import re

import config
import run_as
from run_as import RunResult, UserContext
from tools.files import NO_TASK, worker
from tools.registry import Tool, ToolError

# A requirement specifier: a name, optional extras, optional version clauses.
# Anything else - options, paths, URLs, VCS links - is refused.
_VERSION = r"(?:===|==|!=|~=|>=|<=|>|<)\s*[A-Za-z0-9.*+!_-]+"
SPEC_PATTERN = re.compile(
    rf"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    rf"(?:\s*{_VERSION}(?:\s*,\s*{_VERSION})*)?"
)
MODULE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")

logger = logging.getLogger(__name__)


def _env(ctx: UserContext) -> dict[str, str]:
    return {
        "PATH": f"{ctx.venv / 'bin'}:{run_as.DEFAULT_PATH}",
        "VIRTUAL_ENV": str(ctx.venv),
        "PYTHONUNBUFFERED": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }


async def ensure_venv(ctx: UserContext) -> None:
    """Create `<home>/.venv` the first time it is needed, with a current pip."""
    python = ctx.venv / "bin" / "python"
    check = await run_as.run_as(ctx, ["test", "-x", str(python)], ctx.home, 10)
    if check.exit_code == 0:
        return
    result = await run_as.run_as(ctx, [config.SYSTEM_PYTHON, "-m", "venv", str(ctx.venv)], ctx.home,
                                 config.PIP_TIMEOUT_SEC, env=_env(ctx))
    if result.exit_code != 0:
        raise ToolError(f"не удалось создать venv: {result.stderr.strip()[-1000:]}")
    # The bundled pip still works when this cannot reach the index.
    result = await run_as.run_as(ctx, [str(python), "-m", "pip", "install", "--upgrade", "pip"], ctx.home,
                                 config.PIP_TIMEOUT_SEC, env=_env(ctx))
    if result.exit_code != 0:
        logger.warning("pip upgrade in %s failed: %s", ctx.venv, result.stderr.strip()[-500:])


def check_spec(spec: str) -> None:
    if not SPEC_PATTERN.fullmatch(spec.strip()):
        raise ToolError(
            f"«{spec}» — не спецификация пакета. Допустимо: name, name[extra], name==1.2, name>=1,<2"
        )


async def pip_install(ctx: UserContext, arguments: dict) -> RunResult:
    packages = arguments.get("packages") or []
    requirements = arguments.get("requirements")
    if not packages and not requirements:
        raise ToolError("укажите packages или requirements")
    for spec in packages:
        check_spec(spec)
    argv = ["-m", "pip", "install", *(spec.strip() for spec in packages)]
    cwd = ctx.home
    if requirements:
        if ctx.task_id is None:
            raise ToolError(NO_TASK)
        argv += ["-r", await worker(ctx, "resolve", {"path": requirements})]
        cwd = ctx.workspace
    await ensure_venv(ctx)
    return await run_as.run_as(ctx, [str(ctx.venv / "bin" / "python"), *argv], cwd, config.PIP_TIMEOUT_SEC,
                               env=_env(ctx))


async def pip_list(ctx: UserContext, arguments: dict) -> RunResult:
    await ensure_venv(ctx)
    return await run_as.run_as(ctx, [str(ctx.venv / "bin" / "python"), "-m", "pip", "list", "--format=freeze"],
                               ctx.home, config.TOOL_TIMEOUT_SEC, env=_env(ctx))


async def run_python(ctx: UserContext, arguments: dict) -> RunResult:
    target = arguments["target"]
    if ctx.task_id is None:
        raise ToolError(NO_TASK)
    if arguments.get("module"):
        if not MODULE_PATTERN.fullmatch(target):
            raise ToolError(f"«{target}» — не имя модуля")
        head = ["-m", target]
    else:
        head = [await worker(ctx, "resolve", {"path": target})]
    await ensure_venv(ctx)
    return await run_as.run_as(
        ctx,
        [str(ctx.venv / "bin" / "python"), *head, *arguments.get("args", [])],
        ctx.workspace,
        config.RUN_TIMEOUT_SEC,
        env=_env(ctx),
    )


# Room for creating the venv before the work itself.
_SLACK = config.PIP_TIMEOUT_SEC * 2 + 30

TOOLS = [
    Tool(
        "pip_install",
        "Установить пакеты в venv пользователя. `packages` — спецификации (`name`, `name[extra]`, "
        "`name==1.2`, `name>=1,<2`); `requirements` — путь к файлу требований в рабочей копии.",
        {
            "type": "object",
            "properties": {
                "packages": {"type": "array", "items": {"type": "string", "minLength": 1}, "default": []},
                "requirements": {"type": ["string", "null"], "default": None},
            },
            "additionalProperties": False,
        },
        pip_install,
        timeout=config.PIP_TIMEOUT_SEC + _SLACK,
    ),
    Tool(
        "pip_list",
        "Пакеты, установленные в venv пользователя.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        pip_list,
        timeout=config.TOOL_TIMEOUT_SEC + _SLACK,
    ),
    Tool(
        "run_python",
        "Запустить Python-скрипт из рабочей копии (`target` — путь) или модуль (`module=true`, "
        "`target` — имя модуля, например `pytest`, `ruff`, `mypy`) в venv пользователя. "
        "Возвращает exit_code, duration, stdout и stderr.",
        {
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "args": {"type": "array", "items": {"type": "string"}, "default": []},
                "module": {"type": "boolean", "default": False},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
        run_python,
        timeout=config.RUN_TIMEOUT_SEC + _SLACK,
    ),
]
