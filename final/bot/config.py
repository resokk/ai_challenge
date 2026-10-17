"""Settings from `.env`, with every path resolved against the bot's own
directory rather than wherever the bot happens to be started from."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BOT_DIR = Path(__file__).resolve().parent
ENV_PATH = BOT_DIR / ".env"
load_dotenv(ENV_PATH)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default))
    return path if path.is_absolute() else BOT_DIR / path


# Read lazily by the code that needs them, so tests and `admin.py list` work
# without secrets in the environment.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

DB_PATH = _path("DB_PATH", "bot.db")
MCP_CONFIG_PATH = _path("MCP_CONFIG_PATH", "mcp.json")
PROMPTS_DIR = BOT_DIR / "prompts"
FS_WORKER_PATH = BOT_DIR / "tools" / "fs_worker.py"
# Interpreter for fs_worker and for creating venvs. It must be readable by the
# per-user OS accounts, so it is the system one, not the bot's own venv.
SYSTEM_PYTHON = os.environ.get("SYSTEM_PYTHON", "/usr/bin/python3")

CONTEXT_LIMIT_TOKENS = _int("CONTEXT_LIMIT_TOKENS", 64000)
SUMMARIZE_AT = 0.6
AGENTS_MD_MAX_CHARS = _int("AGENTS_MD_MAX_CHARS", 20000)
MEMORY_MAX_CHARS = _int("MEMORY_MAX_CHARS", 4000)

TOOL_TIMEOUT_SEC = _int("TOOL_TIMEOUT_SEC", 30)
TOOL_RESULT_MAX_CHARS = _int("TOOL_RESULT_MAX_CHARS", 20000)
MAX_TOOL_ITERATIONS = _int("MAX_TOOL_ITERATIONS", 8)

FILE_READ_MAX_CHARS = _int("FILE_READ_MAX_CHARS", 20000)
FILE_WRITE_MAX_BYTES = _int("FILE_WRITE_MAX_BYTES", 1_000_000)

GIT_TIMEOUT_SEC = _int("GIT_TIMEOUT_SEC", 60)
GIT_DIFF_MAX_CHARS = _int("GIT_DIFF_MAX_CHARS", 20000)

PIP_TIMEOUT_SEC = _int("PIP_TIMEOUT_SEC", 300)
RUN_TIMEOUT_SEC = _int("RUN_TIMEOUT_SEC", 60)
RUN_OUTPUT_MAX_CHARS = _int("RUN_OUTPUT_MAX_CHARS", 3000)

# Resource limits for every process started as a user's OS account.
RUN_MEMORY_MB = _int("RUN_MEMORY_MB", 1024)
RUN_CPU_SEC = _int("RUN_CPU_SEC", 600)
RUN_FILE_MB = _int("RUN_FILE_MB", 512)
RUN_NPROC = _int("RUN_NPROC", 128)
