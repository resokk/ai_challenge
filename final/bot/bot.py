"""Telegram handlers and the entry point."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import html
import json
import logging
import os
import re
import shlex
import sys
from collections import defaultdict
from io import BytesIO

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

import config
import context as chat_context
import git_ops
import llm
import os_users
from git_ops import GitError
from run_as import RunResult
from storage import Storage
from tools import builtin, files, runner
from tools.mcp_client import McpServers
from tools.registry import Registry, ToolError, ToolReply, clip_middle

logger = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096
CHUNK_CHARS = 3800
TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")

COMMANDS = [
    ("tasks", "Задачи: список, переключение, создание"),
    ("memory", "Память активной задачи"),
    ("git", "Статус, commit, push, diff, reset, ssh"),
    ("run", "Запустить скрипт или модуль: /run main.py, /run -m pytest"),
    ("pip", "Установить пакеты или показать установленные"),
    ("tools", "Каталог инструментов"),
    ("tool", "Вызвать инструмент: /tool <name> {json}"),
    ("agents", "Показать AGENTS.md (прислать — файлом .md)"),
    ("agents_clear", "Удалить AGENTS.md"),
    ("context", "Заполнение контекста"),
    ("summary", "Краткое изложение диалога"),
    ("reset", "Очистить историю чата"),
    ("help", "Справка; /help <вопрос> — ответ по справке"),
]

HELP = "Команды:\n" + "\n".join(f"/{name} — {text}" for name, text in COMMANDS) + (
    "\n\nОбычные сообщения уходят модели. AGENTS.md проекта можно прислать файлом."
)
# The tool of the `help` server in mcp.json.
HELP_TOOL = "help__help"
GIT_HELP = (
    "/git — статус рабочей копии\n"
    "/git commit [сообщение] — закоммитить всё (без сообщения его напишет модель)\n"
    "/git push — отправить в origin\n"
    "/git diff — разбор незакоммиченных изменений\n"
    "/git reset — откатить незакоммиченные изменения (с подтверждением)\n"
    "/git ssh [new] — показать или перегенерировать SSH-ключ"
)
TASKS_HELP = "/tasks — список задач\n/tasks <id> — переключиться\n/tasks <id> <owner/name | https://github.com/owner/name> — создать или обновить"
NO_TASK = "Нет активной задачи — выберите её: /tasks <id>"

COMMIT_PROMPT = (
    "Напиши сообщение коммита по diff. Первая строка — не длиннее 72 символов, в повелительном "
    "наклонении (например, «Add retry to client»), на английском, если по diff не видно, что в "
    "проекте принят другой язык. Если нужно — пустая строка и короткое тело. Только текст сообщения."
)
REVIEW_PROMPT = (
    "Ты делаешь ревью незакоммиченных изменений. По diff кратко опиши: что изменено по файлам, "
    "смысл изменений, замеченные риски (баги, забытые debug-вызовы и print, секреты в коде). "
    "Отвечай на русском."
)

# A run's stream longer than this is summarized by the model instead of shown.
RUN_SUMMARIZE_BYTES = 1024
# How much of a stream the summary request gets: its start and its end.
RUN_SUMMARY_INPUT_CHARS = 20000
RUN_OUTPUT_PROMPT = (
    "Тебе дан вывод (stdout или stderr) запущенной команды. Кратко изложи его на русском: что "
    "программа сделала и чем закончилась; для ошибок и traceback — тип исключения, сообщение и место "
    "(файл:строка); для тестов — сколько прошло и упало и какие именно упали. Ключевые строки "
    "приводи дословно. Не выдумывай того, чего нет в выводе."
)

_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


# sending


def split_message(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Parts no longer than `limit`, cut at line boundaries. A part that ends
    inside a code block closes it, and the next one reopens it."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    fence: str | None = None
    for line in text.split("\n"):
        pieces = [line[i:i + limit - 20] for i in range(0, len(line), limit - 20)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > limit - 4:
                parts.append("\n".join(current + (["```"] if fence else [])))
                current = [fence] if fence else []
                size = sum(len(item) + 1 for item in current)
            current.append(piece)
            size += len(piece) + 1
        if line.strip().startswith("```"):
            fence = None if fence else line.strip()
    if current and any(item.strip() for item in current):
        parts.append("\n".join(current))
    return parts or [""]


def _inline(line: str) -> str:
    line = html.escape(line, quote=False)
    line = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", line)
    return re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", line)


def to_html(text: str) -> str:
    """Telegram HTML for the Markdown the model writes: code blocks, inline
    code and bold; everything else escaped."""
    out: list[str] = []
    code: list[str] | None = None
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if code is None:
                code = []
            else:
                out.append(f"<pre>{html.escape(chr(10).join(code), quote=False)}</pre>")
                code = None
            continue
        if code is not None:
            code.append(line)
        else:
            out.append(_inline(line))
    if code is not None:
        out.append(f"<pre>{html.escape(chr(10).join(code), quote=False)}</pre>")
    return "\n".join(out)


async def send_text(message, text: str) -> None:
    for part in split_message(text or "(пусто)"):
        formatted = to_html(part)
        try:
            if len(formatted) > TELEGRAM_LIMIT:
                raise BadRequest("too long after formatting")
            await message.reply_text(formatted, parse_mode=ParseMode.HTML)
        except BadRequest:
            await message.reply_text(part)


async def send_code(message, text: str) -> None:
    await message.reply_text(f"<pre>{html.escape(text, quote=False)}</pre>", parse_mode=ParseMode.HTML)


async def _summarize_stream(title: str, name: str, stream: str) -> str | None:
    """The model's short account of a stream too long to read in chat; None
    when the model is unavailable."""
    try:
        return await llm.text_reply(
            [
                {"role": "system", "content": RUN_OUTPUT_PROMPT},
                {"role": "user", "content": f"Команда: {title}\nПоток: {name}\n\n"
                                            f"{clip_middle(stream, RUN_SUMMARY_INPUT_CHARS)}"},
            ],
            temperature=0.2,
        )
    except Exception:
        logger.exception("Summarizing %s of %s failed", name, title)
        return None


async def send_run(message, title: str, run: RunResult) -> None:
    """A run, as the user sees it: a header, then stdout and stderr each in
    its own message. A stream over RUN_SUMMARIZE_BYTES is shown as the model's
    summary of it (or cut to its start and end if the model is unavailable),
    and the full stream comes as a file whenever it is not shown whole."""
    status = "таймаут" if run.timed_out else f"код выхода {run.exit_code}"
    await message.reply_text(f"{title}\n{status}, {run.duration:.1f} с")
    for name, stream in (("stdout", run.stdout), ("stderr", run.stderr)):
        if not stream.strip():
            continue
        summary = None
        if len(stream.encode()) > RUN_SUMMARIZE_BYTES:
            async with typing(message.chat):
                summary = await _summarize_stream(title, name, stream)
        if summary:
            await send_text(message, f"{name} (кратко, полный вывод — в файле):\n{summary}")
        else:
            await send_code(message, f"{name}:\n{clip_middle(stream, config.RUN_OUTPUT_MAX_CHARS)}")
        if summary or len(stream) > config.RUN_OUTPUT_MAX_CHARS:
            await message.reply_document(InputFile(BytesIO(stream.encode()), filename=f"{name}.txt"))


@contextlib.asynccontextmanager
async def typing(chat):
    async def keep_typing():
        while True:
            with contextlib.suppress(Exception):
                await chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(4)

    task = asyncio.create_task(keep_typing())
    try:
        yield
    finally:
        task.cancel()


class Status:
    """A transient "working on it" line, replaced as work moves on and removed at the end."""

    def __init__(self, message):
        self.message = message
        self.sent = None

    async def show(self, text: str) -> None:
        with contextlib.suppress(Exception):
            if self.sent is None:
                self.sent = await self.message.reply_text(text)
            else:
                await self.sent.edit_text(text)

    async def clear(self) -> None:
        if self.sent is not None:
            with contextlib.suppress(Exception):
                await self.sent.delete()
            self.sent = None


# plumbing


def command_text(update: Update) -> str:
    """Everything after the command, line breaks included."""
    match = re.match(r"/\S+\s?(.*)", update.message.text or "", re.DOTALL)
    return match[1].strip() if match else ""


def user_handler(func):
    """Handlers run one at a time per user: a user's messages, tool runs, git
    commands and task switches never overlap. Different users run in parallel."""

    @functools.wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        async with _locks[update.effective_user.id]:
            return await func(update, context)

    return wrapped


def _store(context) -> Storage:
    return context.bot_data["store"]


def _registry(context) -> Registry:
    return context.bot_data["registry"]


def user_ctx(store: Storage, telegram_id: int):
    return os_users.user_context(store.get_user(telegram_id), store.current_task(telegram_id))


async def gate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The one access check: anything but /start from a user who is not
    approved stops here - no model call, nothing stored."""
    user = update.effective_user
    if user is None:
        raise ApplicationHandlerStop
    message = update.effective_message
    command = (message.text or "").split(maxsplit=1)[0] if message is not None and message.text else ""
    if command == "/start" or command.startswith("/start@"):
        return
    row = _store(context).get_user(user.id)
    if row is not None and row["status"] == "approved":
        return
    text = "Доступ заблокирован." if row is not None and row["status"] == "blocked" else (
        "Доступа нет: отправьте /start и дождитесь одобрения администратора."
    )
    if update.callback_query is not None:
        await update.callback_query.answer(text, show_alert=True)
    elif message is not None:
        await message.reply_text(text)
    raise ApplicationHandlerStop


# commands


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    row, _ = _store(context).register(user.id, user.username, user.first_name)
    if row["status"] == "approved":
        await update.message.reply_text("Доступ открыт.\n\n" + HELP)
    elif row["status"] == "blocked":
        await update.message.reply_text("Доступ заблокирован администратором.")
    else:
        await update.message.reply_text("Заявка принята и ждёт одобрения администратора.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/help <вопрос>` goes to the help MCP server's tool when it is up."""
    question = command_text(update)
    if not question:
        await update.message.reply_text(HELP)
    elif HELP_TOOL in _registry(context).tools:
        await call_for_user(update, context, HELP_TOOL, {"question": question})
    else:
        await update.message.reply_text("Справочный сервер недоступен, вопросы пока не принимаются.\n\n" + HELP)


@user_handler
async def agents_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if not (document.file_name or "").lower().endswith(".md"):
        await update.message.reply_text("Пришлите AGENTS.md или другой файл .md.")
        return
    limit = config.AGENTS_MD_MAX_CHARS
    if document.file_size and document.file_size > limit * 4:
        await update.message.reply_text(f"Файл слишком большой: лимит {limit} символов.")
        return
    data = await (await document.get_file()).download_as_bytearray()
    try:
        text = bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError:
        await update.message.reply_text("Файл не в UTF-8.")
        return
    if len(text) > limit:
        await update.message.reply_text(f"Файл слишком большой: {len(text)} символов при лимите {limit}.")
        return
    _store(context).set_agents_md(update.effective_user.id, text)
    await update.message.reply_text(f"AGENTS.md сохранён ({len(text)} символов). Он будет в контексте каждого запроса.")


@user_handler
async def agents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = _store(context).get_user(update.effective_user.id)["agents_md"]
    if text:
        await send_text(update.message, text)
    else:
        await update.message.reply_text("AGENTS.md не задан. Пришлите файл .md, чтобы задать его.")


@user_handler
async def agents_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _store(context).set_agents_md(update.effective_user.id, None)
    await update.message.reply_text("AGENTS.md удалён.")


def _tasks_list(store: Storage, telegram_id: int) -> str:
    rows = store.list_tasks(telegram_id)
    if not rows:
        return "Задач нет. Создайте: /tasks <id> <owner/name>"
    current = store.get_user(telegram_id)["current_task_id"]
    lines = [
        f"{'▶' if row['task_id'] == current else '•'} {row['task_id']} — {row['repo_url']}"
        + (" [память]" if row["memory"] else "")
        for row in rows
    ]
    return "Задачи:\n" + "\n".join(lines)


@user_handler
async def tasks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    telegram_id = update.effective_user.id
    args = context.args or []
    if not args:
        await update.message.reply_text(_tasks_list(store, telegram_id))
        return
    if len(args) > 2:
        await update.message.reply_text(TASKS_HELP)
        return
    task_id = args[0]
    if not TASK_ID_PATTERN.fullmatch(task_id):
        await update.message.reply_text("id задачи: 1–64 символа из A-Z, a-z, 0-9, _, ., -")
        return
    if len(args) == 2:
        repo_url = git_ops.normalize_repo(args[1])
        if repo_url is None:
            await update.message.reply_text("Репозиторий: owner/name или https://github.com/owner/name")
            return
        store.upsert_task(telegram_id, task_id, repo_url)
    elif store.get_task(telegram_id, task_id) is None:
        ids = ", ".join(row["task_id"] for row in store.list_tasks(telegram_id)) or "нет"
        await update.message.reply_text(f"Задачи {task_id} нет. Есть: {ids}\n\n{TASKS_HELP}")
        return
    store.set_current_task(telegram_id, task_id)
    task = store.current_task(telegram_id)
    status = Status(update.message)
    await status.show(f"Клонирую {task['repo_url']}…")
    try:
        result = await git_ops.switch_workspace(user_ctx(store, telegram_id))
    except (ToolError, GitError) as error:
        result = git_ops.SwitchResult(False, str(error))
    finally:
        await status.clear()
    lines = [f"Активная задача: {task_id} — {task['repo_url']}"]
    if result.lost_changes:
        lines.append("⚠️ В прежней рабочей копии были незакоммиченные изменения — они потеряны.")
    if result.error:
        lines += [
            f"Клонировать не удалось:\n{result.error}",
            "Если репозиторий приватный — создайте ключ /git ssh, добавьте его в GitHub и повторите /tasks "
            + task_id,
        ]
    else:
        lines.append("Рабочая копия готова.")
    await update.message.reply_text("\n".join(lines))


@user_handler
async def memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    telegram_id = update.effective_user.id
    task = store.current_task(telegram_id)
    if task is None:
        await update.message.reply_text(NO_TASK)
        return
    text = command_text(update)
    if not text:
        if task["memory"]:
            await send_text(update.message, f"Память задачи {task['task_id']}:\n\n{task['memory']}")
        else:
            await update.message.reply_text(f"Память задачи {task['task_id']} пуста. Задать: /memory <текст>")
        return
    if len(text) > config.MEMORY_MAX_CHARS:
        await update.message.reply_text(
            f"Слишком длинно: {len(text)} символов при лимите {config.MEMORY_MAX_CHARS}. Память не изменена."
        )
        return
    store.set_memory(telegram_id, task["task_id"], text)
    await update.message.reply_text(f"Память задачи {task['task_id']} обновлена.")


@user_handler
async def git_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    telegram_id = update.effective_user.id
    args = context.args or []
    sub = args[0] if args else ""
    ctx = user_ctx(store, telegram_id)
    try:
        if sub == "ssh" and len(args) <= 2 and args[1:] in ([], ["new"]):
            await _git_ssh(update, ctx, regenerate=args[1:] == ["new"])
        elif ctx.task_id is None and sub in ("", "commit", "push", "diff", "reset"):
            await update.message.reply_text(NO_TASK)
        elif sub == "":
            await send_code(update.message, await git_ops.status(ctx))
        elif sub == "commit":
            message = command_text(update).removeprefix("commit").strip() or None
            await _git_commit(update, store, ctx, message)
        elif sub == "push" and len(args) == 1:
            async with typing(update.effective_chat):
                await send_code(update.message, await git_ops.push(ctx))
        elif sub == "diff" and len(args) == 1:
            await _git_diff(update, ctx)
        elif sub == "reset" and len(args) == 1:
            await _git_reset_prompt(update, ctx)
        else:
            await update.message.reply_text(GIT_HELP)
    except (GitError, ToolError) as error:
        await update.message.reply_text(str(error))


async def _git_ssh(update: Update, ctx, regenerate: bool) -> None:
    key, created = await git_ops.public_key(ctx, regenerate)
    lines = []
    if regenerate:
        lines.append("⚠️ Создан новый ключ: старый ключ в GitHub больше не работает — замените его.")
    elif created:
        lines.append("Создан новый SSH-ключ.")
    lines.append("Публичный ключ:")
    await update.message.reply_text("\n".join(lines))
    await send_code(update.message, key)
    await update.message.reply_text(
        "Добавьте его в GitHub: Settings → SSH and GPG keys → New SSH key, "
        "или в репозитории Settings → Deploy keys (отметьте Allow write access). "
        "Затем повторите /tasks <id>, чтобы клонировать по SSH."
    )


async def _git_commit(update: Update, store: Storage, ctx, message: str | None) -> None:
    user = store.get_user(ctx.telegram_id)
    author = user["first_name"] or user["username"] or str(ctx.telegram_id)

    async def make_message(diff: str) -> str:
        text = await llm.text_reply(
            [{"role": "system", "content": COMMIT_PROMPT}, {"role": "user", "content": diff}], temperature=0.2
        )
        return "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()

    async with typing(update.effective_chat):
        await send_code(update.message, await git_ops.commit(ctx, message, author, make_message))


async def _git_diff(update: Update, ctx) -> None:
    diff = await git_ops.diff_for_review(ctx)
    if diff is None:
        await update.message.reply_text("Изменений нет.")
        return
    async with typing(update.effective_chat):
        try:
            review = await llm.text_reply(
                [{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": diff}]
            )
        except Exception:
            logger.exception("Diff review failed")
            await update.message.reply_text("Модель недоступна, попробуйте позже.")
            return
    await send_text(update.message, review)


async def _git_reset_prompt(update: Update, ctx) -> None:
    changes = await git_ops.pending_changes(ctx)
    if not changes:
        await update.message.reply_text("Нечего откатывать: рабочая копия чистая.")
        return
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Откатить", callback_data=f"gitreset:{ctx.telegram_id}:{ctx.task_id}"),
        InlineKeyboardButton("Отмена", callback_data=f"gitcancel:{ctx.telegram_id}:{ctx.task_id}"),
    ]])
    listing = "\n".join(changes[:50]) + (f"\n… и ещё {len(changes) - 50}" if len(changes) > 50 else "")
    await update.message.reply_text(
        f"Будут потеряны незакоммиченные изменения задачи {ctx.task_id}:\n{listing}\n\nОткатить?",
        reply_markup=keyboard,
    )


@user_handler
async def git_reset_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    action, owner, task_id = query.data.split(":", 2)
    store = _store(context)
    telegram_id = update.effective_user.id
    if str(telegram_id) != owner or store.get_user(telegram_id)["current_task_id"] != task_id:
        await query.answer("Кнопка устарела или не ваша.", show_alert=True)
        return
    await query.answer()
    if action == "gitcancel":
        await query.edit_message_text("Откат отменён.")
        return
    try:
        await git_ops.reset(user_ctx(store, telegram_id))
    except (GitError, ToolError) as error:
        await query.edit_message_text(f"Откат не удался: {error}")
        return
    await query.edit_message_text(f"Незакоммиченные изменения задачи {task_id} откачены.")


async def call_for_user(update: Update, context, name: str, arguments: dict, title: str | None = None) -> ToolReply:
    """A tool call whose result goes to the user only: no model, no history."""
    store = _store(context)
    ctx = user_ctx(store, update.effective_user.id)
    status = Status(update.message)
    await status.show(f"выполняется {name}…")
    try:
        reply = await _registry(context).call_tool(ctx, name, arguments)
    finally:
        await status.clear()
    if reply.run is not None:
        await send_run(update.message, title or f"{name} {json.dumps(arguments, ensure_ascii=False)}", reply.run)
    else:
        await send_text(update.message, reply.text)
    return reply


@user_handler
async def tool_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = command_text(update)
    if not text:
        await update.message.reply_text(
            "Использование: /tool <name> [json-аргументы]\nНапример: /tool read_file {\"path\": \"main.py\"}\n"
            "Список инструментов: /tools"
        )
        return
    name, _, raw = text.partition(" ")
    try:
        arguments = json.loads(raw.strip() or "{}")
    except json.JSONDecodeError:
        arguments = None
    if not isinstance(arguments, dict):
        await update.message.reply_text('Аргументы — JSON-объект, например: /tool pypi_package_info {"name": "httpx"}')
        return
    await call_for_user(update, context, name.strip(), arguments)


async def tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_text(update.message, _registry(context).catalog())


def _shell_args(update: Update) -> list[str] | None:
    try:
        return shlex.split(command_text(update))
    except ValueError:
        return None


@user_handler
async def pip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = _shell_args(update)
    if args is None:
        await update.message.reply_text("Не удалось разобрать аргументы (незакрытая кавычка?).")
    elif not args:
        await call_for_user(update, context, "pip_list", {}, "pip list")
    elif args[0] == "-r":
        if len(args) != 2:
            await update.message.reply_text("Использование: /pip -r <путь к requirements.txt>")
            return
        await call_for_user(update, context, "pip_install", {"requirements": args[1]}, f"pip install -r {args[1]}")
    else:
        await call_for_user(update, context, "pip_install", {"packages": args}, "pip install " + " ".join(args))


@user_handler
async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = _shell_args(update)
    if args is None:
        await update.message.reply_text("Не удалось разобрать аргументы (незакрытая кавычка?).")
    elif not args or (args[0] == "-m" and len(args) < 2):
        await update.message.reply_text("Использование: /run <file.py> [args…] или /run -m <module> [args…]")
    elif args[0] == "-m":
        await call_for_user(update, context, "run_python", {"target": args[1], "module": True, "args": args[2:]},
                            "python " + shlex.join(args))
    else:
        await call_for_user(update, context, "run_python", {"target": args[0], "args": args[1:]},
                            "python " + shlex.join(args))


@user_handler
async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    store.clear_history(update.effective_user.id)
    store.set_last_tokens(update.effective_user.id, 0)
    await update.message.reply_text("История чата очищена. AGENTS.md, задачи и их память сохранены.")


@user_handler
async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    telegram_id = update.effective_user.id
    user = store.get_user(telegram_id)
    task = store.current_task(telegram_id)
    history = store.history(telegram_id)
    tokens = user["last_tokens"]
    limit = config.CONTEXT_LIMIT_TOKENS
    lines = [
        f"Задача: {task['task_id']} — {task['repo_url']}" if task else "Задача: не выбрана",
        f"Память задачи: {'есть' if task and task['memory'] else 'нет'}",
        f"Токенов в последнем запросе: {tokens} из {limit} ({tokens / limit:.0%})",
        f"Сообщений в истории: {sum(row['role'] != 'summary' for row in history)}",
        f"Сводка: {'есть' if any(row['role'] == 'summary' for row in history) else 'нет'}",
    ]
    await update.message.reply_text("\n".join(lines))


@user_handler
async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    request = chat_context.dialog_summary_request(_store(context), update.effective_user.id)
    if request is None:
        await update.message.reply_text("История пуста — излагать нечего.")
        return
    async with typing(update.effective_chat):
        try:
            text = await llm.text_reply(request)
        except Exception:
            logger.exception("Summary request failed")
            await update.message.reply_text("Модель недоступна, попробуйте позже.")
            return
    await send_text(update.message, text)


@user_handler
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    registry = _registry(context)
    telegram_id = update.effective_user.id
    async with typing(update.effective_chat):
        if chat_context.needs_summary(store.get_user(telegram_id)):
            await chat_context.summarize(store, telegram_id)
        store.add_message(telegram_id, "user", update.message.text)
        ctx = user_ctx(store, telegram_id)
        status = Status(update.message)

        async def call_tool(name: str, arguments: dict) -> str:
            return (await registry.call_tool(ctx, name, arguments)).text

        def save(message: dict) -> None:
            store.add_message(telegram_id, message["role"], message.get("content"),
                              message.get("tool_calls"), message.get("tool_call_id"))

        try:
            answer, tokens = await llm.run_turn(
                chat_context.build(store, telegram_id, registry.catalog()),
                call_tool,
                save,
                lambda name: status.show(f"вызываю {name}…"),
            )
        except Exception:
            logger.exception("LLM turn failed for %s", telegram_id)
            await update.message.reply_text("Не удалось получить ответ модели. Попробуйте ещё раз.")
            return
        finally:
            await status.clear()
    store.set_last_tokens(telegram_id, tokens)
    await send_text(update.message, answer)


# startup


def build_registry() -> Registry:
    registry = Registry()
    for tool in (*builtin.TOOLS, *files.TOOLS, *runner.TOOLS):
        registry.register(tool)
    return registry


async def ensure_accounts(store: Storage) -> None:
    for user in store.list_users("approved"):
        try:
            if await os_users.ensure_os_user(user):
                logger.warning("OS account for user %s was missing and has been created", user["telegram_id"])
        except Exception:
            logger.exception("OS account for user %s is not usable", user["telegram_id"])


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error", exc_info=context.error)
    message = getattr(update, "effective_message", None)
    if message is not None:
        with contextlib.suppress(Exception):
            await message.reply_text("Внутренняя ошибка, подробности в логе бота.")


async def post_init(application: Application) -> None:
    await ensure_accounts(application.bot_data["store"])
    await application.bot_data["mcp"].start(config.MCP_CONFIG_PATH, application.bot_data["registry"])
    await application.bot.set_my_commands([BotCommand(name, text) for name, text in COMMANDS])


async def post_shutdown(application: Application) -> None:
    await application.bot_data["mcp"].stop()
    application.bot_data["store"].close()


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os_users.require_root()
    # Whatever the bot creates (bot.db and its journal) is root's alone.
    os.umask(0o077)
    missing = [name for name in ("TELEGRAM_BOT_TOKEN", "DEEPSEEK_API_KEY") if not getattr(config, name)]
    if missing:
        print(f"Ошибка: в .env не заданы {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    store = Storage(config.DB_PATH)
    os_users.secure_bot_files()

    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data.update(store=store, registry=build_registry(), mcp=McpServers())

    application.add_handler(TypeHandler(Update, gate), group=-1)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tasks", tasks))
    application.add_handler(CommandHandler("memory", memory))
    application.add_handler(CommandHandler("git", git_command))
    application.add_handler(CommandHandler("tool", tool_command))
    application.add_handler(CommandHandler("tools", tools_command))
    application.add_handler(CommandHandler("pip", pip_command))
    application.add_handler(CommandHandler("run", run_command))
    application.add_handler(CommandHandler("agents", agents))
    application.add_handler(CommandHandler("agents_clear", agents_clear))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("context", context_command))
    application.add_handler(CommandHandler("summary", summary_command))
    application.add_handler(CallbackQueryHandler(git_reset_button, pattern=r"^git(reset|cancel):"))
    application.add_handler(MessageHandler(filters.Document.ALL, agents_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    application.add_error_handler(on_error)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
