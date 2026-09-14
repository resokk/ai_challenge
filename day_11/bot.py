from __future__ import annotations

import logging
import re

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN
from deepseek_client import DeepSeekClient
from storage import UserStore

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
MAX_HISTORY_MESSAGES = 100
MAX_PROMPT_LENGTH = 2000
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

BOT_COMMANDS = (
    BotCommand("profile", "Show or update your profile"),
    BotCommand("tasks", "List your tasks, or switch to one"),
    BotCommand("task", "Show or update the current task's prompt, or 'none' to leave it"),
    BotCommand("clear", "Forget this conversation"),
)


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(BOT_COMMANDS)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data.clear()
    await update.message.reply_text(
        f"Hi {update.effective_user.first_name}! I'm alive. Ask me anything."
    )


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Without arguments, show this user's profile; with text, replace it. The
    profile becomes the system prompt for everything the user says next."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    new_profile = _parse_arg(update)

    if not new_profile:
        current = store.load_profile(user_id)
        text = (
            f"Your profile:\n\n{current}"
            if current
            else "You have no profile yet. Use /profile <text> to set one - I'll use it as my instructions for you."
        )
        for chunk in _split_message(text):
            await update.message.reply_text(chunk)
        return

    if len(new_profile) > MAX_PROMPT_LENGTH:
        await update.message.reply_text(
            f"That profile is too long ({len(new_profile)} chars). Keep it under {MAX_PROMPT_LENGTH} chars."
        )
        return

    store.save_profile(user_id, new_profile)
    await update.message.reply_text("Profile updated.")


async def tasks_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Without arguments, list this user's tasks; with a task id, switch to it,
    starting it if it is new. Each task is a separate conversation."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = _parse_arg(update)
    current = store.current_task(user_id)

    if not task_id:
        lines = [f"Current task: {current}" if current else "You are not on a task."]
        tasks = store.list_tasks(user_id)
        if tasks:
            lines += ["", "Your tasks:"] + [f"{'-> ' if t == current else '   '}{t}" for t in tasks]
        lines += ["", "Use /tasks <id> to switch to a task, and /task <text> to set its prompt."]
        await update.message.reply_text("\n".join(lines))
        return

    if not TASK_ID_PATTERN.match(task_id):
        await update.message.reply_text(
            "A task id must be 1-32 characters: letters, digits, dashes or underscores."
        )
        return

    store.switch_task(user_id, task_id)
    suffix = "" if store.load_task_prompt(user_id, task_id) else " It has no prompt yet - set one with /task <text>."
    await update.message.reply_text(f"Switched to task {task_id}.{suffix}")


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Without arguments, show the current task's prompt; with text, replace it;
    with 'none', leave the task. The prompt is added to the user's profile for
    every message on that task."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = store.current_task(user_id)

    if not task_id:
        await update.message.reply_text("You are not on a task. Use /tasks <id> to start or switch to one.")
        return

    new_prompt = _parse_arg(update)

    if new_prompt.lower() == "none":
        store.clear_task(user_id)
        await update.message.reply_text(
            f"Left task {task_id}. Its prompt is kept - use /tasks {task_id} to come back."
        )
        return

    if not new_prompt:
        current = store.load_task_prompt(user_id, task_id)
        text = (
            f"Prompt for task {task_id}:\n\n{current}"
            if current
            else f"Task {task_id} has no prompt yet. Use /task <text> to set one."
        )
        for chunk in _split_message(text):
            await update.message.reply_text(chunk)
        return

    if len(new_prompt) > MAX_PROMPT_LENGTH:
        await update.message.reply_text(
            f"That prompt is too long ({len(new_prompt)} chars). Keep it under {MAX_PROMPT_LENGTH} chars."
        )
        return

    store.save_task_prompt(user_id, task_id, new_prompt)
    await update.message.reply_text(f"Prompt for task {task_id} updated.")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forget the messages of the thread the user is on. Profiles and task
    prompts live in the database, not in the history, so they are untouched."""
    task_id = context.bot_data["store"].current_task(update.effective_user.id)
    context.chat_data.get("histories", {}).pop(task_id, None)
    where = f"for task {task_id}" if task_id else "off task"
    await update.message.reply_text(
        f"Conversation {where} cleared. Your profile and task prompts are unchanged."
    )


def _parse_arg(update: Update) -> str:
    parts = update.message.text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _system_prompt(store: UserStore, user_id: int, task_id: str | None) -> str | None:
    """Who the user is, then what they are working on. Either half may be
    missing, and with neither the client falls back to its own default."""
    task_prompt = store.load_task_prompt(user_id, task_id) if task_id else None
    parts = [part for part in (store.load_profile(user_id), task_prompt) if part]
    return "\n\n".join(parts) or None


def _split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list:
    """Split text into <= limit-sized chunks, preferring to break on blank
    lines, then newlines, then spaces, so a long reply is not cut mid-word."""
    chunks = []
    while len(text) > limit:
        split_at = -1
        for boundary in ("\n\n", "\n", " "):
            idx = text.rfind(boundary, 0, limit)
            if idx > 0:
                split_at = idx + len(boundary)
                break
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:]
    if text:
        chunks.append(text)
    return chunks


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = store.current_task(user_id)

    # One conversation per task, so switching tasks switches the thread and
    # switching back returns to it. No task is itself a thread.
    history = context.chat_data.setdefault("histories", {}).setdefault(task_id, [])
    history.append({"role": "user", "content": update.message.text})
    del history[:-MAX_HISTORY_MESSAGES]

    await update.message.chat.send_action("typing")

    system_prompt = _system_prompt(store, user_id, task_id)

    try:
        reply = await DeepSeekClient.get_reply(history, system_prompt)
    except Exception:
        logger.exception("DeepSeek API call failed")
        history.pop()
        await update.message.reply_text("Sorry, I couldn't reach DeepSeek right now. Try again in a bit.")
        return

    if not reply:
        logger.warning("DeepSeek returned an empty reply")
        history.pop()
        await update.message.reply_text("Sorry, I didn't get a usable reply from DeepSeek. Try rephrasing.")
        return

    history.append({"role": "assistant", "content": reply})

    for chunk in _split_message(reply):
        await update.message.reply_text(chunk)


def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()

    # Read on every request, so a profile edit takes effect on the next message.
    application.bot_data["store"] = UserStore()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("tasks", tasks_command))
    application.add_handler(CommandHandler("task", task_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
