from __future__ import annotations

import logging
import re

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN
from deepseek_client import DeepSeekClient
from storage import PROFILES, TASKS, Collection, UserStore

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
MAX_HISTORY_MESSAGES = 100
MAX_TEXT_LENGTH = 2000
ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

BOT_COMMANDS = (
    BotCommand("profiles", "List your profiles, or switch to one"),
    BotCommand("profile", "Show or update the current profile's description, or 'none' to leave it"),
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


def _list_command(collection: Collection):
    """/profiles and /tasks: without an argument, list what the user has and
    mark the one they are on; with an id, switch to it, creating it if it is
    new. Selecting is how a profile or a task comes into existence."""

    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        store = context.bot_data["store"]
        user_id = update.effective_user.id
        item_id = _parse_arg(update)
        current = store.current(collection, user_id)
        noun = collection.noun

        if not item_id:
            lines = [f"Current {noun}: {current}" if current else f"No {noun} selected."]
            item_ids = store.list_ids(collection, user_id)
            if item_ids:
                lines += ["", f"Your {noun}s:"]
                lines += [f"{'-> ' if i == current else '   '}{i}" for i in item_ids]
            lines += [
                "",
                f"Use /{noun}s <id> to switch to a {noun} or start a new one, "
                f"and /{noun} <text> to set its {collection.text_noun}.",
            ]
            await update.message.reply_text("\n".join(lines))
            return

        if not ITEM_ID_PATTERN.match(item_id):
            await update.message.reply_text(
                f"A {noun} id must be 1-32 characters: letters, digits, dashes or underscores."
            )
            return

        store.select(collection, user_id, item_id)
        suffix = (
            ""
            if store.load_text(collection, user_id, item_id)
            else f" It has no {collection.text_noun} yet - set one with /{noun} <text>."
        )
        await update.message.reply_text(f"Switched to {noun} {item_id}.{suffix}")

    return handler


def _text_command(collection: Collection):
    """/profile and /task: without an argument, show the current item's text;
    with text, replace it; with 'none', leave the item. Both texts feed the
    system prompt, so an edit applies to the user's very next message."""

    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        store = context.bot_data["store"]
        user_id = update.effective_user.id
        item_id = store.current(collection, user_id)
        noun, text_noun = collection.noun, collection.text_noun

        if not item_id:
            await update.message.reply_text(
                f"No {noun} selected. Use /{noun}s <id> to start or switch to one."
            )
            return

        new_text = _parse_arg(update)

        if new_text.lower() == "none":
            store.deselect(collection, user_id)
            await update.message.reply_text(
                f"Left {noun} {item_id}. Its {text_noun} is kept - use /{noun}s {item_id} to come back."
            )
            return

        if not new_text:
            current = store.load_text(collection, user_id, item_id)
            text = (
                f"{text_noun.capitalize()} for {noun} {item_id}:\n\n{current}"
                if current
                else f"{noun.capitalize()} {item_id} has no {text_noun} yet. Use /{noun} <text> to set one."
            )
            for chunk in _split_message(text):
                await update.message.reply_text(chunk)
            return

        if len(new_text) > MAX_TEXT_LENGTH:
            await update.message.reply_text(
                f"That {text_noun} is too long ({len(new_text)} chars). Keep it to {MAX_TEXT_LENGTH} chars or fewer."
            )
            return

        store.save_text(collection, user_id, item_id, new_text)
        await update.message.reply_text(f"{text_noun.capitalize()} for {noun} {item_id} updated.")

    return handler


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forget the messages of the thread the user is on. Profiles and task
    prompts live in the database, not in the history, so they are untouched."""
    task_id = context.bot_data["store"].current(TASKS, update.effective_user.id)
    context.chat_data.get("histories", {}).pop(task_id, None)
    where = f"for task {task_id}" if task_id else "off task"
    await update.message.reply_text(
        f"Conversation {where} cleared. Your profiles and task prompts are unchanged."
    )


def _parse_arg(update: Update) -> str:
    parts = update.message.text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _system_prompt(store: UserStore, user_id: int, task_id: str | None) -> str | None:
    """Who the user is, then what they are working on: the description of the
    profile they are on, then the prompt of the task they are on. Either half
    may be missing, and with neither the client falls back to its own default."""
    profile_id = store.current(PROFILES, user_id)
    parts = [
        store.load_text(collection, user_id, item_id)
        for collection, item_id in ((PROFILES, profile_id), (TASKS, task_id))
        if item_id
    ]
    return "\n\n".join(part for part in parts if part) or None


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
    task_id = store.current(TASKS, user_id)

    # One conversation per task, so switching tasks switches the thread and
    # switching back returns to it. No task is itself a thread. Profiles say
    # who the user is rather than what they are doing, so they share a thread.
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

    # Read on every request, so an edit takes effect on the next message.
    application.bot_data["store"] = UserStore()

    application.add_handler(CommandHandler("start", start))
    for collection in (PROFILES, TASKS):
        application.add_handler(CommandHandler(f"{collection.noun}s", _list_command(collection)))
        application.add_handler(CommandHandler(collection.noun, _text_command(collection)))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
