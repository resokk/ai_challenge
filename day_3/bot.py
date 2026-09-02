import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import DEEPSEEK_MODEL, TELEGRAM_BOT_TOKEN
from deepseek_client import (
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_SYSTEM_PROMPT,
    MAX_STOP_SEQUENCES,
    VALID_RESPONSE_FORMATS,
    get_reply,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20
MAX_SYSTEM_PROMPT_LENGTH = 2000
TELEGRAM_MESSAGE_LIMIT = 4096
MAX_TOKENS_LIMIT = 8192


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data.clear()
    await update.message.reply_text(f"Hi {update.effective_user.first_name}! I'm alive. Ask me anything.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me any message and I'll reply using DeepSeek.\n"
        "/clear - clear our conversation history\n"
        "/reset - clear our conversation history and revert all settings to default\n"
        "/system [prompt|reset] - view, set, or reset the system prompt\n"
        "/format [text|markdown|json|xml] - view or set the response format\n"
        "/maxlength [tokens|unlimited] - view, set, or clear the max reply length\n"
        "/stopon [seq1,seq2,...|clear] - view, set, or clear stop sequences (max 4)\n"
        "/stats - show usage stats for this chat"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Conversation history cleared.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data.clear()
    await update.message.reply_text("Chat cleared and all settings reverted to default.")


def _parse_arg(update: Update, lower: bool = False) -> str:
    parts = update.message.text.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        current = context.chat_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        await update.message.reply_text(f"Current system prompt:\n\n{current}")
        return

    if arg.lower() == "reset":
        context.chat_data.pop("system_prompt", None)
        await update.message.reply_text("System prompt reset to default.")
        return

    if len(arg) > MAX_SYSTEM_PROMPT_LENGTH:
        await update.message.reply_text(
            f"System prompt is too long ({len(arg)} chars). Keep it under {MAX_SYSTEM_PROMPT_LENGTH} chars."
        )
        return

    context.chat_data["system_prompt"] = arg
    await update.message.reply_text("System prompt updated.")


async def format_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update, lower=True)
    options = ", ".join(VALID_RESPONSE_FORMATS)

    if not arg:
        current = context.chat_data.get("response_format", DEFAULT_RESPONSE_FORMAT)
        await update.message.reply_text(f"Current format: {current}\nOptions: {options}")
        return

    if arg not in VALID_RESPONSE_FORMATS:
        await update.message.reply_text(f"Unknown format '{arg}'. Options: {options}")
        return

    context.chat_data["response_format"] = arg
    await update.message.reply_text(f"Response format set to {arg}.")


async def maxlength_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update, lower=True)

    if not arg:
        current = context.chat_data.get("max_tokens")
        await update.message.reply_text(f"Current max length: {current if current else 'unlimited'} tokens")
        return

    if arg == "unlimited":
        context.chat_data.pop("max_tokens", None)
        await update.message.reply_text("Max length reset to unlimited.")
        return

    try:
        value = int(arg)
    except ValueError:
        value = None

    if value is None or not (0 < value <= MAX_TOKENS_LIMIT):
        await update.message.reply_text(
            f"Max length must be a number between 1 and {MAX_TOKENS_LIMIT}, or 'unlimited'."
        )
        return

    context.chat_data["max_tokens"] = value
    await update.message.reply_text(f"Max length set to {value} tokens.")


async def stopon_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        current = context.chat_data.get("stop")
        if current:
            await update.message.reply_text("Current stop sequences:\n" + "\n".join(current))
        else:
            await update.message.reply_text("No stop sequences set.")
        return

    if arg.lower() == "clear":
        context.chat_data.pop("stop", None)
        await update.message.reply_text("Stop sequences cleared.")
        return

    sequences = [s.strip() for s in arg.split(",") if s.strip()]

    if not sequences:
        await update.message.reply_text("Provide one or more comma-separated stop sequences, or 'clear'.")
        return

    if len(sequences) > MAX_STOP_SEQUENCES:
        await update.message.reply_text(
            f"Too many stop sequences ({len(sequences)}). Max is {MAX_STOP_SEQUENCES}."
        )
        return

    context.chat_data["stop"] = sequences
    await update.message.reply_text(f"Stop sequences set: {', '.join(sequences)}")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = context.chat_data.get("history", [])
    message_count = context.chat_data.get("message_count", 0)
    has_custom_prompt = "system_prompt" in context.chat_data
    response_format = context.chat_data.get("response_format", DEFAULT_RESPONSE_FORMAT)
    max_tokens = context.chat_data.get("max_tokens")
    stop = context.chat_data.get("stop")

    await update.message.reply_text(
        "Stats for this chat:\n"
        f"Messages sent: {message_count}\n"
        f"Messages in context: {len(history)}/{MAX_HISTORY_MESSAGES}\n"
        f"Model: {DEEPSEEK_MODEL}\n"
        f"System prompt: {'custom' if has_custom_prompt else 'default'}\n"
        f"Format: {response_format}\n"
        f"Max length: {max_tokens if max_tokens else 'unlimited'} tokens\n"
        f"Stop sequences: {', '.join(stop) if stop else 'none'}"
    )


def _split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list:
    """Split text into <= limit-sized chunks, preferring to break on blank
    lines, then newlines, then spaces, so Markdown entities (e.g. **bold**)
    are less likely to be cut in half by a hard slice."""
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
    history = context.chat_data.setdefault("history", [])
    history.append({"role": "user", "content": update.message.text})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    system_prompt = context.chat_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    response_format = context.chat_data.get("response_format", DEFAULT_RESPONSE_FORMAT)
    max_tokens = context.chat_data.get("max_tokens")
    stop = context.chat_data.get("stop")

    await update.message.chat.send_action("typing")

    try:
        reply = await get_reply(history, system_prompt, response_format, max_tokens, stop)
    except Exception:
        logger.exception("DeepSeek API call failed")
        if history and history[-1]["role"] == "user":
            history.pop()
        await update.message.reply_text("Sorry, I couldn't reach DeepSeek right now. Try again in a bit.")
        return

    if not reply:
        logger.warning("DeepSeek returned an empty reply")
        if history and history[-1]["role"] == "user":
            history.pop()
        await update.message.reply_text("Sorry, I didn't get a usable reply from DeepSeek. Try rephrasing.")
        return

    history.append({"role": "assistant", "content": reply})
    context.chat_data["message_count"] = context.chat_data.get("message_count", 0) + 1

    parse_mode = ParseMode.MARKDOWN if response_format == "markdown" else None
    for chunk in _split_message(reply):
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except BadRequest:
            logger.warning("Failed to render reply as Markdown, falling back to plain text")
            await update.message.reply_text(chunk)


def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("system", system_command))
    application.add_handler(CommandHandler("format", format_command))
    application.add_handler(CommandHandler("maxlength", maxlength_command))
    application.add_handler(CommandHandler("stopon", stopon_command))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
