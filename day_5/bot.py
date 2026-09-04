from __future__ import annotations

import logging

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from config import DEEPSEEK_MODEL, TELEGRAM_BOT_TOKEN
from deepseek_client import (
    DEFAULT_MODE,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    MAX_STOP_SEQUENCES,
    VALID_MODES,
    VALID_TEMPERATURES,
    generate_smart_prompt,
    get_broadcast_and_compare,
    get_broadcast_replies,
    get_reply,
    get_team_reply,
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

BOT_COMMANDS = (
    BotCommand("clear", "Clear history"),
    BotCommand("reset", "Reset everything"),
    BotCommand("mode", "Set reasoning mode"),
    BotCommand("temperature", "Set temperature"),
    BotCommand("broadcast", "Ask all models at once"),
    BotCommand("broadcast_and_compare", "Ask all models and compare replies"),
)

TEMPERATURE_OPTIONS = tuple(str(t) for t in VALID_TEMPERATURES)
TEMPERATURE_BY_LABEL = {str(t): t for t in VALID_TEMPERATURES}


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(BOT_COMMANDS)


ROOT_MENU_BUTTONS = (
    ("Mode", "mode"),
    ("Temperature", "temperature"),
    ("Max length", "maxlength"),
    ("Stop sequences", "stopon"),
    ("System prompt", "system"),
    ("Stats", "stats"),
    ("Clear history", "clear"),
    ("Reset all", "reset"),
)


def _build_root_menu() -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(label, callback_data=f"menu:{action}") for label, action in ROOT_MENU_BUTTONS]
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


def _back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("« Menu", callback_data="menu:root")]])


def _build_menu(prefix: str, options: tuple, current: str, columns: int = 1) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(f"✓ {opt}" if opt == current else opt, callback_data=f"{prefix}:{opt}")
        for opt in options
    ]
    rows = [buttons[i : i + columns] for i in range(0, len(buttons), columns)]
    rows.append([InlineKeyboardButton("« Menu", callback_data="menu:root")])
    return InlineKeyboardMarkup(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data.clear()
    await update.message.reply_text(
        f"Hi {update.effective_user.first_name}! I'm alive. Ask me anything, or use the menu below.",
        reply_markup=_build_root_menu(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me any message and I'll reply using DeepSeek.\n"
        "/menu - open the settings menu\n"
        "/clear - clear our conversation history\n"
        "/reset - clear our conversation history and revert all settings to default\n"
        "/system [prompt|reset] - view, set, or reset the system prompt\n"
        "/maxlength [tokens|unlimited] - view, set, or clear the max reply length\n"
        "/stopon [seq1,seq2,...|clear] - view, set, or clear stop sequences (max 4)\n"
        "/mode [DIRECT|STEP_BY_STEP|SMART_PROMPT|TEAM] - view or set the reasoning mode\n"
        f"/temperature [{'|'.join(TEMPERATURE_OPTIONS)}] - view or set the temperature\n"
        "/broadcast <prompt> - ask all DeepSeek models and return every reply\n"
        "/broadcast_and_compare <prompt> - ask all DeepSeek models, then have the pro model compare the "
        "replies as caveman, engineer, and humanitarian\n"
        "/stats - show usage stats for this chat"
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Main menu:", reply_markup=_build_root_menu())


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Conversation history cleared.", reply_markup=_back_menu())


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data.clear()
    await update.message.reply_text("Chat cleared and all settings reverted to default.", reply_markup=_back_menu())


def _parse_arg(update: Update, lower: bool = False) -> str:
    parts = update.message.text.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


def _system_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    text = (
        f"Current system prompt:\n\n{current}\n\n"
        "Use /system <prompt> to change it, or /system reset to restore the default."
    )
    return text, _back_menu()


def _maxlength_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("max_tokens")
    text = (
        f"Current max length: {current if current else 'unlimited'} tokens\n\n"
        "Use /maxlength <tokens> to set it, or /maxlength unlimited to clear it."
    )
    return text, _back_menu()


def _stopon_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("stop")
    body = "\n".join(current) if current else "none"
    text = (
        f"Current stop sequences:\n{body}\n\n"
        "Use /stopon <seq1,seq2,...> to set them, or /stopon clear to remove them."
    )
    return text, _back_menu()


def _mode_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("mode", DEFAULT_MODE)
    return f"Current mode: {current}", _build_menu("mode", VALID_MODES, current)


def _temperature_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("temperature", DEFAULT_TEMPERATURE)
    return f"Current temperature: {current}", _build_menu("temp", TEMPERATURE_OPTIONS, str(current), columns=3)


async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        text, markup = _system_view(context)
        await update.message.reply_text(text, reply_markup=markup)
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


async def maxlength_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update, lower=True)

    if not arg:
        text, markup = _maxlength_view(context)
        await update.message.reply_text(text, reply_markup=markup)
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
        text, markup = _stopon_view(context)
        await update.message.reply_text(text, reply_markup=markup)
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


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        text, markup = _mode_view(context)
        await update.message.reply_text(text, reply_markup=markup)
        return

    mode = arg.upper()
    if mode not in VALID_MODES:
        await update.message.reply_text(f"Unknown mode '{arg}'. Options: {', '.join(VALID_MODES)}")
        return

    context.chat_data["mode"] = mode
    await update.message.reply_text(f"Mode set to {mode}.")


async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        text, markup = _temperature_view(context)
        await update.message.reply_text(text, reply_markup=markup)
        return

    if arg not in TEMPERATURE_BY_LABEL:
        await update.message.reply_text(f"Unknown temperature '{arg}'. Options: {', '.join(TEMPERATURE_OPTIONS)}")
        return

    value = TEMPERATURE_BY_LABEL[arg]
    context.chat_data["temperature"] = value
    await update.message.reply_text(f"Temperature set to {value}.")


def _format_broadcast_result(result) -> str:
    usage = f"{result.usage['total_tokens']} tokens" if result.usage else "tokens n/a"
    stats = f"{usage}, {result.elapsed_seconds:.1f}s"
    return f"\r\n\r\n*{result.model}*\n{result.content or '(no reply)'}\n_{stats}_"


async def _send_broadcast_result(update: Update, result) -> None:
    for chunk in _split_message(_format_broadcast_result(result)):
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except BadRequest:
            await update.message.reply_text(chunk)


async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = _parse_arg(update)
    if not prompt:
        await update.message.reply_text("Usage: /broadcast <prompt>")
        return

    system_prompt = context.chat_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    await update.message.chat.send_action("typing")
    replies = await get_broadcast_replies(prompt, system_prompt)

    for result in replies:
        await _send_broadcast_result(update, result)


async def broadcast_and_compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = _parse_arg(update)
    if not prompt:
        await update.message.reply_text("Usage: /broadcast_and_compare <prompt>")
        return

    system_prompt = context.chat_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    await update.message.chat.send_action("typing")
    replies, comparisons = await get_broadcast_and_compare(prompt, system_prompt)

    for result in replies:
        await _send_broadcast_result(update, result)

    for result in comparisons:
        await _send_broadcast_result(update, result)


def _stats_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    history = context.chat_data.get("history", [])
    message_count = context.chat_data.get("message_count", 0)
    has_custom_prompt = "system_prompt" in context.chat_data
    max_tokens = context.chat_data.get("max_tokens")
    stop = context.chat_data.get("stop")
    mode = context.chat_data.get("mode", DEFAULT_MODE)
    temperature = context.chat_data.get("temperature", DEFAULT_TEMPERATURE)

    return (
        "Stats for this chat:\n"
        f"Messages sent: {message_count}\n"
        f"Messages in context: {len(history)}/{MAX_HISTORY_MESSAGES}\n"
        f"Model: {DEEPSEEK_MODEL}\n"
        f"System prompt: {'custom' if has_custom_prompt else 'default'}\n"
        f"Max length: {max_tokens if max_tokens else 'unlimited'} tokens\n"
        f"Stop sequences: {', '.join(stop) if stop else 'none'}\n"
        f"Mode: {mode}\n"
        f"Temperature: {temperature}"
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(_stats_text(context), reply_markup=_back_menu())


ROOT_MENU_VIEWS = {
    "mode": _mode_view,
    "temperature": _temperature_view,
    "system": _system_view,
    "maxlength": _maxlength_view,
    "stopon": _stopon_view,
}


async def menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    prefix, _, value = query.data.partition(":")

    if prefix == "mode" and value in VALID_MODES:
        context.chat_data["mode"] = value
        text, markup = _mode_view(context)
        await query.edit_message_text(text, reply_markup=markup)
        return

    if prefix == "temp" and value in TEMPERATURE_BY_LABEL:
        context.chat_data["temperature"] = TEMPERATURE_BY_LABEL[value]
        text, markup = _temperature_view(context)
        await query.edit_message_text(text, reply_markup=markup)
        return

    if prefix != "menu":
        return

    if value == "root":
        await query.edit_message_text("Main menu:", reply_markup=_build_root_menu())
    elif value in ROOT_MENU_VIEWS:
        text, markup = ROOT_MENU_VIEWS[value](context)
        await query.edit_message_text(text, reply_markup=markup)
    elif value == "stats":
        await query.edit_message_text(_stats_text(context), reply_markup=_back_menu())
    elif value == "clear":
        context.chat_data["history"] = []
        await query.edit_message_text("Conversation history cleared.", reply_markup=_back_menu())
    elif value == "reset":
        context.chat_data.clear()
        await query.edit_message_text(
            "Chat cleared and all settings reverted to default.", reply_markup=_back_menu()
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


async def _get_mode_reply(
    update: Update,
    history: list,
    system_prompt: str,
    response_format: str,
    max_tokens,
    stop,
    mode: str,
    temperature: float,
) -> str:
    if mode == "STEP_BY_STEP":
        outgoing = [*history[:-1], {**history[-1], "content": f"{history[-1]['content']}\nProceed step by step."}]
        return await get_reply(outgoing, system_prompt, response_format, max_tokens, stop, temperature)

    if mode == "SMART_PROMPT":
        generated_prompt = await generate_smart_prompt(history, temperature)
        for chunk in _split_message(f"Generated prompt:\n\n{generated_prompt}"):
            await update.message.reply_text(chunk)
        await update.message.chat.send_action("typing")
        outgoing = [*history[:-1], {"role": "user", "content": generated_prompt}]
        return await get_reply(outgoing, system_prompt, response_format, max_tokens, stop, temperature)

    if mode == "TEAM":
        await update.message.reply_text("Assembling the team (analyst, scientist, engineer, critic)...")
        await update.message.chat.send_action("typing")
        return await get_team_reply(history, system_prompt, response_format, max_tokens, stop, temperature)

    return await get_reply(history, system_prompt, response_format, max_tokens, stop, temperature)


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = context.chat_data.setdefault("history", [])
    history.append({"role": "user", "content": update.message.text})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    system_prompt = context.chat_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    response_format = context.chat_data.get("response_format", DEFAULT_RESPONSE_FORMAT)
    max_tokens = context.chat_data.get("max_tokens")
    stop = context.chat_data.get("stop")
    mode = context.chat_data.get("mode", DEFAULT_MODE)
    temperature = context.chat_data.get("temperature", DEFAULT_TEMPERATURE)

    await update.message.chat.send_action("typing")

    try:
        reply = await _get_mode_reply(
            update, history, system_prompt, response_format, max_tokens, stop, mode, temperature
        )
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
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("system", system_command))
    application.add_handler(CommandHandler("maxlength", maxlength_command))
    application.add_handler(CommandHandler("stopon", stopon_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("temperature", temperature_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("broadcast_and_compare", broadcast_and_compare_command))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CallbackQueryHandler(menu_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
