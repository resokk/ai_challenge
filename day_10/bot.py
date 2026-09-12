from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PersistenceInput,
    PicklePersistence,
    filters,
)

from config import DEEPSEEK_MODEL, TELEGRAM_BOT_TOKEN
from deepseek_client import DeepSeekClient, DeepSeekClientInterface
from storage import DEFAULT_BRANCH, MessageStore

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).with_name("chat_settings.pkl")

MAX_SYSTEM_PROMPT_LENGTH = 2000
TELEGRAM_MESSAGE_LIMIT = 4096
MAX_TOKENS_LIMIT = 8192

STARTED_AT = time.monotonic()

BOT_COMMANDS = (
    BotCommand("clear", "Clear history"),
    BotCommand("stats", "Show chat and token stats"),
    BotCommand("branch", "Switch conversation branch"),
    BotCommand("mode", "Set reasoning mode"),
    BotCommand("uptime", "Show how long the bot has been running"),
    BotCommand("stop", "Shut the bot down"),
)

TEMPERATURE_OPTIONS = tuple(str(t) for t in DeepSeekClient.VALID_TEMPERATURES)
TEMPERATURE_BY_LABEL = {str(t): t for t in DeepSeekClient.VALID_TEMPERATURES}

COMPRESSION_OPTIONS = tuple(str(c) for c in DeepSeekClient.VALID_COMPRESSIONS)
COMPRESSION_BY_LABEL = {str(c): c for c in DeepSeekClient.VALID_COMPRESSIONS}

COMPRESSION_MODE_SUMMARIES = {
    "SUMMARIZE": "replace the oldest half with a summary of it",
    "WINDOW": "keep the newest N messages, forget the rest",
    "BRANCH": "carry on in a new branch, keeping this one",
    "FACTS": "remember facts from every exchange, keep the newest N messages",
}


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(BOT_COMMANDS)


ROOT_MENU_BUTTONS = (
    ("Mode", "mode"),
    ("Temperature", "temperature"),
    ("Max length", "maxlength"),
    ("Stop sequences", "stopon"),
    ("System prompt", "system"),
    ("Compression", "compression"),
    ("Compression mode", "compression_mode"),
    ("Branch", "branch"),
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


def _get_client(update: Update, context: ContextTypes.DEFAULT_TYPE) -> DeepSeekClientInterface:
    """Clients live in bot_data, not chat_data: chat_data is pickled between
    restarts to keep the chat's settings, and a client owns a SQLite connection
    that cannot be pickled (nor should be - its history is already durable)."""
    clients = context.bot_data.setdefault("clients", {})
    chat_id = update.effective_chat.id
    client = clients.get(chat_id)
    if client is None:
        branch = context.chat_data.get("branch", DEFAULT_BRANCH)
        client = clients[chat_id] = DeepSeekClient(context.bot_data["store"], branch)
    return client


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _reset_chat(update, context)
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
        f"/compression [{'|'.join(COMPRESSION_OPTIONS)}] - view or set the message limit at which "
        "the conversation is compressed (0 is off)\n"
        f"/compression_mode [{'|'.join(DeepSeekClient.VALID_COMPRESSION_MODES)}] - view or set how it "
        "is compressed when that limit is reached\n"
        "/branch [name] - view your branches, or switch to (or start) one\n"
        "/broadcast <prompt> - ask all DeepSeek models and return every reply\n"
        "/broadcast_and_compare <prompt> - ask all DeepSeek models, then have the pro model compare the "
        "replies as caveman, engineer, and humanitarian\n"
        "/stats - show usage stats for this chat, including the tokens behind the last message\n"
        "/uptime - show how long the bot has been running\n"
        "/stop - shut the bot down"
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Main menu:", reply_markup=_build_root_menu())


async def uptime_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    elapsed = timedelta(seconds=int(time.monotonic() - STARTED_AT))
    await update.message.reply_text(f"Uptime: {elapsed}")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Shutting down. Start me again from the command line.")
    logger.info("Shutdown requested by chat %s", update.effective_chat.id)
    # Ends run_polling() gracefully, so main() returns and the process exits.
    context.application.stop_running()


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _get_client(update, context).clear_history()
    await update.message.reply_text("Conversation history cleared.", reply_markup=_back_menu())


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _reset_chat(update, context)
    await update.message.reply_text("Chat cleared and all settings reverted to default.", reply_markup=_back_menu())


def _reset_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Empty the current branch and revert every setting - including the branch
    itself, so the client and the settings cannot end up disagreeing about which
    conversation this chat is on."""
    client = _get_client(update, context)
    client.clear_history()
    context.chat_data.clear()
    client.switch_branch(DEFAULT_BRANCH)


def _parse_arg(update: Update, lower: bool = False) -> str:
    parts = update.message.text.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


def _system_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("system_prompt", DeepSeekClient.DEFAULT_SYSTEM_PROMPT)
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
    current = context.chat_data.get("mode", DeepSeekClient.DEFAULT_MODE)
    return f"Current mode: {current}", _build_menu("mode", DeepSeekClient.VALID_MODES, current)


def _temperature_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("temperature", DeepSeekClient.DEFAULT_TEMPERATURE)
    return f"Current temperature: {current}", _build_menu("temp", TEMPERATURE_OPTIONS, str(current), columns=3)


def _compression_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = context.chat_data.get("compression", DeepSeekClient.DEFAULT_COMPRESSION)
    return f"Current compression: {_compression_label(current)}", _build_menu(
        "comp", COMPRESSION_OPTIONS, str(current), columns=3
    )


def _compression_label(value: int) -> str:
    return "off" if not value else f"every {value} messages"


def _compression_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.chat_data.get("compression_mode", DeepSeekClient.DEFAULT_COMPRESSION_MODE)


def _compression_mode_view(context: ContextTypes.DEFAULT_TYPE) -> tuple:
    current = _compression_mode(context)
    body = "\n".join(f"{mode} - {summary}" for mode, summary in COMPRESSION_MODE_SUMMARIES.items())
    text = f"Current compression mode: {current}\n\nWhat each one does when the limit is reached:\n{body}"
    return text, _build_menu("cmode", DeepSeekClient.VALID_COMPRESSION_MODES, current)


def _branch_view(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple:
    client = _get_client(update, context)
    branches = client.list_branches()
    text = (
        f"Current branch: {client.branch}\n\n"
        "Each branch is a separate conversation with its own history, summary and facts.\n"
        "Use /branch <name> to switch to one, or to start it if it is new."
    )
    return text, _build_menu("branch", tuple(branches), client.branch)


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

    if len(sequences) > DeepSeekClient.MAX_STOP_SEQUENCES:
        await update.message.reply_text(
            f"Too many stop sequences ({len(sequences)}). Max is {DeepSeekClient.MAX_STOP_SEQUENCES}."
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
    if mode not in DeepSeekClient.VALID_MODES:
        await update.message.reply_text(f"Unknown mode '{arg}'. Options: {', '.join(DeepSeekClient.VALID_MODES)}")
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


async def compression_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        text, markup = _compression_view(context)
        await update.message.reply_text(text, reply_markup=markup)
        return

    if arg not in COMPRESSION_BY_LABEL:
        await update.message.reply_text(
            f"Unknown compression '{arg}'. Options: {', '.join(COMPRESSION_OPTIONS)} (0 turns it off)."
        )
        return

    value = COMPRESSION_BY_LABEL[arg]
    context.chat_data["compression"] = value
    await update.message.reply_text(f"Compression set to {_compression_label(value)}.")


async def compression_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        text, markup = _compression_mode_view(context)
        await update.message.reply_text(text, reply_markup=markup)
        return

    mode = arg.upper()
    if mode not in DeepSeekClient.VALID_COMPRESSION_MODES:
        await update.message.reply_text(
            f"Unknown compression mode '{arg}'. Options: {', '.join(DeepSeekClient.VALID_COMPRESSION_MODES)}"
        )
        return

    context.chat_data["compression_mode"] = mode
    await update.message.reply_text(f"Compression mode set to {mode} - {COMPRESSION_MODE_SUMMARIES[mode]}.")


async def branch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = _parse_arg(update)

    if not arg:
        text, markup = _branch_view(update, context)
        await update.message.reply_text(text, reply_markup=markup)
        return

    problem = _branch_name_problem(arg)
    if problem:
        await update.message.reply_text(problem)
        return

    await update.message.reply_text(_switch_branch(update, context, arg), reply_markup=_back_menu())


def _branch_name_problem(name: str) -> str | None:
    """Branch names travel in callback data, which is length-limited and colon
    separated, so keep them short and colon-free."""
    if len(name) > DeepSeekClient.MAX_BRANCH_NAME_LENGTH:
        return f"Branch name is too long. Keep it under {DeepSeekClient.MAX_BRANCH_NAME_LENGTH} characters."
    if ":" in name:
        return "Branch names cannot contain ':'."
    return None


def _switch_branch(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str) -> str:
    client = _get_client(update, context)
    if name == client.branch:
        return f"Already on branch '{name}'."
    existed = name in client.list_branches()
    client.switch_branch(name)
    context.chat_data["branch"] = name
    if existed:
        return f"Switched to branch '{name}' ({len(client.history)} messages in context)."
    return f"Started branch '{name}'. It begins empty; the branch you were on is untouched."


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

    system_prompt = context.chat_data.get("system_prompt", DeepSeekClient.DEFAULT_SYSTEM_PROMPT)
    await update.message.chat.send_action("typing")
    replies = await _get_client(update, context).get_broadcast_replies(prompt, system_prompt)

    for result in replies:
        await _send_broadcast_result(update, result)


async def broadcast_and_compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = _parse_arg(update)
    if not prompt:
        await update.message.reply_text("Usage: /broadcast_and_compare <prompt>")
        return

    system_prompt = context.chat_data.get("system_prompt", DeepSeekClient.DEFAULT_SYSTEM_PROMPT)
    await update.message.chat.send_action("typing")
    replies, comparisons = await _get_client(update, context).get_broadcast_and_compare(prompt, system_prompt)

    for result in replies:
        await _send_broadcast_result(update, result)

    for result in comparisons:
        await _send_broadcast_result(update, result)


def _token_usage_text(client: DeepSeekClientInterface | None) -> str:
    usage = client.last_token_usage if client else None
    if usage is None:
        return "Tokens in last exchange: none yet"
    return (
        "Tokens in last exchange:\n"
        f"  Request: {usage.request}\n"
        f"  Dialog history: {usage.history}\n"
        f"  Reply: {usage.reply}\n"
        f"  Total billed: {usage.total}"
    )


def _stats_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    client = _get_client(update, context)
    history = client.history
    message_count = context.chat_data.get("message_count", 0)
    has_custom_prompt = "system_prompt" in context.chat_data
    max_tokens = context.chat_data.get("max_tokens")
    stop = context.chat_data.get("stop")
    mode = context.chat_data.get("mode", DeepSeekClient.DEFAULT_MODE)
    temperature = context.chat_data.get("temperature", DeepSeekClient.DEFAULT_TEMPERATURE)
    compression = context.chat_data.get("compression", DeepSeekClient.DEFAULT_COMPRESSION)
    compression_mode = _compression_mode(context)
    if compression_mode == "FACTS":
        compression_mode = f"{compression_mode} ({len(client.facts)} facts known)"

    return (
        "Stats for this chat:\n"
        f"Messages sent: {message_count}\n"
        f"Messages in context: {len(history)}/{DeepSeekClient.MAX_HISTORY_MESSAGES}\n"
        f"Model: {DEEPSEEK_MODEL}\n"
        f"System prompt: {'custom' if has_custom_prompt else 'default'}\n"
        f"Max length: {max_tokens if max_tokens else 'unlimited'} tokens\n"
        f"Stop sequences: {', '.join(stop) if stop else 'none'}\n"
        f"Mode: {mode}\n"
        f"Temperature: {temperature}\n"
        f"Branch: {client.branch}\n"
        f"Compression: {_compression_label(compression)}\n"
        f"Compression mode: {compression_mode}\n"
        f"{_token_usage_text(client)}"
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(_stats_text(update, context), reply_markup=_back_menu())


ROOT_MENU_VIEWS = {
    "mode": _mode_view,
    "temperature": _temperature_view,
    "compression": _compression_view,
    "compression_mode": _compression_mode_view,
    "system": _system_view,
    "maxlength": _maxlength_view,
    "stopon": _stopon_view,
}


async def menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    prefix, _, value = query.data.partition(":")

    if prefix == "mode" and value in DeepSeekClient.VALID_MODES:
        context.chat_data["mode"] = value
        text, markup = _mode_view(context)
        await query.edit_message_text(text, reply_markup=markup)
        return

    if prefix == "temp" and value in TEMPERATURE_BY_LABEL:
        context.chat_data["temperature"] = TEMPERATURE_BY_LABEL[value]
        text, markup = _temperature_view(context)
        await query.edit_message_text(text, reply_markup=markup)
        return

    if prefix == "cmode" and value in DeepSeekClient.VALID_COMPRESSION_MODES:
        context.chat_data["compression_mode"] = value
        text, markup = _compression_mode_view(context)
        await query.edit_message_text(text, reply_markup=markup)
        return

    if prefix == "branch":
        await query.edit_message_text(_switch_branch(update, context, value), reply_markup=_back_menu())
        return

    if prefix == "comp" and value in COMPRESSION_BY_LABEL:
        context.chat_data["compression"] = COMPRESSION_BY_LABEL[value]
        text, markup = _compression_view(context)
        await query.edit_message_text(text, reply_markup=markup)
        return

    if prefix != "menu":
        return

    if value == "root":
        await query.edit_message_text("Main menu:", reply_markup=_build_root_menu())
    elif value == "branch":
        text, markup = _branch_view(update, context)
        await query.edit_message_text(text, reply_markup=markup)
    elif value in ROOT_MENU_VIEWS:
        text, markup = ROOT_MENU_VIEWS[value](context)
        await query.edit_message_text(text, reply_markup=markup)
    elif value == "stats":
        await query.edit_message_text(_stats_text(update, context), reply_markup=_back_menu())
    elif value == "clear":
        _get_client(update, context).clear_history()
        await query.edit_message_text("Conversation history cleared.", reply_markup=_back_menu())
    elif value == "reset":
        _reset_chat(update, context)
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
    client: DeepSeekClientInterface,
    system_prompt: str,
    response_format: str,
    max_tokens,
    stop,
    mode: str,
    temperature: float,
) -> str:
    if mode == "STEP_BY_STEP":
        return await client.get_step_by_step_reply(system_prompt, response_format, max_tokens, stop, temperature)

    if mode == "SMART_PROMPT":
        generated_prompt = await client.generate_smart_prompt(temperature)
        for chunk in _split_message(f"Generated prompt:\n\n{generated_prompt}"):
            await update.message.reply_text(chunk)
        await update.message.chat.send_action("typing")
        return await client.get_reply_with_replacement(
            generated_prompt, system_prompt, response_format, max_tokens, stop, temperature
        )

    if mode == "TEAM":
        await update.message.reply_text("Assembling the team (analyst, scientist, engineer, critic)...")
        await update.message.chat.send_action("typing")
        return await client.get_team_reply(system_prompt, response_format, max_tokens, stop, temperature)

    return await client.get_reply(system_prompt, response_format, max_tokens, stop, temperature)


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    client = _get_client(update, context)
    client.add_user_message(update.message.text)

    system_prompt = context.chat_data.get("system_prompt", DeepSeekClient.DEFAULT_SYSTEM_PROMPT)
    response_format = context.chat_data.get("response_format", DeepSeekClient.DEFAULT_RESPONSE_FORMAT)
    max_tokens = context.chat_data.get("max_tokens")
    stop = context.chat_data.get("stop")
    mode = context.chat_data.get("mode", DeepSeekClient.DEFAULT_MODE)
    temperature = context.chat_data.get("temperature", DeepSeekClient.DEFAULT_TEMPERATURE)

    await update.message.chat.send_action("typing")

    try:
        reply = await _get_mode_reply(
            update, client, system_prompt, response_format, max_tokens, stop, mode, temperature
        )
    except Exception:
        logger.exception("DeepSeek API call failed")
        client.discard_last_user_message()
        await update.message.reply_text("Sorry, I couldn't reach DeepSeek right now. Try again in a bit.")
        return

    if not reply:
        logger.warning("DeepSeek returned an empty reply")
        client.discard_last_user_message()
        await update.message.reply_text("Sorry, I didn't get a usable reply from DeepSeek. Try rephrasing.")
        return

    client.add_assistant_message(reply)
    context.chat_data["message_count"] = context.chat_data.get("message_count", 0) + 1

    parse_mode = ParseMode.MARKDOWN if response_format == "markdown" else None
    for chunk in _split_message(reply):
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except BadRequest:
            logger.warning("Failed to render reply as Markdown, falling back to plain text")
            await update.message.reply_text(chunk)

    # After the answer is out, so compression never delays the user's reply.
    compression = context.chat_data.get("compression", DeepSeekClient.DEFAULT_COMPRESSION)
    compressed = await client.compress_history(compression, _compression_mode(context))
    if compressed:
        await update.message.reply_text(_compression_notice(compressed, client))


def _compression_notice(result, client: DeepSeekClientInterface) -> str:
    lines = [f"{result.mode}: {result.detail}" if result.detail else result.mode]
    if result.folded:
        lines.append(f"Took {result.folded} messages out of the context ({len(client.history)} left).")
    lines.append(f"History tokens: {result.tokens_before} before, {result.tokens_after} after.")
    return "\n".join(lines)


def main() -> None:
    # Settings (mode, temperature, compression, system prompt, ...) live in
    # chat_data, so persist it: a restart must not silently revert a chat to
    # defaults the way an unpersisted bot does. Only chat_data is stored -
    # bot_data holds the message store and the live clients.
    persistence = PicklePersistence(
        filepath=SETTINGS_PATH,
        store_data=PersistenceInput(bot_data=False, chat_data=True, user_data=False, callback_data=False),
    )
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .persistence(persistence)
        .post_init(_post_init)
        .build()
    )
    application.bot_data["store"] = MessageStore()

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
    application.add_handler(CommandHandler("compression", compression_command))
    application.add_handler(CommandHandler("compression_mode", compression_mode_command))
    application.add_handler(CommandHandler("branch", branch_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("broadcast_and_compare", broadcast_and_compare_command))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("uptime", uptime_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CallbackQueryHandler(menu_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
