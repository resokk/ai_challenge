from __future__ import annotations

import logging
import re

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ExtBot,
    MessageHandler,
    TypeHandler,
    filters,
)

import protocol_log
from config import TELEGRAM_BOT_TOKEN
from deepseek_client import DeepSeekClient
from storage import PROFILES, TASK_STATE_SEQUENCE, TASK_STATES, TASKS, Collection, UserStore, next_state

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
MAX_HISTORY_MESSAGES = 100
MAX_TEXT_LENGTH = 2000
ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

# A task's todo goes to the model with every message, so it always knows what
# is outstanding. It is a field of the task rather than something either side
# said, so this heading tells the model which of the two it is reading.
TODO_PREFIX = "Current todo list for this task:\n"
# The run a task makes, and what the model is to do once nothing is left
# outstanding. Both come from the sequence itself, so what the model is told and
# what the bot does are the same thing said twice, and it is told which state
# comes next rather than left to work it out.
TODO_WORKFLOW_NOTE = "Task workflow: " + " -> ".join(TASK_STATE_SEQUENCE) + "."
TODO_ADVANCE_NOTE = "When every item on this list is ticked off, move the task to {state} with set_state."

# A checklist item: a box, optionally behind a bullet or a number. Anything in
# the box but a blank counts as ticked, so [x], [X] and [v] all mean done.
TODO_ITEM_PATTERN = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])?[ \t]*\[(.)\]", re.MULTILINE)

TASK_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "set_todo",
            "description": (
                "Replace the todo list of the task the user is working on. Use it whenever the user "
                "asks to tick an item off, add one, drop one or reword the list. Send the whole list "
                "back every time, not only the part that changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo": {
                        "type": "string",
                        "description": (
                            "The complete list, one item per line, each written as '- [ ] item', "
                            "or '- [x] item' once it is done."
                        ),
                    }
                },
                "required": ["todo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_state",
            "description": (
                "Move the task the user is working on to another state. Use it when the user asks to "
                "start it, send it for review, finish it, cancel it, or move it on."
            ),
            "parameters": {
                "type": "object",
                "properties": {"state": {"type": "string", "enum": list(TASK_STATES)}},
                "required": ["state"],
            },
        },
    },
)

BOT_COMMANDS = (
    BotCommand("profiles", "List your profiles, or switch to one"),
    BotCommand("profile", "Show or update the current profile's description, or 'none' to leave it"),
    BotCommand("tasks", "List your tasks, or switch to one"),
    BotCommand("task", "Show or update the current task's prompt, or 'none' to leave it"),
    BotCommand("state", "Show or set the current task's state"),
    BotCommand("todo", "Show or update the current task's todo"),
    BotCommand("rmtask", "Remove a task by id, with everything on it"),
    BotCommand("clear", "Forget this conversation"),
)


class _LoggingBot(ExtBot):
    """Telegram's side of the protocol log. Every reply the bot makes funnels
    into send_message, so overriding it here logs them all - including the ones
    python-telegram-bot sends on its own - with no call at each reply site."""

    async def send_message(self, chat_id, text, *args, **kwargs):
        message = await super().send_message(chat_id, text, *args, **kwargs)
        protocol_log.record("bot -> user", text)
        return message


async def _log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs before every handler: names the user this update belongs to, so
    everything logged while handling it lands in that user's file, then logs
    what they sent. Commands and chat alike - it is all protocol."""
    if update.effective_user:
        protocol_log.bind(update.effective_user.id)
    if update.message and update.message.text:
        protocol_log.record("user -> bot", update.message.text)


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

        note = ""
        if collection is TASKS:
            # Summarizing takes a round trip, so say something is happening.
            await update.message.chat.send_action("typing")
            note = await _switch_thread(context, store, user_id, current, item_id)

        store.select(collection, user_id, item_id)
        suffix = (
            ""
            if store.load_text(collection, user_id, item_id)
            else f" It has no {collection.text_noun} yet - set one with /{noun} <text>."
        )
        await update.message.reply_text(f"Switched to {noun} {item_id}.{suffix}{note}")

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
            note = ""
            if collection is TASKS:
                await update.message.chat.send_action("typing")
                note = await _switch_thread(context, store, user_id, item_id, None)
            store.deselect(collection, user_id)
            await update.message.reply_text(
                f"Left {noun} {item_id}. Its {text_noun} is kept - use /{noun}s {item_id} to come back.{note}"
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


async def _switch_thread(
    context: ContextTypes.DEFAULT_TYPE, store: UserStore, user_id: int, leaving: str | None, joining: str | None
) -> str:
    """Carry the conversation across a change of task: the thread of the task
    being left is summarized onto it, and the thread of the task being joined
    starts again from the summary that task was left carrying.

    So a task is its conversation as well as its prompt, and only the one the
    user is on is live - the rest are held as notes, which is what survives a
    restart. Returns a line for the reply - whether the thread resumed, and
    anything that could not be carried. The switch itself always goes through,
    because the user asked for it."""
    if leaving == joining:
        return ""

    note = ""
    history = context.chat_data.get("history", [])
    if leaving and history:
        try:
            summary = await DeepSeekClient.summarize(history)
        except Exception:
            logger.exception("Failed to summarize the conversation for task %s", leaving)
            summary = ""
        if summary:
            store.save_summary(user_id, leaving, summary)
        else:
            # Better to keep the older notes than to overwrite them with nothing.
            note = f" I couldn't summarize the conversation for task {leaving}, so it is not carried over."

    summary = store.load_summary(user_id, joining) if joining else None
    context.chat_data["history"] = (
        [{"role": "assistant", "content": f"{DeepSeekClient.SUMMARY_PREFIX}{summary}"}] if summary else []
    )
    return (" Picking up where you left off." if summary else "") + note


def _with_todo(history: list[dict], todo: str | None, following: str | None) -> list[dict]:
    """The thread as the model gets it: the task's todo as a standing note right
    after the summary the thread opens with, then the conversation. The note
    also carries the run a task makes and what to do when the list runs out -
    move it to `following` - so finishing the work and moving the task on are
    one instruction, given wherever the list is, rather than something the user
    has to ask for.

    The todo is read from the task on every send rather than kept in the history,
    so it is always the current one, it is there even on a thread that started
    before it was written or that a restart emptied, and it never reaches a
    summary - the summary covers what was said, and the todo is not that."""
    if not todo:
        return history
    content = f"{TODO_PREFIX}{todo}\n\n{TODO_WORKFLOW_NOTE}"
    if following:
        content += f"\n{TODO_ADVANCE_NOTE.format(state=following)}"
    note = {"role": "assistant", "content": content}
    opens_with_summary = bool(history) and history[0]["content"].startswith(DeepSeekClient.SUMMARY_PREFIX)
    at = 1 if opens_with_summary else 0
    return [*history[:at], note, *history[at:]]


async def state_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/state: without an argument, show the current task's state; with one of
    TASK_STATES, move the task to it. A task starts PENDING. State is where a
    task is, not what it says, so it does not reach the system prompt."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = store.current(TASKS, user_id)

    if not task_id:
        await update.message.reply_text("No task selected. Use /tasks <id> to start or switch to one.")
        return

    states = ", ".join(TASK_STATES)
    new_state = _parse_arg(update).upper()

    if not new_state:
        await update.message.reply_text(
            f"Task {task_id} is {store.load_state(user_id, task_id)}.\n\n"
            f"Use /state <state> to move it: {states}."
        )
        return

    if new_state not in TASK_STATES:
        await update.message.reply_text(f"Unknown state. Use one of: {states}.")
        return

    store.save_state(user_id, task_id, new_state)
    await update.message.reply_text(f"Task {task_id} is now {new_state}.")


async def todo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/todo: without an argument, show what the current task has outstanding;
    with text, replace it. The todo travels with the task, not with the
    conversation, so it is re-read on every switch and left out of summaries."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = store.current(TASKS, user_id)

    if not task_id:
        await update.message.reply_text("No task selected. Use /tasks <id> to start or switch to one.")
        return

    new_todo = _parse_arg(update)

    if not new_todo:
        current = store.load_todo(user_id, task_id)
        text = (
            f"Todo for task {task_id}:\n\n{current}"
            if current
            else f"Task {task_id} has no todo yet. Use /todo <text> to set one."
        )
        for chunk in _split_message(text):
            await update.message.reply_text(chunk)
        return

    if len(new_todo) > MAX_TEXT_LENGTH:
        await update.message.reply_text(
            f"That todo is too long ({len(new_todo)} chars). Keep it to {MAX_TEXT_LENGTH} chars or fewer."
        )
        return

    store.save_todo(user_id, task_id, new_todo)

    reply = f"Todo for task {task_id} updated."
    done_note = _all_done_note(store, user_id, task_id, new_todo)
    await update.message.reply_text(f"{reply}\n\n{done_note}" if done_note else reply)


def _all_done_note(store: UserStore, user_id: int, task_id: str, todo: str) -> str:
    """The offer made when a list typed at /todo comes up all ticked. A list the
    assistant writes gets no offer: it is told to move the task on itself, so the
    bot would be offering a move that has just been made."""
    if not _todo_is_done(todo):
        return ""
    following = next_state(store.load_state(user_id, task_id))
    return "Everything on it is done." + (
        f" Move the task to {following}? Use /state {following}." if following else ""
    )


def _todo_is_done(todo: str) -> bool:
    """Whether a todo written as a checklist has every box ticked. A todo with
    no boxes is prose rather than a checklist, so it has nothing to finish."""
    marks = TODO_ITEM_PATTERN.findall(todo)
    return bool(marks) and all(mark.strip() for mark in marks)


async def rmtask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/rmtask <id>: remove a task and everything on it - its prompt, state,
    summary and todo - permanently, with no undo.

    The id has to be typed out even to remove the task you are on, so nothing
    goes because it happened to be the current one; and an id that is not there
    is reported rather than passed over, since the difference between removing
    a task and removing nothing is the whole point of the command."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = _parse_arg(update)

    if not task_id:
        await update.message.reply_text("Use /rmtask <id> to remove a task. /tasks lists yours.")
        return

    was_current = store.current(TASKS, user_id) == task_id

    if not store.delete(TASKS, user_id, task_id):
        await update.message.reply_text(f"There is no task {task_id}. /tasks lists yours.")
        return

    if was_current:
        # The thread belonged to the task, and the task is gone.
        context.chat_data["history"] = []

    await update.message.reply_text(
        f"Removed task {task_id}, with its prompt, state, summary and todo."
        + (" You are on no task now." if was_current else "")
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forget the conversation the user is on: the live messages, and the summary
    the task is carrying, which stands in for the messages already folded into it -
    leaving one behind would bring the conversation back on the next switch.
    Profiles, task prompts, states and todos are fields of their own and are kept."""
    store = context.bot_data["store"]
    user_id = update.effective_user.id
    task_id = store.current(TASKS, user_id)
    context.chat_data["history"] = []
    if task_id:
        store.save_summary(user_id, task_id, "")
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
    if task_id:
        # Not decoration: the assistant can be asked to move the task on, and it
        # cannot pick the next state without knowing the one it is in.
        parts.append(f"Task {task_id} is currently in state {store.load_state(user_id, task_id)}.")
    return "\n\n".join(part for part in parts if part) or None


def _task_tool_runner(store: UserStore, user_id: int, task_id: str, done: list[str]):
    """Runs what the assistant asks for on the task the user is on, and collects
    a line about each change for the reply - a task moving or a list being
    rewritten is the bot's doing, so the user is told in the bot's own words
    rather than having to trust the model's account of it.

    Every argument comes from the model, so each is checked here exactly as the
    matching command checks what a user types; a refusal goes back as the result,
    which is the one thing the model can act on."""

    async def call_tool(name: str, arguments: dict) -> str:
        if name == "set_todo":
            todo = str(arguments.get("todo", "")).strip()
            if not todo:
                return "No todo was given."
            if len(todo) > MAX_TEXT_LENGTH:
                return f"That todo is too long. Keep it to {MAX_TEXT_LENGTH} characters or fewer."
            store.save_todo(user_id, task_id, todo)
            done.append(f"Todo for task {task_id} updated:\n{todo}")
            return "The todo list is updated."

        if name == "set_state":
            state = str(arguments.get("state", "")).upper()
            if state not in TASK_STATES:
                return f"Unknown state. Use one of: {', '.join(TASK_STATES)}."
            store.save_state(user_id, task_id, state)
            done.append(f"Task {task_id} is now {state}.")
            return f"The task is now {state}."

        return f"There is no tool called {name}."

    return call_tool


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

    # Only the task the user is on has a live thread; switching tasks summarizes
    # this one onto its task and resumes the other from its own summary. Profiles
    # say who the user is rather than what they are doing, so switching one keeps
    # the conversation you are in.
    history = context.chat_data.setdefault("history", [])
    history.append({"role": "user", "content": update.message.text})
    del history[:-MAX_HISTORY_MESSAGES]

    await update.message.chat.send_action("typing")

    system_prompt = _system_prompt(store, user_id, task_id)

    todo = store.load_todo(user_id, task_id) if task_id else None
    following = next_state(store.load_state(user_id, task_id)) if task_id else None

    # Only a task the user is on can be acted on, so off task there is nothing
    # to offer the model.
    done: list[str] = []
    tools, call_tool = (
        (list(TASK_TOOLS), _task_tool_runner(store, user_id, task_id, done)) if task_id else (None, None)
    )

    try:
        reply = await DeepSeekClient.get_reply(
            _with_todo(history, todo, following), system_prompt, tools, call_tool
        )
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

    # What the assistant changed is said by the bot, after the answer, and kept
    # out of the history: it is a receipt for the user, not part of the exchange.
    for chunk in _split_message("\n\n".join([reply, *done])):
        await update.message.reply_text(chunk)


def main() -> None:
    application = Application.builder().bot(_LoggingBot(TELEGRAM_BOT_TOKEN)).post_init(_post_init).build()

    # Read on every request, so an edit takes effect on the next message.
    application.bot_data["store"] = UserStore()

    # Group -1, so it sees every update before the handler that answers it.
    application.add_handler(TypeHandler(Update, _log_update), group=-1)

    application.add_handler(CommandHandler("start", start))
    for collection in (PROFILES, TASKS):
        application.add_handler(CommandHandler(f"{collection.noun}s", _list_command(collection)))
        application.add_handler(CommandHandler(collection.noun, _text_command(collection)))
    application.add_handler(CommandHandler("state", state_command))
    application.add_handler(CommandHandler("todo", todo_command))
    application.add_handler(CommandHandler("rmtask", rmtask_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
