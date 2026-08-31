import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN
from deepseek_client import get_reply

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text(f"Hi {update.effective_user.first_name}! I'm alive. Ask me anything.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me any message and I'll reply using DeepSeek.\nUse /reset to clear our conversation history."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Conversation history cleared.")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = context.chat_data.setdefault("history", [])
    history.append({"role": "user", "content": update.message.text})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    await update.message.chat.send_action("typing")

    try:
        reply = await get_reply(history)
    except Exception:
        logger.exception("DeepSeek API call failed")
        await update.message.reply_text("Sorry, I couldn't reach DeepSeek right now. Try again in a bit.")
        return

    history.append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)


def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
