# Day 1 — Telegram Bot

Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible).

## Setup

```bash
cd day_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in TELEGRAM_BOT_TOKEN and DEEPSEEK_API_KEY
```

- Get a bot token from [@BotFather](https://t.me/BotFather) on Telegram.
- Get a DeepSeek API key from the [DeepSeek platform](https://platform.deepseek.com/api_keys).

## Run

```bash
python bot.py
```

## What it does

- `/start` — greets the user and clears conversation history
- `/help` — shows usage info
- `/reset` — clears the conversation history for the current chat
- Any other text message is sent to DeepSeek and the reply is sent back, with the last 20 messages kept as context per chat
