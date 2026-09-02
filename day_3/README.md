# Day 3 — Telegram Bot

Same base as [day_2](../day_2): a Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible), plus per-chat system prompt, response format, max length, and stop sequence controls.

## Setup

```bash
cd day_2
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

- `/start` — greets the user and clears all chat state (history, system prompt, stats)
- `/help` — shows usage info
- `/clear` — clears the conversation history for the current chat (keeps your custom system prompt and other settings)
- `/reset` — clears the conversation history and reverts all settings (system prompt, format, max length, stop sequences) back to default
- `/system` — shows the current system prompt
- `/system <prompt>` — sets a custom system prompt for the current chat (max 2000 chars)
- `/system reset` — resets the system prompt back to the default
- `/format` — shows the current response format
- `/format text` / `/format markdown` / `/format json` / `/format xml` — sets the response format for replies (`text` is the default; `markdown` asks DeepSeek for Markdown and renders it in Telegram, falling back to plain text if the output isn't valid Telegram Markdown; `json` asks DeepSeek to return a valid JSON object; `xml` asks DeepSeek for JSON the same way, then converts it to XML client-side before sending, falling back to the raw JSON if it can't be parsed)
- `/maxlength` — shows the current max reply length
- `/maxlength <tokens>` — caps replies at up to this many tokens (1–8192), passed to DeepSeek as `max_tokens`
- `/maxlength unlimited` — clears the cap (default; DeepSeek's own default limit applies)
- `/stopon` — shows the current stop sequences
- `/stopon <seq1,seq2,...>` — sets up to 4 comma-separated stop sequences; DeepSeek stops generating as soon as it produces one
- `/stopon clear` — clears all stop sequences (default; nothing set)
- `/stats` — shows messages sent, messages currently kept in context, the active model, the response format, the max reply length, stop sequences, and whether a custom system prompt is set
- Any other text message is sent to DeepSeek and the reply is sent back, with the last 20 messages kept as context per chat
