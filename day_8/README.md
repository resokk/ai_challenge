# Day 8 — Telegram Bot

Same base as [day_7](../day_7): a Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible), per-chat settings behind a button menu, reasoning modes, multi-model broadcast, and SQLite-persisted history. This version adds **token accounting**: every answered message records how many tokens the request, the dialog history, and the reply cost, and `/stats` reports them.

## Setup

```bash
cd day_8
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

On startup the bot registers its command list with Telegram (`setMyCommands`), so those commands show up in Telegram's `/` autocomplete menu.

## Token counting

After every message the bot answers, `DeepSeekClient` records a `TokenUsage` for that exchange and `/stats` (and the Stats menu button) shows it:

```
Tokens in last exchange:
  Request: 7            <- your latest message on its own
  Dialog history: 16    <- every message in the context sent with it
  Reply: 9              <- the model's answer
  Total billed: 66      <- the whole call, as reported by the API
```

`Request` and `Dialog history` are counted locally with `tiktoken`'s `cl100k_base` — the same tokenizer `storage.py` already uses for its per-message `tokens` column — so they are close estimates, not DeepSeek's own tokenizer. `Reply` and `Total billed` come from the API's `usage` field, so they are exact; `Total billed` is larger than request + history because it also covers the system prompt and the chat template.

The counts live in memory for the running process only: they describe an exchange, so after a restart (or `/clear`, `/reset`, `/start`) `/stats` reports `none yet` until the bot answers a message again. Multi-call modes (`SMART_PROMPT`, `TEAM`) report the final answer-producing call; `/broadcast` keeps its own per-model token line and does not touch these.

## What it does

- `/start` — greets the user, clears all chat state (history, settings, stats), and shows the main menu
- `/help` — shows usage info
- `/menu` — opens the main button menu: Mode, Temperature, Max length, Stop sequences, System prompt, Stats, Clear history, Reset all. Every submenu has a "« Menu" button to come back
- `/clear` — clears the conversation history for the current chat, in memory and in SQLite (keeps settings)
- `/reset` — clears the conversation history and reverts all settings back to default
- `/system` / `/system <prompt>` / `/system reset` — view, set (max 2000 chars), or reset the system prompt
- `/maxlength` / `/maxlength <tokens>` / `/maxlength unlimited` — view, cap (1–8192, passed as `max_tokens`), or clear the reply length limit
- `/stopon` / `/stopon <seq1,seq2,...>` / `/stopon clear` — view, set (max 4), or clear stop sequences
- `/temperature` / `/temperature <value>` — view or set the sampling temperature (0–2, default 1)
- `/mode` / `/mode <DIRECT|STEP_BY_STEP|SMART_PROMPT|TEAM>` — view or set how each message is processed:
  - `DIRECT` (default) — forwards your message to DeepSeek as-is
  - `STEP_BY_STEP` — appends "Proceed step by step." for that request only; stored history keeps the original text
  - `SMART_PROMPT` — asks DeepSeek to rewrite your message into a better prompt, shows it to you, then answers with it
  - `TEAM` — an analyst, a scientist, and an engineer each draft an answer, then a critic synthesizes one final reply (4 API calls per message)
- `/broadcast <prompt>` — asks every DeepSeek model the same prompt and returns each reply with its token count and latency
- `/broadcast_and_compare <prompt>` — same, then has the pro model compare the replies as a caveman, an engineer, and a humanitarian
- `/stats` — messages sent, messages in context, model, system prompt, max length, stop sequences, mode, temperature, and the token counts for the last exchange
- `/uptime` — how long the bot process has been running
- `/stop` — shuts the bot down gracefully
- Any other text message is sent to DeepSeek and the reply is sent back, with the last 1000 messages kept as context and persisted to `messages.db`, so a restart keeps the conversation
