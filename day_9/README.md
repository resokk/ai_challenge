# Day 9 — Telegram Bot

Same base as [day_8](../day_8): a Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible), per-chat settings behind a button menu, reasoning modes, multi-model broadcast, SQLite-persisted history, and per-exchange token accounting. This version adds **history compression**: once a conversation reaches a chosen length, its oldest half is summarized and the summary takes their place, so long chats keep their context without carrying every message forever.

## Setup

```bash
cd day_9
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

## History compression

`/compression` (or the Compression menu button) takes one of `0`, `10`, `20`, `50`, `100`:

- `0` (default) — compression is off; the history grows until it hits the 1000-message context window.
- anything else — once the history reaches that many messages, the bot summarizes the oldest half and replaces those messages with a single assistant note beginning `Summary of the earlier part of our conversation:`. It reports what that cost and saved:

  ```
  Compressed the 6 oldest messages into a summary (5 messages now in context).
  History tokens: 1280 before, 533 after.
  ```

  Both figures are the whole dialog history counted the same way `/stats` counts it. Compression usually shrinks it sharply, but it is not guaranteed to — a summary of a handful of very short messages can be longer than the messages themselves, which is why the bot reports both numbers rather than a promise.

The split lands on the next user turn rather than exactly halfway, so the summary and the surviving messages still alternate assistant/user — with a 10-message history that folds 6 and keeps 4. Summarizing costs one extra API call, made **after** your reply has been sent so it never adds latency to the answer, and it runs at a fixed low temperature (0.2) regardless of the chat's, since a summary wants fidelity rather than variety. If that call fails the history is simply left uncompressed and the bot retries after the next message.

Compression shrinks the **stored conversation as well as the context**. When a summary is recorded, the messages it covers are deleted, along with any earlier summary the new one folds in — all in one transaction, so a summary is never saved without its messages going, and vice versa. What is left on disk is exactly what the model sees: one summary row plus the messages after its boundary.

This is deliberate and it is lossy: the summary is a model's paraphrase, the originals are gone, and there is no undo. `messages.db` is the conversation as the bot remembers it, not an archive of what was said.

The summary itself is stored like any other message — same table, same columns — with two extra fields telling it apart: `kind` (`message` or `summary`) and `from_message_id`, the id of the oldest message the summary does *not* cover. Because it is appended, it lands *after* the messages it summarizes, so `load_history` does not read it in id order: it takes the newest `summary` row, then the `message` rows from `from_message_id` on, and returns the summary in front of them.

So a restart resumes the compressed conversation rather than re-reading a raw one, and a second round folds the previous summary into the new one and deletes it, leaving exactly one summary at any time. An older `messages.db` upgrades itself on startup: the two columns are added, any row from the previous one-row `summary` table is moved across, and that table is dropped.

`/clear` and `/reset` drop both the messages and the summary; they remain the only things that remove anything.

## Settings survive a restart

Per-chat settings — compression, mode, temperature, system prompt, max length, stop sequences — live in `chat_data`, which is now pickled to `chat_settings.pkl` via python-telegram-bot's `PicklePersistence`. Before this, every restart quietly reverted a chat to defaults while its history came back from SQLite, so a bot redeployed under systemd kept forgetting how it had been configured.

Only `chat_data` is persisted. The `DeepSeekClient` for a chat is cached in `bot_data` instead, because it owns a SQLite connection that cannot be pickled — and does not need to be, since its history is already durable.

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
- `/compression` / `/compression <0|10|20|50|100>` — view or set history compression (see above; `0` is off)
- `/mode` / `/mode <DIRECT|STEP_BY_STEP|SMART_PROMPT|TEAM>` — view or set how each message is processed:
  - `DIRECT` (default) — forwards your message to DeepSeek as-is
  - `STEP_BY_STEP` — appends "Proceed step by step." for that request only; stored history keeps the original text
  - `SMART_PROMPT` — asks DeepSeek to rewrite your message into a better prompt, shows it to you, then answers with it
  - `TEAM` — an analyst, a scientist, and an engineer each draft an answer, then a critic synthesizes one final reply (4 API calls per message)
- `/broadcast <prompt>` — asks every DeepSeek model the same prompt and returns each reply with its token count and latency
- `/broadcast_and_compare <prompt>` — same, then has the pro model compare the replies as a caveman, an engineer, and a humanitarian
- `/stats` — messages sent, messages in context, model, system prompt, max length, stop sequences, mode, temperature, compression, and the token counts for the last exchange
- `/uptime` — how long the bot process has been running
- `/stop` — shuts the bot down gracefully
- Any other text message is sent to DeepSeek and the reply is sent back, with the last 1000 messages kept as context and persisted to `messages.db`, so a restart keeps the conversation
