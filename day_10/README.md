# Day 10 — Telegram Bot

Same base as [day_9](../day_9): a Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible), per-chat settings behind a button menu, reasoning modes, multi-model broadcast, SQLite-persisted history, per-exchange token accounting, and history compression. This version adds `/compression_mode`: compression is no longer one fixed strategy but **four** — summarize, window, branch, or facts — and with them `/branch`, which turns one chat into several independent conversations.

## Setup

```bash
cd day_10
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

`/compression` sets *when* a conversation is compressed — `0` (off), `10`, `20`, `50` or `100` messages. `/compression_mode` sets *what happens* when it reaches that limit. Every mode runs after your reply has been sent, so compression never adds latency to an answer, and a mode that fails is logged and skipped: the conversation stays as it is until the next message. Each round reports what it cost and saved:

```
SUMMARIZE: folded the oldest messages into a summary
Took 6 messages out of the context (5 left).
History tokens: 1280 before, 533 after.
```

Both token figures are the whole context counted the way `/stats` counts it. Compression usually shrinks it sharply, but it is not guaranteed to — a summary of a handful of very short messages can be longer than the messages themselves — which is why the bot reports both numbers rather than a promise.

### SUMMARIZE (default)

The oldest half is summarized and the summary takes its place, at a fixed low temperature (0.2), since a summary wants fidelity rather than variety. The split lands on the next user turn rather than exactly halfway, so the summary — an assistant note — still alternates with what follows: a 10-message conversation folds 6 and keeps 4. Each round folds the previous summary into the new one, so there is only ever one.

Costs one extra API call per round, and is lossy: the messages it covers are deleted.

### WINDOW

Keep the newest N messages, forget the rest. No model call, no summary, nothing to go wrong — and nothing carried forward either. The cheapest mode, and the only instant one.

### BRANCH

Nothing is summarized and nothing is deleted. The conversation is left where it is and the bot continues in a new branch — `main` becomes `main-2` — so the context starts empty while the full conversation stays under its old name, one `/branch` away.

### FACTS

After every exchange the bot extracts durable facts from it — names, preferences, decisions, numbers, constraints — and keeps them. The messages are then windowed to the newest N exactly like WINDOW, but what they established is not lost: the facts are sent ahead of the messages in every request, so the model still knows them long after the conversation that produced them is gone. Facts are deduplicated, so repeating yourself does not grow the list.

The most expensive mode — one extra API call per *message*, not per round. `/stats` shows how many facts are known.

## Branches

A branch is an independent conversation: its own messages, its own summary, its own facts. Nothing is shared or copied between them.

- `/branch` — list your branches and see which one you are on
- `/branch <name>` — switch to a branch, or start it if the name is new

Names are limited to 32 characters and cannot contain `:`, since they travel in Telegram's callback data. `/clear` empties the branch you are on; `/reset` empties it, reverts every setting, and puts you back on `main`.

## What compression does to the database

SUMMARIZE and WINDOW shrink the **stored conversation as well as the context**: the messages they replace are deleted, in one transaction with the summary that supersedes them, so a summary is never saved without its messages going and vice versa. What is left on disk for a branch is exactly what the model sees.

This is deliberate and lossy: a summary is a model's paraphrase, the originals are gone, and there is no undo. `messages.db` is the conversation as the bot remembers it, not an archive of what was said. BRANCH is the exception — it deletes nothing, it just stops adding to one branch and starts another.

Summaries and facts are stored like any other message — same table, same columns — with three extra fields: `kind` (`message`, `summary` or `fact`), `from_message_id` (the oldest message a summary does *not* cover), and `branch`, which scopes every row to one conversation. Because a summary is appended it lands *after* the messages it covers, so `load_history` does not read in id order: it takes the newest `summary` row of the branch, then that branch's `message` rows from `from_message_id` on, and returns the summary in front of them.

A restart therefore resumes each branch exactly as it was — compressed conversation, facts and all. An older `messages.db` upgrades itself on startup: missing columns are added, any row from the one-row `summary` table day_9 briefly used is moved across, and that table is dropped.

## Settings survive a restart

Per-chat settings — compression and its mode, current branch, reasoning mode, temperature, system prompt, max length, stop sequences — live in `chat_data`, which is now pickled to `chat_settings.pkl` via python-telegram-bot's `PicklePersistence`. Before this, every restart quietly reverted a chat to defaults while its history came back from SQLite, so a bot redeployed under systemd kept forgetting how it had been configured.

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
- `/compression` / `/compression <0|10|20|50|100>` — view or set the message limit at which the conversation is compressed (`0` is off)
- `/compression_mode` / `/compression_mode <SUMMARIZE|WINDOW|BRANCH|FACTS>` — view or set how it is compressed when that limit is reached
- `/branch` / `/branch <name>` — view your branches, or switch to (or start) one
- `/mode` / `/mode <DIRECT|STEP_BY_STEP|SMART_PROMPT|TEAM>` — view or set how each message is processed:
  - `DIRECT` (default) — forwards your message to DeepSeek as-is
  - `STEP_BY_STEP` — appends "Proceed step by step." for that request only; stored history keeps the original text
  - `SMART_PROMPT` — asks DeepSeek to rewrite your message into a better prompt, shows it to you, then answers with it
  - `TEAM` — an analyst, a scientist, and an engineer each draft an answer, then a critic synthesizes one final reply (4 API calls per message)
- `/broadcast <prompt>` — asks every DeepSeek model the same prompt and returns each reply with its token count and latency
- `/broadcast_and_compare <prompt>` — same, then has the pro model compare the replies as a caveman, an engineer, and a humanitarian
- `/stats` — messages sent, messages in context, model, system prompt, max length, stop sequences, mode, temperature, branch, compression and its mode, and the token counts for the last exchange
- `/uptime` — how long the bot process has been running
- `/stop` — shuts the bot down gracefully
- Any other text message is sent to DeepSeek and the reply is sent back, with the last 1000 messages kept as context and persisted to `messages.db`, so a restart keeps the conversation
