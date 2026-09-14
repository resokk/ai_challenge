# Day 11 — Telegram Bot

A minimal Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible).

Stripped back from [day_10](../day_10): no settings menu, reasoning modes, broadcast, branches, compression or stats. Send a message, get a reply. What it does add is per-user state in SQLite: a `/profile` that becomes your system prompt, and `/tasks` — named threads, each with its own prompt and its own conversation.

## Setup

```bash
cd day_11
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in TELEGRAM_BOT_TOKEN and DEEPSEEK_API_KEY
```

## Run

```bash
python bot.py
```

Stop it with Ctrl+C.

## Usage

- `/start` — greet, and clear the conversation
- `/profile` — show your profile
- `/profile <text>` — set or replace it (max 2000 chars)
- `/tasks` — show the task you are on, and list your tasks
- `/tasks <id>` — switch to that task, starting it if it is new
- `/task` — show the current task's prompt
- `/task <text>` — set or replace it (max 2000 chars)
- `/task none` — leave the current task, keeping it and its prompt
- `/clear` — forget the current conversation, keeping your profile and task prompts
- Any other text message — answered by DeepSeek, with the conversation so far as context

### Profiles and tasks

The system prompt is your profile, then the prompt of the task you are on — who you are, then what you are working on. Either half may be missing, and with neither the bot falls back to its own default prompt.

```
/profile I write Go. Be blunt and skip the preamble.
/tasks api            -> Switched to task api.
/task Refactor the DeepSeek client.
/task none            -> Left task api. Its prompt is kept.
```

Both are per **user** and read from the database on every request: on each message the bot reads your profile and, when your profile records a current task, that task's prompt, and hands the two to the model as the system prompt, so an edit applies to your very next message and everything survives a restart. A task id is 1–32 characters of letters, digits, dashes or underscores.

Each task is also a separate conversation: switching tasks switches thread, and switching back returns to where you left off. `/clear` forgets the messages of the thread you are on and leaves the others alone; your profile and task prompts are in the database, not the history, so they survive it.

## Files

- `bot.py` — Telegram handlers; owns the per-chat history
- `deepseek_client.py` — the DeepSeek chat-completions call
- `storage.py` — `profiles.db`: one row per user, one row per task
- `config.py` — environment variables

Profiles and task prompts are durable. History is not: it is kept in memory only (the newest 100 messages per thread) and is lost when the bot restarts.
