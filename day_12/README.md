# Day 12 — Telegram Bot

A minimal Telegram bot built with [python-telegram-bot](https://docs.python-telegram-bot.org/), with chat replies powered by the [DeepSeek API](https://platform.deepseek.com/) (OpenAI-compatible).

Same as [day_11](../day_11) — a `/profile` that becomes your system prompt, and `/tasks`, named threads each with their own prompt — except that profiles are now **named** too. You can keep several and switch between them with `/profiles`, exactly the way you switch tasks.

## Setup

```bash
cd day_12
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
- `/profiles` — list your profiles, and show which one you are on
- `/profiles <id>` — switch to that profile, starting it if it is new
- `/profile` — show the current profile's description
- `/profile <text>` — set or replace it (max 2000 chars)
- `/profile none` — leave the current profile, keeping it and its description
- `/tasks` — list your tasks, and show which one you are on
- `/tasks <id>` — switch to that task, starting it if it is new
- `/task` — show the current task's prompt
- `/task <text>` — set or replace it (max 2000 chars)
- `/task none` — leave the current task, keeping it and its prompt
- `/clear` — forget the current conversation, keeping your profiles and task prompts
- Any other text message — answered by DeepSeek, with the conversation so far as context

### Profiles and tasks

The system prompt is the description of the profile you are on, then the prompt of the task you are on — who you are, then what you are working on. Either half may be missing, and with neither the bot falls back to its own default prompt.

```
/profiles                -> No profile selected.
/profiles go             -> Switched to profile go. It has no description yet.
/profile I write Go. Be blunt and skip the preamble.
/profiles writing        -> Switched to profile writing.
/profile I'm drafting a newsletter. Suggest edits, don't rewrite.
/profiles                -> Your profiles:  go / -> writing
/tasks api               -> Switched to task api.
/task Refactor the DeepSeek client.
/profile none            -> Left profile writing. Its description is kept.
```

Profiles and tasks are the same shape — a named text you can have several of, plus the one you are currently on — so they behave identically: selecting an id creates it if it is new, leaving one keeps it, and ids are 1–32 characters of letters, digits, dashes or underscores. They are two independent choices: switching profile does not touch your task, and the two lists are ordered by when you created each entry, so editing one never reshuffles the list.

Both are per **user** and read from the database on every request, so an edit applies to your very next message and everything survives a restart.

Each **task** is also a separate conversation: switching tasks switches thread, and switching back returns to where you left off. Profiles say who you are rather than what you are doing, so they share the thread; switching profile keeps the conversation you are in. `/clear` forgets the messages of the thread you are on and leaves the others alone; profiles and task prompts are in the database, not the history, so they survive it.

## Files

- `bot.py` — Telegram handlers; owns the per-chat history
- `deepseek_client.py` — the DeepSeek chat-completions call
- `storage.py` — `bot.db`: one row per user, one row per profile, one row per task
- `config.py` — environment variables

Profiles and task prompts are durable. History is not: it is kept in memory only (the newest 100 messages per thread) and is lost when the bot restarts.
