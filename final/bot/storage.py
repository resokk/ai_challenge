"""SQLite storage: users, their tasks and their chat history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL UNIQUE,
    username TEXT,
    first_name TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'blocked')),
    agents_md TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    telegram_id INTEGER NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    repo_url TEXT NOT NULL,
    memory TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (telegram_id, task_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'summary')),
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_user ON messages (telegram_id, id);
"""

# Columns added to `users` after the first version of the schema: an older
# database gets them on open.
USER_MIGRATIONS = {
    "current_task_id": "TEXT",
    "last_tokens": "INTEGER NOT NULL DEFAULT 0",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(users)")}
        for name, kind in USER_MIGRATIONS.items():
            if name not in columns:
                self.db.execute(f"ALTER TABLE users ADD COLUMN {name} {kind}")

    def close(self) -> None:
        self.db.close()

    # users

    def register(self, telegram_id: int, username: str | None, first_name: str | None) -> tuple[sqlite3.Row, bool]:
        """The user's row, and whether this call created it."""
        now = _now()
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO users (telegram_id, username, first_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (telegram_id, username, first_name, now, now),
        )
        return self.get_user(telegram_id), cursor.rowcount == 1

    def get_user(self, telegram_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)).fetchone()

    def list_users(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.db.execute("SELECT * FROM users WHERE status = ? ORDER BY id", (status,)).fetchall()
        return self.db.execute("SELECT * FROM users ORDER BY id").fetchall()

    def _update_user(self, telegram_id: int, **fields) -> None:
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self.db.execute(
            f"UPDATE users SET {assignments}, updated_at = ? WHERE telegram_id = ?",
            (*fields.values(), _now(), telegram_id),
        )

    def set_status(self, telegram_id: int, status: str) -> None:
        self._update_user(telegram_id, status=status)

    def set_agents_md(self, telegram_id: int, text: str | None) -> None:
        self._update_user(telegram_id, agents_md=text)

    def set_last_tokens(self, telegram_id: int, tokens: int) -> None:
        self.db.execute("UPDATE users SET last_tokens = ? WHERE telegram_id = ?", (tokens, telegram_id))

    def delete_user(self, telegram_id: int) -> None:
        with self.db:
            self.db.execute("BEGIN")
            self.db.execute("DELETE FROM messages WHERE telegram_id = ?", (telegram_id,))
            self.db.execute("DELETE FROM tasks WHERE telegram_id = ?", (telegram_id,))
            self.db.execute("DELETE FROM users WHERE telegram_id = ?", (telegram_id,))

    # tasks

    def list_tasks(self, telegram_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM tasks WHERE telegram_id = ? ORDER BY task_id", (telegram_id,)
        ).fetchall()

    def get_task(self, telegram_id: int, task_id: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM tasks WHERE telegram_id = ? AND task_id = ?", (telegram_id, task_id)
        ).fetchone()

    def current_task(self, telegram_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT tasks.* FROM tasks JOIN users USING (telegram_id) "
            "WHERE telegram_id = ? AND task_id = users.current_task_id",
            (telegram_id,),
        ).fetchone()

    def upsert_task(self, telegram_id: int, task_id: str, repo_url: str) -> None:
        now = _now()
        self.db.execute(
            "INSERT INTO tasks (telegram_id, task_id, repo_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (telegram_id, task_id) DO UPDATE SET repo_url = excluded.repo_url, "
            "updated_at = excluded.updated_at",
            (telegram_id, task_id, repo_url, now, now),
        )

    def set_current_task(self, telegram_id: int, task_id: str | None) -> None:
        self._update_user(telegram_id, current_task_id=task_id)

    def set_memory(self, telegram_id: int, task_id: str, memory: str | None) -> None:
        self.db.execute(
            "UPDATE tasks SET memory = ?, updated_at = ? WHERE telegram_id = ? AND task_id = ?",
            (memory, _now(), telegram_id, task_id),
        )

    # messages

    def add_message(
        self,
        telegram_id: int,
        role: str,
        content: str | None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> int:
        cursor = self.db.execute(
            "INSERT INTO messages (telegram_id, role, content, tool_calls, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                telegram_id,
                role,
                content,
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                tool_call_id,
                _now(),
            ),
        )
        return cursor.lastrowid

    def history(self, telegram_id: int) -> list[sqlite3.Row]:
        """The user's chat history, oldest first. A summary, when there is one,
        is always the first row: it replaced everything before it."""
        return self.db.execute(
            "SELECT * FROM messages WHERE telegram_id = ? ORDER BY id", (telegram_id,)
        ).fetchall()

    def clear_history(self, telegram_id: int) -> None:
        self.db.execute("DELETE FROM messages WHERE telegram_id = ?", (telegram_id,))

    def replace_with_summary(self, telegram_id: int, summary: str, covered_ids: list[int]) -> None:
        """Swap the covered rows - old summary included - for one summary row,
        in one transaction. The summary takes the id of the oldest row it
        replaces so it stays first in history order."""
        with self.db:
            self.db.execute("BEGIN")
            placeholders = ", ".join("?" * len(covered_ids))
            self.db.execute(
                f"DELETE FROM messages WHERE telegram_id = ? AND id IN ({placeholders})",
                (telegram_id, *covered_ids),
            )
            self.db.execute(
                "INSERT INTO messages (id, telegram_id, role, content, created_at) VALUES (?, ?, 'summary', ?, ?)",
                (min(covered_ids), telegram_id, summary, _now()),
            )
