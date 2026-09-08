from __future__ import annotations

import sqlite3
from pathlib import Path

import tiktoken

DB_PATH = Path(__file__).with_name("messages.db")

# DeepSeek ships its own tokenizer; cl100k_base is close enough for a size
# column but is an estimate, not the exact count DeepSeek bills against.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Chat roles are the client's vocabulary; in/out is how they are stored.
DIRECTION_BY_ROLE = {"user": "in", "assistant": "out"}
ROLE_BY_DIRECTION = {direction: role for role, direction in DIRECTION_BY_ROLE.items()}

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    message TEXT NOT NULL,
    tokens INTEGER NOT NULL
)
"""


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class MessageStore:
    """SQLite-backed conversation history: every message is stored as it is sent
    or received, and reloaded on startup so a restart keeps its context."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.execute(SCHEMA)
        self._connection.commit()

    def record(self, role: str, message: str) -> None:
        self._connection.execute(
            "INSERT INTO messages (direction, message, tokens) VALUES (?, ?, ?)",
            (DIRECTION_BY_ROLE[role], message, count_tokens(message)),
        )
        self._connection.commit()

    def load_history(self, limit: int) -> list[dict]:
        rows = self._connection.execute(
            "SELECT direction, message FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"role": ROLE_BY_DIRECTION[direction], "content": message} for direction, message in reversed(rows)]

    def discard_last_incoming(self) -> None:
        self._connection.execute(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages) AND direction = ?",
            (DIRECTION_BY_ROLE["user"],),
        )
        self._connection.commit()

    def clear(self) -> None:
        self._connection.execute("DELETE FROM messages")
        self._connection.commit()
