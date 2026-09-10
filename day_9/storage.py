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
    tokens INTEGER NOT NULL,
    -- A summary is logged like any other message; `kind` tells them apart and
    -- `from_message_id` records the oldest message the summary does NOT cover.
    kind TEXT NOT NULL DEFAULT 'message',
    from_message_id INTEGER
);
"""


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class MessageStore:
    """SQLite-backed conversation store: every message is saved as it is sent or
    received, and reloaded on startup so a restart keeps its context.

    What is stored is the context, not an archive of the chat. Compression
    deletes the messages it summarizes (see record_summary), so the rows on disk
    stay equal to what the model is given: at most one summary, plus the
    messages after it."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Bring an older database up to the current shape, in place."""
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(messages)")}
        with self._connection:
            if "kind" not in columns:
                self._connection.execute("ALTER TABLE messages ADD COLUMN kind TEXT NOT NULL DEFAULT 'message'")
                self._connection.execute("ALTER TABLE messages ADD COLUMN from_message_id INTEGER")
            # Summaries used to live in their own table; move them into the log.
            if self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'summary'"
            ).fetchone():
                rows = self._connection.execute("SELECT message, from_message_id FROM summary ORDER BY id").fetchall()
                self._connection.executemany(
                    "INSERT INTO messages (direction, message, tokens, kind, from_message_id) "
                    "VALUES (?, ?, ?, 'summary', ?)",
                    [
                        (DIRECTION_BY_ROLE["assistant"], message, count_tokens(message), from_message_id)
                        for message, from_message_id in rows
                    ],
                )
                self._connection.execute("DROP TABLE summary")

    def record(self, role: str, message: str) -> None:
        self._connection.execute(
            "INSERT INTO messages (direction, message, tokens) VALUES (?, ?, ?)",
            (DIRECTION_BY_ROLE[role], message, count_tokens(message)),
        )
        self._connection.commit()

    def record_summary(self, message: str, keep_last: int) -> None:
        """Save a summary standing in for every stored message except the most
        recent `keep_last`, and delete what it replaces: the messages it covers
        and any earlier summary, which the new one already folds in.

        The deletion is permanent and the summary is a lossy paraphrase, so what
        those messages said is gone. That is deliberate - the database holds the
        conversation as the model sees it, not as an archive - and it is why the
        whole thing happens in one transaction: a summary must never be saved
        without its messages going, nor messages go without the summary saved."""
        oldest_kept = self._connection.execute(
            "SELECT id FROM messages WHERE kind = 'message' ORDER BY id DESC LIMIT 1 OFFSET ?", (keep_last - 1,)
        ).fetchone()
        # Nothing kept means the summary covers everything stored so far.
        from_message_id = oldest_kept[0] if oldest_kept else self._next_message_id()
        with self._connection:
            summary_id = self._connection.execute(
                "INSERT INTO messages (direction, message, tokens, kind, from_message_id) "
                "VALUES (?, ?, ?, 'summary', ?)",
                (DIRECTION_BY_ROLE["assistant"], message, count_tokens(message), from_message_id),
            ).lastrowid
            self._connection.execute("DELETE FROM messages WHERE kind = 'message' AND id < ?", (from_message_id,))
            # Earlier summaries are superseded, and can outlive the boundary:
            # they are appended, so their ids can be higher than from_message_id.
            self._connection.execute("DELETE FROM messages WHERE kind = 'summary' AND id < ?", (summary_id,))

    def _next_message_id(self) -> int:
        row = self._connection.execute("SELECT MAX(id) FROM messages").fetchone()
        return (row[0] or 0) + 1

    def load_history(self, limit: int) -> list[dict]:
        summary = self._connection.execute(
            "SELECT message, from_message_id FROM messages WHERE kind = 'summary' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        message, from_message_id = summary if summary else (None, 0)
        # The summary occupies one of the slots it saved.
        rows = self._connection.execute(
            "SELECT direction, message FROM messages WHERE kind = 'message' AND id >= ? ORDER BY id DESC LIMIT ?",
            (from_message_id, limit - 1 if message else limit),
        ).fetchall()
        history = [{"role": ROLE_BY_DIRECTION[direction], "content": text} for direction, text in reversed(rows)]
        return [{"role": "assistant", "content": message}, *history] if message else history

    def discard_last_incoming(self) -> None:
        self._connection.execute(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages WHERE kind = 'message') "
            "AND direction = ?",
            (DIRECTION_BY_ROLE["user"],),
        )
        self._connection.commit()

    def clear(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM messages")
