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

DEFAULT_BRANCH = "main"

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    message TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    -- Summaries and facts are stored like any other message; `kind` tells them
    -- apart and `from_message_id` records the oldest message a summary does NOT
    -- cover. Everything is scoped to a branch, which is an independent
    -- conversation the user can switch between.
    kind TEXT NOT NULL DEFAULT 'message',
    from_message_id INTEGER,
    branch TEXT NOT NULL DEFAULT 'main'
);
"""


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class MessageStore:
    """SQLite-backed conversation store: every message is saved as it is sent or
    received, and reloaded on startup so a restart keeps its context.

    What is stored is the context, not an archive of the chat. Compression
    deletes the messages it replaces - a summary supersedes them, a window drops
    them - so the rows for a branch stay equal to what the model is given."""

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
            if "branch" not in columns:
                self._connection.execute(
                    f"ALTER TABLE messages ADD COLUMN branch TEXT NOT NULL DEFAULT '{DEFAULT_BRANCH}'"
                )
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

    # ---- writing ----

    def record(self, role: str, message: str, branch: str = DEFAULT_BRANCH) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO messages (direction, message, tokens, branch) VALUES (?, ?, ?, ?)",
                (DIRECTION_BY_ROLE[role], message, count_tokens(message), branch),
            )

    def record_summary(self, message: str, keep_last: int, branch: str = DEFAULT_BRANCH) -> None:
        """Save a summary standing in for every stored message except the most
        recent `keep_last`, and delete what it replaces: the messages it covers
        and any earlier summary, which the new one already folds in.

        The deletion is permanent and the summary is a lossy paraphrase, so what
        those messages said is gone. That is deliberate - the database holds the
        conversation as the model sees it, not an archive - and it is why the
        whole thing happens in one transaction: a summary must never be saved
        without its messages going, nor messages go without the summary saved."""
        from_message_id = self._boundary_id(keep_last, branch)
        with self._connection:
            summary_id = self._connection.execute(
                "INSERT INTO messages (direction, message, tokens, kind, from_message_id, branch) "
                "VALUES (?, ?, ?, 'summary', ?, ?)",
                (DIRECTION_BY_ROLE["assistant"], message, count_tokens(message), from_message_id, branch),
            ).lastrowid
            self._connection.execute(
                "DELETE FROM messages WHERE branch = ? AND kind = 'message' AND id < ?", (branch, from_message_id)
            )
            # Earlier summaries are superseded, and can outlive the boundary:
            # they are appended, so their ids can be higher than from_message_id.
            self._connection.execute(
                "DELETE FROM messages WHERE branch = ? AND kind = 'summary' AND id < ?", (branch, summary_id)
            )

    def record_fact(self, fact: str, branch: str = DEFAULT_BRANCH) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO messages (direction, message, tokens, kind, branch) VALUES (?, ?, ?, 'fact', ?)",
                (DIRECTION_BY_ROLE["assistant"], fact, count_tokens(fact), branch),
            )

    def trim_to_last(self, keep_last: int, branch: str = DEFAULT_BRANCH) -> None:
        """Keep only the most recent `keep_last` messages of a branch, dropping
        the rest for good. Facts are not messages and are never dropped here."""
        with self._connection:
            self._connection.execute(
                "DELETE FROM messages WHERE branch = ? AND kind = 'message' AND id < ?",
                (branch, self._boundary_id(keep_last, branch)),
            )

    def discard_last_incoming(self, branch: str = DEFAULT_BRANCH) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM messages WHERE id = "
                "(SELECT MAX(id) FROM messages WHERE branch = ? AND kind = 'message') AND direction = ?",
                (branch, DIRECTION_BY_ROLE["user"]),
            )

    def clear(self, branch: str = DEFAULT_BRANCH) -> None:
        """Empty one branch - messages, summary and facts alike."""
        with self._connection:
            self._connection.execute("DELETE FROM messages WHERE branch = ?", (branch,))

    # ---- reading ----

    def load_history(self, limit: int, branch: str = DEFAULT_BRANCH) -> list[dict]:
        summary = self._connection.execute(
            "SELECT message, from_message_id FROM messages WHERE branch = ? AND kind = 'summary' "
            "ORDER BY id DESC LIMIT 1",
            (branch,),
        ).fetchone()
        message, from_message_id = summary if summary else (None, 0)
        # The summary occupies one of the slots it saved.
        rows = self._connection.execute(
            "SELECT direction, message FROM messages WHERE branch = ? AND kind = 'message' AND id >= ? "
            "ORDER BY id DESC LIMIT ?",
            (branch, from_message_id, limit - 1 if message else limit),
        ).fetchall()
        history = [{"role": ROLE_BY_DIRECTION[direction], "content": text} for direction, text in reversed(rows)]
        return [{"role": "assistant", "content": message}, *history] if message else history

    def load_facts(self, branch: str = DEFAULT_BRANCH) -> list[str]:
        return [
            row[0]
            for row in self._connection.execute(
                "SELECT message FROM messages WHERE branch = ? AND kind = 'fact' ORDER BY id", (branch,)
            )
        ]

    def list_branches(self) -> list[str]:
        """Every branch that has anything stored, oldest first, with the default
        branch always present so there is somewhere to start."""
        branches = [
            row[0]
            for row in self._connection.execute("SELECT branch, MIN(id) FROM messages GROUP BY branch ORDER BY MIN(id)")
        ]
        return branches if DEFAULT_BRANCH in branches else [DEFAULT_BRANCH, *branches]

    def _boundary_id(self, keep_last: int, branch: str) -> int:
        """The id of the `keep_last`-th message from the end of a branch: the
        oldest one that survives. Everything below it is dropped."""
        oldest_kept = self._connection.execute(
            "SELECT id FROM messages WHERE branch = ? AND kind = 'message' ORDER BY id DESC LIMIT 1 OFFSET ?",
            (branch, max(keep_last - 1, 0)),
        ).fetchone()
        if oldest_kept:
            return oldest_kept[0]
        # Nothing kept means everything stored so far is covered.
        row = self._connection.execute("SELECT MAX(id) FROM messages WHERE branch = ?", (branch,)).fetchone()
        return (row[0] or 0) + 1
