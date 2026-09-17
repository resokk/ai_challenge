from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).with_name("bot.db")

# The run a task normally makes, in order. The first is where every task starts,
# so a task that has never been moved reads as PENDING without anything written.
TASK_STATE_SEQUENCE = ("PENDING", "IN_WORK", "REVIEW", "DONE")

# The free text a task carries besides its prompt: what is left to do on it, and
# the rules for working on it. They are written and read the same way, so one
# pair of queries serves both and the names double as the commands that set them.
TASK_TEXT_FIELDS = ("todo", "rules")
# CANCELED is a state a task can be in but not a step in the run - a canceled
# task has stopped rather than advanced - so it is outside the sequence.
TASK_STATES = (*TASK_STATE_SEQUENCE, "CANCELED")
DEFAULT_TASK_STATE = TASK_STATE_SEQUENCE[0]


def next_state(state: str) -> str | None:
    """The state after `state` in the usual run, or None where there is no next
    one: the end of the run, or a state outside it."""
    if state not in TASK_STATE_SEQUENCE:
        return None
    following = TASK_STATE_SEQUENCE.index(state) + 1
    return TASK_STATE_SEQUENCE[following] if following < len(TASK_STATE_SEQUENCE) else None

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    current_profile TEXT,
    current_task TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER NOT NULL,
    profile_id TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, profile_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    user_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '{DEFAULT_TASK_STATE}',
    summary TEXT NOT NULL DEFAULT '',
    todo TEXT NOT NULL DEFAULT '',
    rules TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, task_id)
);
"""


@dataclass(frozen=True)
class Collection:
    """One named-text collection: its own table of per-user items, and the
    column on `users` recording which item that user is currently on.

    Profiles and tasks are the same shape - who the user is, and what they are
    working on - so both are described here and share one set of queries. The
    nouns are how the bot names the collection and an item's text to the user;
    the rest are the SQL identifiers, and are never user input."""

    noun: str
    text_noun: str
    table: str
    id_column: str
    pointer: str


PROFILES = Collection("profile", "description", "profiles", "profile_id", "current_profile")
TASKS = Collection("task", "prompt", "tasks", "task_id", "current_task")


class UserStore:
    """SQLite-backed per-user state: the profiles and tasks a user has, and
    which of each they are on. Everything here is read fresh on each request,
    so an edit applies to the very next message."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def _scalar(self, sql: str, parameters: tuple) -> str | None:
        row = self._connection.execute(sql, parameters).fetchone()
        # An empty string is stored where an item's text was never set; callers
        # only care whether there is one, so both arrive as None.
        return row[0] or None if row else None

    def current(self, collection: Collection, user_id: int) -> str | None:
        """The id of the item this user is on, or None if they are on none."""
        return self._scalar(
            f"SELECT {collection.pointer} FROM users WHERE user_id = ?", (user_id,)
        )

    def list_ids(self, collection: Collection, user_id: int) -> list[str]:
        """This user's item ids, oldest first. Creation order, not edit order,
        so the list a user picks from does not reshuffle when they edit one."""
        rows = self._connection.execute(
            f"SELECT {collection.id_column} FROM {collection.table} WHERE user_id = ? ORDER BY rowid",
            (user_id,),
        )
        return [item_id for (item_id,) in rows]

    def select(self, collection: Collection, user_id: int, item_id: str) -> None:
        """Make this the user's current item, creating it if it is new. The
        users row may not exist yet - selecting is enough to create it."""
        with self._connection:
            self._connection.execute(
                f"""
                INSERT INTO users (user_id, {collection.pointer}) VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET {collection.pointer} = excluded.{collection.pointer},
                                                   updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, item_id),
            )
            self._connection.execute(
                f"INSERT OR IGNORE INTO {collection.table} (user_id, {collection.id_column}) VALUES (?, ?)",
                (user_id, item_id),
            )

    def deselect(self, collection: Collection, user_id: int) -> None:
        """Take the user off their current item. The item and its text stay, so
        selecting it by id comes back to it."""
        with self._connection:
            self._connection.execute(
                f"UPDATE users SET {collection.pointer} = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,),
            )

    def delete(self, collection: Collection, user_id: int, item_id: str) -> bool:
        """Remove an item and everything stored on it. If the user is on it they
        come off it in the same transaction - a pointer to something that is no
        longer there is worse than no pointer. Says whether there was one."""
        with self._connection:
            deleted = self._connection.execute(
                f"DELETE FROM {collection.table} WHERE user_id = ? AND {collection.id_column} = ?",
                (user_id, item_id),
            ).rowcount
            self._connection.execute(
                f"""
                UPDATE users SET {collection.pointer} = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND {collection.pointer} = ?
                """,
                (user_id, item_id),
            )
        return deleted > 0

    def load_text(self, collection: Collection, user_id: int, item_id: str) -> str | None:
        return self._scalar(
            f"SELECT text FROM {collection.table} WHERE user_id = ? AND {collection.id_column} = ?",
            (user_id, item_id),
        )

    def save_text(self, collection: Collection, user_id: int, item_id: str, text: str) -> None:
        with self._connection:
            self._connection.execute(
                f"""
                INSERT INTO {collection.table} (user_id, {collection.id_column}, text) VALUES (?, ?, ?)
                ON CONFLICT(user_id, {collection.id_column}) DO UPDATE SET text = excluded.text,
                                                                           updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, item_id, text),
            )

    def _task_column(self, column: str, user_id: int, task_id: str) -> str | None:
        """Read one per-task column. Only tasks have these: a profile is who the
        user is, and neither moves through states nor holds a conversation."""
        return self._scalar(
            f"SELECT {column} FROM {TASKS.table} WHERE user_id = ? AND {TASKS.id_column} = ?",
            (user_id, task_id),
        )

    def _set_task_column(self, column: str, user_id: int, task_id: str, value: str) -> None:
        """Write one per-task column. `column` is a literal from this module,
        never user input - only `value` is bound."""
        with self._connection:
            self._connection.execute(
                f"""
                INSERT INTO {TASKS.table} (user_id, {TASKS.id_column}, {column}) VALUES (?, ?, ?)
                ON CONFLICT(user_id, {TASKS.id_column}) DO UPDATE SET {column} = excluded.{column},
                                                                      updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, task_id, value),
            )

    def load_state(self, user_id: int, task_id: str) -> str:
        """Where this task is. A task that has never been moved is PENDING."""
        return self._task_column("state", user_id, task_id) or DEFAULT_TASK_STATE

    def save_state(self, user_id: int, task_id: str, state: str) -> None:
        """Move a task to `state`, which the caller has checked is in TASK_STATES."""
        self._set_task_column("state", user_id, task_id, state)

    def load_task_text(self, field: str, user_id: int, task_id: str) -> str | None:
        """One of a task's TASK_TEXT_FIELDS, or None where it was never set."""
        return self._task_column(field, user_id, task_id)

    def save_task_text(self, field: str, user_id: int, task_id: str, text: str) -> None:
        """Replace one of a task's TASK_TEXT_FIELDS. An empty string is how a
        field is left holding nothing."""
        self._set_task_column(field, user_id, task_id, text)

    def load_summary(self, user_id: int, task_id: str) -> str | None:
        """The conversation this task was left with, or None if it has none."""
        return self._task_column("summary", user_id, task_id)

    def save_summary(self, user_id: int, task_id: str, summary: str) -> None:
        """Leave a task carrying `summary`, replacing whatever it carried before.
        An empty string is how a task is left carrying nothing."""
        self._set_task_column("summary", user_id, task_id, summary)
