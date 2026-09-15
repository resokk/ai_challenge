from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).with_name("bot.db")

SCHEMA = """
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
