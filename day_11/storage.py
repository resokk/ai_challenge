from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("profiles.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY,
    profile TEXT NOT NULL DEFAULT '',
    current_task TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    user_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, task_id)
);
"""


class UserStore:
    """SQLite-backed per-user state: a profile, the task the user is currently
    on, and one prompt per task. Everything here is read fresh on each request,
    so an edit applies to the very next message."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Bring a database written before tasks existed up to the current shape."""
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(profiles)")}
        if "current_task" not in columns:
            with self._connection:
                self._connection.execute("ALTER TABLE profiles ADD COLUMN current_task TEXT")

    def _scalar(self, sql: str, parameters: tuple) -> str | None:
        row = self._connection.execute(sql, parameters).fetchone()
        # An empty string is stored where a profile or prompt was never set;
        # callers only care whether there is one, so both arrive as None.
        return row[0] or None if row else None

    # ---- profile ----

    def load_profile(self, user_id: int) -> str | None:
        return self._scalar("SELECT profile FROM profiles WHERE user_id = ?", (user_id,))

    def save_profile(self, user_id: int, profile: str) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO profiles (user_id, profile) VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET profile = excluded.profile,
                                                   updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, profile),
            )

    # ---- tasks ----

    def current_task(self, user_id: int) -> str | None:
        return self._scalar("SELECT current_task FROM profiles WHERE user_id = ?", (user_id,))

    def list_tasks(self, user_id: int) -> list[str]:
        rows = self._connection.execute(
            "SELECT task_id FROM tasks WHERE user_id = ? ORDER BY updated_at", (user_id,)
        )
        return [task_id for (task_id,) in rows]

    def switch_task(self, user_id: int, task_id: str) -> None:
        """Make this the user's current task, starting it if it is new. The
        profile row may not exist yet - selecting a task is enough to create it."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO profiles (user_id, profile, current_task) VALUES (?, '', ?)
                ON CONFLICT(user_id) DO UPDATE SET current_task = excluded.current_task,
                                                   updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, task_id),
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO tasks (user_id, task_id) VALUES (?, ?)", (user_id, task_id)
            )

    def clear_task(self, user_id: int) -> None:
        """Take the user off their current task. The task and its prompt stay,
        so /tasks <id> can come back to it."""
        with self._connection:
            self._connection.execute(
                "UPDATE profiles SET current_task = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,),
            )

    def load_task_prompt(self, user_id: int, task_id: str) -> str | None:
        return self._scalar(
            "SELECT prompt FROM tasks WHERE user_id = ? AND task_id = ?", (user_id, task_id)
        )

    def save_task_prompt(self, user_id: int, task_id: str, prompt: str) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO tasks (user_id, task_id, prompt) VALUES (?, ?, ?)
                ON CONFLICT(user_id, task_id) DO UPDATE SET prompt = excluded.prompt,
                                                            updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, task_id, prompt),
            )
