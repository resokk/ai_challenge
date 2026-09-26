from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

# Beside the code by default, so the server finds its data wherever it is
# started from. A deployment points SENSOR_LOG_DB elsewhere so that the
# service can write its data without being able to write its own code.
DB_PATH = Path(os.environ.get("SENSOR_LOG_DB") or Path(__file__).with_name("sensor_log.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id           INTEGER PRIMARY KEY,
    sensor_id    TEXT NOT NULL REFERENCES sensors(id),
    calling_time TEXT NOT NULL,
    sensing_time TEXT NOT NULL,
    temperature  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_by_sensor ON samples(sensor_id, id);
"""


class StorageError(Exception):
    """Input the caller got wrong and can fix."""


def _connect() -> sqlite3.Connection:
    # A connection per call: calls are rare and short, and nothing is shared
    # between threads. The context manager commits or rolls back; closing is
    # separate, hence the explicit close() in each caller.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _timestamp(field: str, value: str) -> str:
    # ISO 8601 in, ISO 8601 out: stored text stays sortable and readable, and
    # a malformed time is refused rather than kept as a string nobody can parse.
    try:
        return datetime.fromisoformat(value.strip()).isoformat()
    except ValueError:
        raise StorageError(f"{field} must be an ISO 8601 time like 2026-09-26T12:00:00+03:00, got {value!r}") from None


def _required(field: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise StorageError(f"{field} must not be empty")
    return value


def save_sample(sensor_name: str, sensor_id: str, calling_time: str, sensing_time: str, temperature: float) -> int:
    sensor_id = _required("sensor_id", sensor_id)
    sensor_name = _required("sensor_name", sensor_name)
    if not math.isfinite(temperature):
        raise StorageError("temperature must be a finite number")
    calling_time = _timestamp("calling_time", calling_time)
    sensing_time = _timestamp("sensing_time", sensing_time)
    conn = _connect()
    try:
        with conn:
            # The id identifies the sensor; the name is a label that may be
            # renamed, so the latest one wins.
            conn.execute(
                "INSERT INTO sensors (id, name) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name",
                (sensor_id, sensor_name),
            )
            cursor = conn.execute(
                "INSERT INTO samples (sensor_id, calling_time, sensing_time, temperature) VALUES (?, ?, ?, ?)",
                (sensor_id, calling_time, sensing_time, temperature),
            )
        return cursor.lastrowid
    finally:
        conn.close()


def list_sensors() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute("SELECT id, name FROM sensors ORDER BY name, id")]
    finally:
        conn.close()


def last_samples(sensor_id: str, n: int) -> list[dict[str, Any]]:
    if n < 1:
        raise StorageError("n must be at least 1")
    conn = _connect()
    try:
        if conn.execute("SELECT 1 FROM sensors WHERE id = ?", (sensor_id,)).fetchone() is None:
            raise StorageError(f"No sensor with id {sensor_id!r}; list_sensors shows the known ones")
        # "Last" means last saved: ids grow with every insert, whereas times
        # sent with different UTC offsets do not sort correctly as text.
        rows = conn.execute(
            "SELECT calling_time, sensing_time, temperature FROM samples WHERE sensor_id = ? ORDER BY id DESC LIMIT ?",
            (sensor_id, n),
        )
        return [dict(row) for row in rows]
    finally:
        conn.close()
