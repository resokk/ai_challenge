"""Vendor property files (db/vendor-properties/<databaseId>.properties).

Camunda feeds these into MyBatis configuration variables (${paging.after}, ${escapeChar}, ...)
and into Liquibase parameters (userCharColumn.size -> ${userCharColumnSize}).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel


class VendorProperties(BaseModel):
    database_id: str
    path: str
    values: dict[str, str]
    lines: dict[str, int]


def parse_java_properties(text: str) -> tuple[dict[str, str], dict[str, int]]:
    values: dict[str, str] = {}
    lines: dict[str, int] = {}
    logical, start = "", 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip() if not logical else raw.lstrip()
        if not logical:
            start = i
            if not line or line[0] in "#!":
                continue
        if line.endswith("\\") and not line.endswith("\\\\"):
            logical += line[:-1]
            continue
        logical += line
        key, value, esc = [], [], False
        target, sep_seen = key, False
        for ch in logical:
            if esc:
                target.append({"t": "\t", "n": "\n", "r": "\r"}.get(ch, ch))
                esc = False
            elif ch == "\\":
                esc = True
            elif not sep_seen and ch in "=:":
                sep_seen, target = True, value
            else:
                target.append(ch)
        k = "".join(key).strip()
        values[k] = "".join(value).strip()
        lines[k] = start
        logical = ""
    return values, lines


def load_vendor_properties(repo: Path, paths: list[str]) -> dict[str, VendorProperties]:
    out: dict[str, VendorProperties] = {}
    for rel in paths:
        if "/src/main/" not in rel:
            continue
        values, lines = parse_java_properties((repo / rel).read_text("utf-8"))
        db_id = PurePosixPath(rel).stem
        out[db_id] = VendorProperties(database_id=db_id, path=rel, values=values, lines=lines)
    return out
