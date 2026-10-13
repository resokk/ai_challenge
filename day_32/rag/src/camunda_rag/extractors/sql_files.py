"""Standalone .sql files: dialect from file name/path, statements split by SQLGlot."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import sqlglot
from sqlglot.errors import ParseError, TokenError

from camunda_rag.config import SQLGLOT_DIALECTS, sqlglot_dialect
from camunda_rag.models import ChangeRecord, Diagnostic, Provenance, QueryDef, QueryVariant

EXTRACTOR = "sql-file"


def dialect_of(rel: str) -> tuple[str | None, str]:
    stem = PurePosixPath(rel).name.lower().rsplit(".", 1)[0]
    for part in reversed(re.split(r"[-_.]", stem)):
        if part in SQLGLOT_DIALECTS:
            return part, f"file name token {part!r}"
    for seg in reversed(PurePosixPath(rel.lower()).parts[:-1]):
        if seg in SQLGLOT_DIALECTS:
            return seg, f"directory {seg}/"
    return None, "no dialect evidence in path"


def _statements(text: str) -> list[tuple[str, int]]:
    """Split at top-level semicolons (outside quotes / comments), keeping start lines."""
    out, buf, line, start = [], [], 1, 1
    i, q = 0, None
    while i < len(text):
        ch = text[i]
        if q:
            buf.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            buf.append(ch)
        elif text.startswith("--", i):
            j = text.find("\n", i)
            j = len(text) if j < 0 else j
            buf.append(text[i:j])
            i = j
            continue
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append((stmt, start))
            buf = []
            start = line
        else:
            if not buf or not "".join(buf).strip():
                start = line
            buf.append(ch)
        if ch == "\n":
            line += 1
        i += 1
    stmt = "".join(buf).strip()
    if stmt:
        out.append((stmt, start))
    return out


def extract(repo: Path, rel: str, is_test: bool, default_dialect: str) -> tuple[list[QueryDef], list[ChangeRecord], list[Diagnostic]]:
    text = (repo / rel).read_text("utf-8", "replace")
    dialect, evidence = dialect_of(rel)
    diags: list[Diagnostic] = []
    if dialect is None:
        dialect = default_dialect
        evidence += f"; default dialect {default_dialect} assumed"
        diags.append(Diagnostic(path=rel, extractor=EXTRACTOR, severity="info", code="dialect_assumed", message=evidence))
    queries, changes = [], []
    for n, (stmt, line) in enumerate(_statements(text), 1):
        body = re.sub(r"^\s*(--[^\n]*\n\s*)+", "", stmt)
        if not body.strip():
            continue
        end = line + stmt.count("\n")
        prov = Provenance(path=rel, line_start=line, line_end=end, symbol=f"statement {n}", extractor=EXTRACTOR, evidence_kind="sql_file_statement")
        kind = body.split(None, 1)[0].lower()
        queries.append(QueryDef(key=f"sql:{rel}:{line}", language="sql", origin="sql_file", statement_kind=kind, raw_text=stmt,
                                is_test=is_test, provenance=prov,
                                variants=[QueryVariant(dialect=dialect, dialect_evidence=evidence, sql=body)]))
        if kind in {"create", "alter", "drop"}:
            changes.append(ChangeRecord(order=n, component=f"sqlfile:{rel}", changelog_path=rel, changeset_id=f"statement-{n}",
                                        change_type="sql", details={"sql": body}, provenance=prov))
    try:
        sqlglot.parse(text, read=sqlglot_dialect(dialect) or None)
    except (ParseError, TokenError) as e:
        diags.append(Diagnostic(path=rel, extractor=EXTRACTOR, code="file_parse_warning", message=str(e)[:500]))
    return queries, changes, diags
