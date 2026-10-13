"""Static SQL validation with SQLGlot against the structured catalog (never executes SQL)."""

from __future__ import annotations

import difflib
import re
from typing import Any

import sqlglot
from pydantic import BaseModel, Field
from sqlglot.errors import ParseError, TokenError

from camunda_rag.config import SQLGLOT_DIALECTS, sqlglot_dialect
from camunda_rag.sql_analysis import PARSER, analyze


class Issue(BaseModel):
    severity: str  # error | warning | info
    code: str
    message: str
    line: int | None = None
    column: int | None = None


class StatementResult(BaseModel):
    index: int
    status: str  # valid | invalid | unsupported | parse_error
    statement_type: str | None = None
    normalized_sql: str | None = None
    tables_read: list[str] = Field(default_factory=list)
    tables_written: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class ValidationResult(BaseModel):
    status: str  # valid | invalid | unsupported | parse_error
    dialect: str
    profile_key: str | None
    profile_completeness: str | None
    parser: str = PARSER
    statements: list[StatementResult] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


def _split(sql: str, read: str | None) -> list[str]:
    try:
        trees = [t for t in sqlglot.parse(sql, read=read) if t is not None]
    except (ParseError, TokenError):
        return [sql]
    return [t.sql(dialect=read) for t in trees] if len(trees) > 1 else [sql]


def validate(sql: str, dialect: str, catalog: dict[str, Any]) -> ValidationResult:
    """`catalog`: {"profile_key", "completeness", "tables": {NAME: {COL,...}}, "non_relational": {NAME: storage_kind}}."""
    if dialect not in SQLGLOT_DIALECTS:
        raise ValueError(f"unknown dialect {dialect!r}; known: {sorted(SQLGLOT_DIALECTS)}")
    res = ValidationResult(status="valid", dialect=dialect, profile_key=catalog.get("profile_key"),
                           profile_completeness=catalog.get("completeness"))
    if re.search(r"[#$]\{[^}]*\}", sql):
        res.status = "invalid"
        res.issues.append(Issue(severity="error", code="unrendered_template",
                                message="MyBatis #{}/${} placeholders must be rendered first (use ? for bind parameters)"))
        return res
    read = sqlglot_dialect(dialect) or None
    if read is None:
        res.issues.append(Issue(severity="warning", code="generic_dialect", message=f"SQLGlot has no {dialect} dialect; generic parser used"))
    if catalog.get("completeness") == "partial":
        res.issues.append(Issue(severity="warning", code="partial_schema",
                                message="schema profile is partial (unknown preconditions/effects); absence checks may be wrong"))
    try:
        sqlglot.parse(sql, read=read)
    except ParseError as e:
        res.status = "parse_error"
        for err in e.errors or [{}]:
            res.issues.append(Issue(severity="error", code="parse_error", message=err.get("description") or str(e),
                                    line=err.get("line"), column=err.get("col")))
        return res
    except TokenError as e:
        res.status = "parse_error"
        res.issues.append(Issue(severity="error", code="token_error", message=str(e)))
        return res

    tables: dict[str, set[str]] = catalog.get("tables", {})
    non_rel: dict[str, str] = catalog.get("non_relational", {})
    for i, stmt in enumerate(_split(sql, read)):
        a = analyze(stmt, dialect, tables)
        sr = StatementResult(index=i, status="valid", statement_type=a.statement_type, normalized_sql=a.normalized)
        sr.issues += [Issue(severity="info", code="parser_note", message=d) for d in a.diagnostics]
        if a.status == "unsupported":
            sr.status = "unsupported"
            sr.issues.append(Issue(severity="error", code="unsupported_construct",
                                   message="SQLGlot produced an opaque Command node; statement cannot be validated"))
            res.statements.append(sr)
            continue
        if a.status != "parsed":
            sr.status = "parse_error"
            res.statements.append(sr)
            continue
        created = {t.name.upper() for t in a.tables if t.access == "write" and a.statement_type == "create"}
        scope_tables: set[str] = set()
        for t in a.tables:
            up = t.name.upper()
            (sr.tables_written if t.access == "write" else sr.tables_read).append(t.name)
            if t.via == "cte" or up in created:
                continue
            scope_tables.add(up)
            if up in tables:
                continue
            if up in non_rel:
                sr.issues.append(Issue(severity="error", code="non_relational_entity",
                                       message=f"{t.name} is a {non_rel[up]}, not a SQL table"))
            else:
                near = difflib.get_close_matches(up, list(tables), n=3, cutoff=0.75)
                sr.issues.append(Issue(severity="error", code="unknown_table",
                                       message=f"table {t.name} not in profile" + (f"; did you mean {', '.join(near)}?" if near else "")))
        for c in a.columns:
            col = c.column.upper()
            if c.table:
                tu = c.table.upper()
                if tu in tables and col not in tables[tu]:
                    near = difflib.get_close_matches(col, list(tables[tu]), n=3, cutoff=0.75)
                    sr.issues.append(Issue(severity="error", code="unknown_column",
                                           message=f"column {c.table}.{c.column} not found" + (f"; did you mean {', '.join(near)}?" if near else "")))
                continue
            if c.qualifier:  # qualified by a derived source/CTE: not checked
                continue
            owners = [t for t in scope_tables if t in tables and col in tables[t]]
            known_scope = [t for t in scope_tables if t in tables]
            if not owners and known_scope and len(known_scope) == len(scope_tables):
                select_aliases = {p["output"].upper() for p in a.lineage}
                if col not in select_aliases:
                    sr.issues.append(Issue(severity="error", code="unknown_column",
                                           message=f"column {c.column} not found in {', '.join(sorted(known_scope))}"))
            elif len(owners) > 1:
                sr.issues.append(Issue(severity="warning", code="ambiguous_column",
                                       message=f"unqualified column {c.column} exists in {', '.join(sorted(owners))}"))
        if any(x.severity == "error" for x in sr.issues):
            sr.status = "invalid"
        res.statements.append(sr)
    order = ["parse_error", "unsupported", "invalid", "valid"]
    res.status = next((s for s in order if any(st.status == s for st in res.statements)), "valid")
    return res
