"""SQLGlot-based analysis of a single SQL text in a concrete dialect.

Produces AST (serialized), normalized SQL, fingerprint, table/column references with clause,
joins, read/write sets, parameters and per-projection column lineage. A statement that SQLGlot
only recognises as an opaque `Command` is reported as `unsupported`, never as parsed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope

from camunda_rag.config import sqlglot_dialect

PARSER = f"sqlglot {sqlglot.__version__}"

WRITE_TYPES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.TruncateTable)


class TableRef(BaseModel):
    name: str
    schema_name: str | None = None
    alias: str | None = None
    access: str  # read | write
    via: str | None = None  # cte / subquery when the name is a derived source


class ColumnRef(BaseModel):
    table: str | None  # resolved base table; None when ambiguous/unknown
    column: str
    clause: str  # select | where | join | group | order | having | set | insert | other
    qualifier: str | None = None
    resolved: bool = True


class JoinRef(BaseModel):
    kind: str
    left_table: str | None
    left_column: str | None
    right_table: str | None
    right_column: str | None
    condition: str


class SqlAnalysis(BaseModel):
    status: str  # parsed | error | unsupported
    dialect: str
    parser: str = PARSER
    statement_type: str | None = None
    normalized: str | None = None
    fingerprint: str | None = None
    ast_json: str | None = None
    tables: list[TableRef] = Field(default_factory=list)
    columns: list[ColumnRef] = Field(default_factory=list)
    joins: list[JoinRef] = Field(default_factory=list)
    ctes: list[str] = Field(default_factory=list)
    subqueries: int = 0
    placeholders: int = 0
    named_parameters: list[str] = Field(default_factory=list)
    aggregates: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    lineage: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


def _clause_of(node: exp.Expression) -> str:
    cur = node.parent
    while cur is not None:
        if isinstance(cur, exp.Where):
            return "where"
        if isinstance(cur, exp.Join):
            return "join"
        if isinstance(cur, exp.Group):
            return "group"
        if isinstance(cur, exp.Order):
            return "order"
        if isinstance(cur, exp.Having):
            return "having"
        if isinstance(cur, exp.Update) or (isinstance(cur, exp.EQ) and isinstance(cur.parent, exp.Update)):
            return "set"
        if isinstance(cur, exp.Schema) and isinstance(cur.parent, exp.Insert):
            return "insert"
        if isinstance(cur, exp.Select):
            return "select"
        cur = cur.parent
    return "other"


def _fingerprint(tree: exp.Expression, dialect: str) -> str:
    t = tree.copy()
    for lit in list(t.find_all(exp.Literal)):
        lit.replace(exp.Placeholder())
    norm = t.sql(dialect=dialect, normalize=True, comments=False)
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _ast_json(tree: exp.Expression) -> str:
    def dump(node: Any) -> Any:
        if isinstance(node, exp.Expression):
            out: dict[str, Any] = {"t": node.key}
            for k, v in node.args.items():
                if v is None or v is False or v == []:
                    continue
                out[k] = [dump(x) for x in v] if isinstance(v, list) else dump(v)
            return out
        return node if isinstance(node, (str, int, float, bool)) else str(node)

    return json.dumps(dump(tree), separators=(",", ":"))


def _base_table(scope: Scope, qualifier: str | None, column: str, schema: dict[str, set[str]] | None) -> tuple[str | None, bool]:
    """Resolve a column qualifier (alias) to a base table name within the scope."""
    tables = {alias: src for alias, src in scope.sources.items() if isinstance(src, exp.Table)}
    if qualifier:
        src = scope.sources.get(qualifier)
        if isinstance(src, exp.Table):
            return src.name, True
        if isinstance(src, Scope):
            return None, False  # derived table / CTE column: lineage not followed here
        return None, False
    if len(scope.sources) == 1 and len(tables) == 1:
        return next(iter(tables.values())).name, True
    if schema:
        owners = [t.name for t in tables.values() if column.upper() in schema.get(t.name.upper(), set())]
        if len(owners) == 1:
            return owners[0], True
    return None, False


def analyze(sql: str, dialect: str, schema: dict[str, set[str]] | None = None) -> SqlAnalysis:
    """Analyse one SQL statement. `schema` (TABLE -> {COLUMNS}, upper-case) helps unqualified columns."""
    read = sqlglot_dialect(dialect)
    res = SqlAnalysis(status="error", dialect=dialect)
    if not read:
        res.diagnostics.append(f"no SQLGlot dialect for {dialect!r}; parsed with generic SQLGlot dialect")
    try:
        trees = [t for t in sqlglot.parse(sql, read=read or None) if t is not None]
    except (ParseError, TokenError) as e:
        res.diagnostics.append(f"parse error: {e}"[:2000])
        return res
    if len(trees) != 1:
        res.diagnostics.append(f"expected one statement, got {len(trees)}")
        if not trees:
            return res
    tree = trees[0]
    res.statement_type = tree.key
    if isinstance(tree, exp.Command) or tree.find(exp.Command):
        res.status = "unsupported"
        res.diagnostics.append(f"SQLGlot fell back to an opaque Command node for {tree.key!r}; not a full AST")
        return res
    res.status = "parsed"
    out_dialect = read or None
    res.normalized = tree.sql(dialect=out_dialect, comments=False)
    res.fingerprint = _fingerprint(tree, out_dialect or "")
    res.ast_json = _ast_json(tree)
    res.placeholders = len(list(tree.find_all(exp.Placeholder)))
    res.named_parameters = sorted({p.name for p in tree.find_all(exp.Placeholder) if p.name} |
                                  {p.name for p in tree.find_all(exp.Parameter)})
    res.ctes = [c.alias for c in tree.find_all(exp.CTE)]
    res.subqueries = len(list(tree.find_all(exp.Subquery)))

    write_target: exp.Table | None = None
    if isinstance(tree, WRITE_TYPES):
        target = tree.this
        if isinstance(target, exp.Schema):
            target = target.this
        if isinstance(target, exp.Table):
            write_target = target

    cte_names = {c.upper() for c in res.ctes}
    seen_tables: set[tuple[str, str]] = set()
    for t in tree.find_all(exp.Table):
        if not t.name:
            continue
        access = "write" if t is write_target else "read"
        via = "cte" if t.name.upper() in cte_names else None
        key = (t.name.upper(), access)
        if key in seen_tables and not t.alias:
            continue
        seen_tables.add(key)
        res.tables.append(TableRef(name=t.name, schema_name=t.db or None, alias=t.alias or None, access=access, via=via))

    try:
        scopes = traverse_scope(tree)
    except Exception as e:  # noqa: BLE001 - SQLGlot scope building is best effort
        scopes = []
        res.diagnostics.append(f"scope analysis failed: {e}")
    for scope in scopes:
        for col in scope.columns:
            if isinstance(col.this, exp.Star):
                continue
            table, ok = _base_table(scope, col.table or None, col.name, schema)
            res.columns.append(ColumnRef(table=table, column=col.name, clause=_clause_of(col), qualifier=col.table or None, resolved=ok))
        select = scope.expression
        if isinstance(select, exp.Select):
            for join in select.args.get("joins") or []:
                on = join.args.get("on")
                kind = " ".join(x for x in [join.side, join.kind] if x) or "INNER"
                if on is None:
                    res.joins.append(JoinRef(kind=kind, left_table=None, left_column=None, right_table=None, right_column=None,
                                             condition=join.sql(dialect=out_dialect)))
                    continue
                for eq in on.find_all(exp.EQ):
                    l, r = eq.left, eq.right
                    if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                        lt, _ = _base_table(scope, l.table or None, l.name, schema)
                        rt, _ = _base_table(scope, r.table or None, r.name, schema)
                        res.joins.append(JoinRef(kind=kind, left_table=lt, left_column=l.name, right_table=rt, right_column=r.name,
                                                 condition=eq.sql(dialect=out_dialect)))
            for proj in select.expressions:
                sources = []
                for c in proj.find_all(exp.Column):
                    t, _ = _base_table(scope, c.table or None, c.name, schema)
                    sources.append(f"{t or '?'}.{c.name}")
                if sources:
                    res.lineage.append({"output": proj.alias_or_name, "sources": sorted(set(sources))})
    if isinstance(tree, exp.Insert) and isinstance(tree.this, exp.Schema) and write_target is not None:
        for c in tree.this.expressions:
            name = c.name if isinstance(c, (exp.Identifier, exp.Column)) else str(c)
            res.columns.append(ColumnRef(table=write_target.name, column=name, clause="insert"))
    if isinstance(tree, exp.Update) and write_target is not None:
        for eq in tree.expressions:
            if isinstance(eq, exp.EQ) and isinstance(eq.left, exp.Column):
                res.columns.append(ColumnRef(table=write_target.name, column=eq.left.name, clause="set"))

    res.aggregates = sorted({a.key.upper() for a in tree.find_all(exp.AggFunc)})
    res.group_by = [g.sql(dialect=out_dialect) for grp in tree.find_all(exp.Group) for g in grp.expressions]
    res.order_by = [o.sql(dialect=out_dialect) for ordr in tree.find_all(exp.Order) for o in ordr.expressions]
    res.filters = [w.this.sql(dialect=out_dialect)[:500] for w in tree.find_all(exp.Where)]
    return res
