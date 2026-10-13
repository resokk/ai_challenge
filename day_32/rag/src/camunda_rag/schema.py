"""Schema reconstruction for one profile: (component, dialect, contexts, labels, parameters).

Replays ChangeRecords in changelog order against an in-memory schema. dbms / context / label
filters and preconditions are evaluated with three-valued logic; an unknown precondition or a
change with no modelled effect lowers completeness instead of being silently ignored.
Historical definitions are not merged: the result is the state after replay.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import sqlglot
from sqlglot import exp

from camunda_rag.config import sqlglot_dialect
from camunda_rag.models import (
    ChangelogProperty, ChangeRecord, ColumnDef, ConstraintDef, EntityDef, FactStatus, IndexDef, Provenance,
    SchemaProfileResult, StorageKind,
)

PARAM_RE = re.compile(r"\$\{([^}]+)\}")
DATA_CHANGES = {"insert", "update", "delete", "loadData", "loadUpdateData", "output", "empty", "tagDatabase", "stop"}


# ------------------------------------------------------------------ filters


def _match_dbms(dbms: list[str], dialect: str) -> bool:
    if not dbms:
        return True
    pos = [d for d in dbms if not d.startswith("!")]
    neg = [d[1:] for d in dbms if d.startswith("!")]
    if dialect in neg:
        return False
    return not pos or dialect in pos or "all" in pos


def eval_filter_expression(expr: str | None, active: list[str]) -> bool:
    """Liquibase context/label expression. No active values -> everything runs."""
    if not expr or not active:
        return True
    tokens = re.findall(r"\(|\)|!|[^\s(),!]+|,", expr)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> str:
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def p_or() -> bool:
        v = p_and()
        while peek() in {",", "or", "OR"}:
            take()
            v = p_and() or v
        return v

    def p_and() -> bool:
        v = p_not()
        while peek() in {"and", "AND"}:
            take()
            v = p_not() and v
        return v

    def p_not() -> bool:
        if peek() in {"!", "not", "NOT"}:
            take()
            return not p_not()
        if peek() == "(":
            take()
            v = p_or()
            if peek() == ")":
                take()
            return v
        tok = take() if peek() is not None else ""
        return tok.lower() in {a.lower() for a in active}

    return p_or()


# ------------------------------------------------------------------ state


class SchemaState:
    def __init__(self, component: str, profile_key: str, dialect: str, is_test: bool) -> None:
        self.component = component
        self.profile_key = profile_key
        self.dialect = dialect
        self.is_test = is_test
        self.tables: dict[str, EntityDef] = {}

    def table(self, name: str) -> EntityDef | None:
        return self.tables.get(name.upper())

    def index_exists(self, name: str, table: str | None) -> bool:
        tables = [self.table(table)] if table else list(self.tables.values())
        return any(t and any(i.name.upper() == name.upper() for i in t.indexes) for t in tables)


def _column(name: str, data_type: str | None, prov: Provenance, position: int, status: FactStatus, type_source: str,
            nullable: bool | None = None, default: str | None = None, pk: bool = False, remarks: str | None = None) -> ColumnDef:
    return ColumnDef(name=name, data_type=data_type, type_source=type_source, nullable=nullable, default=default,
                     primary_key=pk, position=position, remarks=remarks, status=status, provenance=prov)


class Replayer:
    def __init__(self, state: SchemaState, params: dict[str, str]) -> None:
        self.s = state
        self.params = params
        self.unresolved: set[str] = set()
        self.outcomes: list[tuple[int, str]] = []
        self.notes: list[str] = []

    def sub(self, v: Any) -> Any:
        if not isinstance(v, str):
            return v

        def repl(m: re.Match[str]) -> str:
            k = m.group(1)
            if k in self.params:
                return self.params[k]
            self.unresolved.add(k)
            return m.group(0)

        return PARAM_RE.sub(repl, v)

    # three-valued precondition evaluation
    def pre(self, node: dict[str, Any]) -> bool | None:
        op = node["op"]
        if op in {"and", "or", "not"}:
            vals = [self.pre(c) for c in node.get("children", [])]
            if op == "not":
                # Liquibase <not> with several children = NOT(any)
                inner = None if any(v is None for v in vals) and not any(vals) else any(v for v in vals if v)
                return None if inner is None else not inner
            if op == "and":
                if any(v is False for v in vals):
                    return False
                return None if any(v is None for v in vals) else True
            if any(v is True for v in vals):
                return True
            return None if any(v is None for v in vals) else False
        a = {k: self.sub(v) for k, v in node.get("attrs", {}).items()}
        if op == "tableExists":
            return self.s.table(a.get("tableName", "")) is not None
        if op == "columnExists":
            t = self.s.table(a.get("tableName", ""))
            return t is not None and any(c.name.upper() == a.get("columnName", "").upper() for c in t.columns)
        if op == "indexExists":
            return self.s.index_exists(a.get("indexName", ""), a.get("tableName"))
        if op == "dbms":
            return _match_dbms([x.strip() for x in a.get("type", "").split(",")], self.s.dialect)
        if op == "primaryKeyExists":
            t = self.s.table(a.get("tableName", ""))
            return t is not None and any(c.kind == "primary_key" for c in t.constraints)
        if op == "foreignKeyConstraintExists":
            name = a.get("foreignKeyName", "").upper()
            return any(c.kind == "foreign_key" and (c.name or "").upper() == name for t in self.s.tables.values() for c in t.constraints)
        return None  # sqlCheck, changeSetExecuted, runningAs, custom...

    def apply(self, ch: ChangeRecord) -> str:
        d = {k: self.sub(v) for k, v in ch.details.items() if not isinstance(v, (list, dict))}
        prov = ch.provenance.model_copy(update={"status": FactStatus.recovered})
        t = ch.change_type
        name = d.get("tableName")
        if t == "createTable":
            ent = EntityDef(storage_kind=StorageKind.relational_table, component=self.s.component, schema_name=d.get("schemaName"),
                            name=name, remarks=d.get("remarks"), provenance=ch.provenance, is_test=self.s.is_test)
            for i, c in enumerate(ch.details.get("columns", [])):
                self._add_column(ent, c["column"], ch, i)
            ent.attributes["created_by"] = f"{ch.changelog_path}#{ch.changeset_id}"
            self.s.tables[name.upper()] = ent
            return "applied"
        tbl = self.s.table(name) if name else None
        if t == "addColumn":
            if tbl is None:
                return self._missing(name)
            for c in ch.details.get("columns", []):
                self._add_column(tbl, c["column"], ch, len(tbl.columns))
            return "applied"
        if t == "dropColumn":
            if tbl is None:
                return self._missing(name)
            names = [d.get("columnName")] if d.get("columnName") else [self.sub(c["column"].get("name")) for c in ch.details.get("columns", [])]
            tbl.columns = [c for c in tbl.columns if c.name.upper() not in {n.upper() for n in names if n}]
            return "applied"
        if t == "renameColumn":
            if tbl is None:
                return self._missing(name)
            for c in tbl.columns:
                if c.name.upper() == (d.get("oldColumnName") or "").upper():
                    c.history.append(f"renamed from {c.name} ({ch.changeset_id})")
                    c.name = d.get("newColumnName")
                    if d.get("columnDataType"):
                        c.data_type = d["columnDataType"]
            return "applied"
        if t == "renameTable":
            old = self.s.table(d.get("oldTableName", ""))
            if old is None:
                return self._missing(d.get("oldTableName"))
            del self.s.tables[old.name.upper()]
            old.attributes.setdefault("renamed_from", []).append(old.name)
            old.name = d.get("newTableName")
            self.s.tables[old.name.upper()] = old
            return "applied"
        if t == "dropTable":
            if tbl is None:
                return self._missing(name)
            del self.s.tables[tbl.name.upper()]
            return "applied"
        if t == "modifyDataType":
            if tbl is None:
                return self._missing(name)
            for c in tbl.columns:
                if c.name.upper() == (d.get("columnName") or "").upper():
                    c.history.append(f"type {c.data_type} -> {d.get('newDataType')} ({ch.changeset_id})")
                    c.data_type = d.get("newDataType")
            return "applied"
        if t in {"addNotNullConstraint", "dropNotNullConstraint"}:
            if tbl is None:
                return self._missing(name)
            for c in tbl.columns:
                if c.name.upper() == (d.get("columnName") or "").upper():
                    c.nullable = t == "dropNotNullConstraint"
                    c.history.append(f"{t} ({ch.changeset_id})")
            return "applied"
        if t in {"addDefaultValue", "dropDefaultValue"}:
            if tbl is None:
                return self._missing(name)
            for c in tbl.columns:
                if c.name.upper() == (d.get("columnName") or "").upper():
                    c.default = None if t == "dropDefaultValue" else next(
                        (d[k] for k in d if k.startswith("defaultValue")), None)
            return "applied"
        if t in {"addPrimaryKey", "addUniqueConstraint"}:
            if tbl is None:
                return self._missing(name)
            cols = [x.strip() for x in (d.get("columnNames") or "").split(",") if x.strip()]
            tbl.constraints.append(ConstraintDef(kind="primary_key" if t == "addPrimaryKey" else "unique", name=d.get("constraintName"),
                                                 columns=cols, provenance=prov))
            if t == "addPrimaryKey":
                for c in tbl.columns:
                    if c.name.upper() in {x.upper() for x in cols}:
                        c.primary_key = True
            return "applied"
        if t in {"dropPrimaryKey", "dropUniqueConstraint", "dropForeignKeyConstraint"}:
            target = self.s.table(d.get("baseTableName") or name or "")
            if target is None:
                return self._missing(d.get("baseTableName") or name)
            kind = {"dropPrimaryKey": "primary_key", "dropUniqueConstraint": "unique", "dropForeignKeyConstraint": "foreign_key"}[t]
            cname = (d.get("constraintName") or "").upper()
            target.constraints = [c for c in target.constraints if not (c.kind == kind and (not cname or (c.name or "").upper() == cname))]
            return "applied"
        if t == "addForeignKeyConstraint":
            base = self.s.table(d.get("baseTableName", ""))
            if base is None:
                return self._missing(d.get("baseTableName"))
            base.constraints.append(ConstraintDef(
                kind="foreign_key", name=d.get("constraintName"),
                columns=[x.strip() for x in d.get("baseColumnNames", "").split(",")], ref_table=d.get("referencedTableName"),
                ref_columns=[x.strip() for x in d.get("referencedColumnNames", "").split(",")], provenance=prov))
            return "applied"
        if t == "createIndex":
            if tbl is None:
                return self._missing(name)
            cols = [self.sub(c["column"].get("name")) for c in ch.details.get("columns", [])]
            tbl.indexes.append(IndexDef(name=d.get("indexName", ""), columns=cols, unique=str(d.get("unique", "false")).lower() == "true",
                                        provenance=prov))
            return "applied"
        if t == "dropIndex":
            for tb in ([tbl] if tbl else self.s.tables.values()):
                tb.indexes = [i for i in tb.indexes if i.name.upper() != (d.get("indexName") or "").upper()]
            return "applied"
        if t == "sql":
            return self.apply_sql(self.sub(ch.details.get("sql", "")), ch)
        if t in DATA_CHANGES:
            return "applied:data_change_no_schema_effect"
        self.notes.append(f"{ch.changelog_path}#{ch.changeset_id}: <{t}> has no modelled schema effect")
        return "unknown_effect"

    def _missing(self, name: str | None) -> str:
        self.notes.append(f"change targets table {name!r} absent from reconstructed state")
        return "unknown_effect:target_missing"

    def _add_column(self, ent: EntityDef, c: dict[str, Any], ch: ChangeRecord, pos: int) -> None:
        cons = c.get("constraints") or {}
        dtype = self.sub(c.get("type"))
        default = next((self.sub(c[k]) for k in c if k.startswith("defaultValue")), None)
        prov = ch.provenance.model_copy(update={"line_start": c.get("_line") or ch.provenance.line_start,
                                               "line_end": c.get("_line") or ch.provenance.line_end})
        nullable = None
        if "nullable" in cons:
            nullable = str(cons["nullable"]).lower() == "true"
        pk = str(cons.get("primaryKey", "false")).lower() == "true"
        if pk:
            nullable = False
        col = _column(self.sub(c.get("name")), dtype, prov, pos, FactStatus.declared, "liquibase_type", nullable, default, pk,
                      self.sub(c.get("remarks")))
        col.history.append(f"added by {ch.change_type} ({ch.changeset_id})")
        ent.columns.append(col)
        if pk:
            existing = next((x for x in ent.constraints if x.kind == "primary_key"), None)
            if existing:
                existing.columns.append(col.name)
            else:
                ent.constraints.append(ConstraintDef(kind="primary_key", name=self.sub(cons.get("primaryKeyName")), columns=[col.name],
                                                     provenance=prov))
        if str(cons.get("unique", "false")).lower() == "true":
            ent.constraints.append(ConstraintDef(kind="unique", name=self.sub(cons.get("uniqueConstraintName")), columns=[col.name],
                                                 provenance=prov))
        ref = cons.get("references") or cons.get("referencedTableName")
        if ref:
            ref_table, ref_cols = self.sub(ref), [self.sub(cons.get("referencedColumnNames", ""))]
            m = re.match(r"\s*([\w${}.]+)\s*\(([^)]*)\)", ref_table)
            if m:
                ref_table, ref_cols = m.group(1), [x.strip() for x in m.group(2).split(",")]
            ent.constraints.append(ConstraintDef(kind="foreign_key", name=self.sub(cons.get("foreignKeyName")), columns=[col.name],
                                                 ref_table=ref_table, ref_columns=ref_cols, provenance=prov))

    def apply_sql(self, sql: str, ch: ChangeRecord) -> str:
        read = sqlglot_dialect(self.s.dialect) or None
        try:
            stmts = [x for x in sqlglot.parse(sql, read=read) if x is not None]
        except Exception as e:  # noqa: BLE001 - SQLGlot raises several error types
            self.notes.append(f"{ch.changelog_path}#{ch.changeset_id}: SQL not parsed by SQLGlot: {e}"[:300])
            return "unknown_effect:sql_unparsed"
        outcomes = {self._sql_stmt(st, ch) for st in stmts}
        if any(o.startswith("unknown") for o in outcomes):
            return "unknown_effect:sql"
        return "applied" if "applied" in outcomes else next(iter(outcomes), "applied")

    def _sql_stmt(self, st: exp.Expression, ch: ChangeRecord) -> str:
        prov = ch.provenance.model_copy(update={"status": FactStatus.recovered})
        if isinstance(st, exp.Command):
            self.notes.append(f"{ch.changelog_path}#{ch.changeset_id}: opaque SQL command {st.sql()[:80]!r}")
            return "unknown_effect"
        if isinstance(st, (exp.Insert, exp.Update, exp.Delete, exp.Merge)):
            return "applied:data_change_no_schema_effect"
        if isinstance(st, exp.Create):
            kind = (st.args.get("kind") or "").upper()
            if kind == "TABLE" and isinstance(st.this, exp.Schema):
                tname = st.this.this.name
                ent = EntityDef(storage_kind=StorageKind.relational_table, component=self.s.component, name=tname,
                                provenance=ch.provenance, is_test=self.s.is_test)
                for i, cd in enumerate(x for x in st.this.expressions if isinstance(x, exp.ColumnDef)):
                    kinds = {type(c.args.get("kind")).__name__ for c in cd.constraints}
                    ent.columns.append(_column(cd.name, cd.args["kind"].sql() if cd.args.get("kind") else None, prov, i,
                                               FactStatus.declared, "sql_ddl",
                                               False if "NotNullColumnConstraint" in kinds or "PrimaryKeyColumnConstraint" in kinds else None,
                                               None, "PrimaryKeyColumnConstraint" in kinds))
                for pk in st.this.find_all(exp.PrimaryKey):
                    ent.constraints.append(ConstraintDef(kind="primary_key", columns=[c.name for c in pk.expressions], provenance=prov))
                for fk in st.this.find_all(exp.ForeignKey):
                    ref = fk.args.get("reference")
                    ent.constraints.append(ConstraintDef(
                        kind="foreign_key", columns=[c.name for c in fk.expressions],
                        ref_table=ref.this.this.name if ref and isinstance(ref.this, exp.Schema) else None,
                        ref_columns=[c.name for c in ref.this.expressions] if ref and isinstance(ref.this, exp.Schema) else [],
                        provenance=prov))
                self.s.tables[tname.upper()] = ent
                return "applied"
            if kind == "INDEX":
                idx = st.this
                table = idx.args.get("table") if isinstance(idx, exp.Index) else None
                tb = self.s.table(table.name) if table is not None else None
                if tb is None:
                    return self._missing(table.name if table is not None else None)
                params = idx.args.get("params")
                cols = [c.name for c in (params.args.get("columns") if params else []) or []] if params else []
                tb.indexes.append(IndexDef(name=idx.name, columns=cols, unique=bool(st.args.get("unique")), provenance=prov))
                return "applied"
        if isinstance(st, exp.Alter):
            tb = self.s.table(st.this.name)
            if tb is None:
                return self._missing(st.this.name)
            res = "applied"
            for action in st.args.get("actions") or []:
                if isinstance(action, exp.ColumnDef):
                    tb.columns.append(_column(action.name, action.args["kind"].sql() if action.args.get("kind") else None, prov,
                                              len(tb.columns), FactStatus.declared, "sql_ddl"))
                elif isinstance(action, exp.Drop) and (action.args.get("kind") or "").upper() == "COLUMN":
                    tb.columns = [c for c in tb.columns if c.name.upper() != action.this.name.upper()]
                else:
                    self.notes.append(f"{ch.changelog_path}#{ch.changeset_id}: ALTER action {action.key} not modelled")
                    res = "unknown_effect"
            return res
        if isinstance(st, exp.Drop):
            kind = (st.args.get("kind") or "").upper()
            if kind == "TABLE":
                self.s.tables.pop(st.this.name.upper(), None)
                return "applied"
            if kind == "INDEX":
                for tb in self.s.tables.values():
                    tb.indexes = [i for i in tb.indexes if i.name.upper() != st.this.name.upper()]
                return "applied"
        self.notes.append(f"{ch.changelog_path}#{ch.changeset_id}: SQL statement {st.key} has no modelled schema effect")
        return "unknown_effect"


def reconstruct(changes: list[ChangeRecord], properties: list[ChangelogProperty], *, component: str, dialect: str,
                contexts: list[str], labels: list[str], parameters: dict[str, str], is_test: bool) -> SchemaProfileResult:
    params = dict(parameters)
    for p in properties:  # changelog <property> applies when its dbms/context matches; first definition wins
        if _match_dbms(p.dbms, dialect) and eval_filter_expression(p.context, contexts) and p.name not in params:
            params[p.name] = p.value
    key = f"{component}|{dialect}|ctx={','.join(contexts) or '*'}|labels={','.join(labels) or '*'}|" + \
          ",".join(f"{k}={v}" for k, v in sorted(parameters.items()))
    state = SchemaState(component, key, dialect, is_test)
    r = Replayer(state, params)
    counts: dict[str, int] = {}
    halted = False
    for ch in sorted(changes, key=lambda c: c.order):
        if halted:
            outcome = "not_reached:halted"
        elif not _match_dbms(ch.dbms, dialect):
            outcome = "skipped:dbms"
        elif not eval_filter_expression(ch.contexts, contexts):
            outcome = "skipped:context"
        elif not eval_filter_expression(ch.labels, labels):
            outcome = "skipped:label"
        else:
            pre = r.pre(ch.preconditions) if ch.preconditions else True
            if pre is False and (ch.on_fail or "HALT").upper() in {"MARK_RAN", "CONTINUE"}:
                outcome = "skipped:precondition"
            elif pre is False and (ch.on_fail or "HALT").upper() == "HALT":
                outcome, halted = "halted:precondition_failed", True
            else:
                outcome = r.apply(ch)
                if pre is None:
                    outcome = f"uncertain_precondition:{outcome}"
        r.outcomes.append((ch.order, outcome))
        bucket = outcome.split(":")[0]
        counts[bucket] = counts.get(bucket, 0) + 1
    reasons = list(r.notes)
    if r.unresolved:
        reasons.append("unresolved changelog parameters: " + ", ".join(sorted(r.unresolved)))
    uncertain = counts.get("uncertain_precondition", 0) + counts.get("unknown_effect", 0) + counts.get("halted", 0)
    level = "complete" if not uncertain and not r.unresolved else "partial"
    completeness = {"level": level, "changes": len(changes), "outcomes": counts, "reasons": reasons[:200],
                    "unresolved_parameters": sorted(r.unresolved)}
    entities = [deepcopy(t) for t in state.tables.values()]
    for e in entities:
        e.attributes["completeness"] = level
    return SchemaProfileResult(profile_key=key, component=component, dialect=dialect, contexts=contexts, labels=labels,
                               parameters=params, is_test=is_test, entities=entities, applied_changes=r.outcomes,
                               completeness=completeness)
