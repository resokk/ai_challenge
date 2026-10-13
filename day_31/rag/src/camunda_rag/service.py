"""Single service layer used by CLI, HTTP API and MCP tools.

Every call runs against one published generation: the generation is pinned when the service
opens it and swapped atomically (under a lock) when data/CURRENT changes.
"""

from __future__ import annotations

import difflib
import json
import threading
from typing import Any

from camunda_rag.config import Settings
from camunda_rag.retrieval import HybridRetriever
from camunda_rag.storage import current_generation, generations_dir, open_readonly
from camunda_rag.validation import ValidationResult, validate

NOT_ESTABLISHED = "назначение не установлено по исходникам"


class NoIndexError(RuntimeError):
    pass


class _Generation:
    def __init__(self, settings: Settings, gen_id: str) -> None:
        self.id = gen_id
        self.dir = generations_dir(settings.data_dir) / gen_id
        self.db = open_readonly(self.dir / "catalog.sqlite")
        self.meta = dict(self.db.execute("SELECT * FROM index_generations").fetchone() or {})
        self.index = None
        self.retriever = HybridRetriever(self.db)


def _j(v: str | None, default: Any = None) -> Any:
    return json.loads(v) if v else default


class CatalogService:
    def __init__(self, settings: Settings, load_embedder: bool = True, embedder=None) -> None:
        self.s = settings
        self.load_embedder = load_embedder or embedder is not None
        self._lock = threading.RLock()
        self._gen: _Generation | None = None
        self._embedder_cache = embedder

    # ------------------------------------------------------------------ generation pinning

    def _current(self) -> _Generation:
        gen_id = current_generation(self.s.data_dir)
        if gen_id is None:
            raise NoIndexError(f"no published index in {self.s.data_dir}; run `camunda-rag index` first")
        with self._lock:
            if self._gen is None or self._gen.id != gen_id:
                new = _Generation(self.s, gen_id)
                if self.load_embedder and (new.dir / "vectors.faiss").exists():
                    import faiss

                    from camunda_rag.embeddings import Embedder

                    if self._embedder_cache is None:
                        self._embedder_cache = Embedder(self.s.embedding)
                    new.index = faiss.read_index(str(new.dir / "vectors.faiss"))
                    new.retriever = HybridRetriever(new.db, new.index, self._embedder_cache)
                self._gen = new
            return self._gen

    def status(self) -> dict[str, Any]:
        g = self._current()
        db = g.db
        return {
            "generation": g.id, "commit_sha": g.meta.get("commit_sha"), "repository": self.s.repository_url,
            "stats": _j(g.meta.get("stats_json"), {}), "dense_search": g.index is not None,
            "profiles": [dict(r) | {"completeness": _j(r["completeness_json"], {}).get("level")}
                         for r in db.execute("SELECT id, profile_key, component, storage_family, dialect, is_test, is_default, completeness_json FROM schema_profiles")],
        }

    # ------------------------------------------------------------------ search

    def search(self, query: str, k: int = 10, storage_kind: str | None = None, component: str | None = None,
               doc_types: list[str] | None = None, include_tests: bool = False) -> dict[str, Any]:
        g = self._current()
        with self._lock:
            out = g.retriever.search(query, k=k, storage_kind=storage_kind, component=component, doc_types=doc_types,
                                     include_tests=include_tests)
        out["generation"] = g.id
        out["commit_sha"] = g.meta.get("commit_sha")
        return out

    # ------------------------------------------------------------------ describe

    def _profiles(self, db, dialect: str | None, component: str | None, include_tests: bool) -> list[int]:
        """Default: Liquibase product profiles (+ non-relational stores). A component selects any family."""
        if dialect:
            rows = db.execute("""SELECT id FROM schema_profiles WHERE dialect=? AND (is_test=0 OR ?)
                                 AND (storage_family='relational' OR ? IS NOT NULL)""", (dialect, int(include_tests), component))
        elif component:
            rows = db.execute("""SELECT id FROM schema_profiles WHERE (is_test=0 OR ?)
                                 AND (dialect IS NULL OR dialect=?)""", (int(include_tests), self.s.default_dialect))
        else:
            rows = db.execute("SELECT id FROM schema_profiles WHERE is_default=1 OR (? AND is_test=1)", (int(include_tests),))
        return [r[0] for r in rows]

    def describe(self, name: str, dialect: str | None = None, storage_kind: str | None = None, component: str | None = None,
                 include_tests: bool = False) -> dict[str, Any]:
        g = self._current()
        db = g.db
        key = name.strip().upper()
        if self.s.table_prefix and key.startswith(self.s.table_prefix.upper()):
            key = key[len(self.s.table_prefix):]
        pids = self._profiles(db, dialect, component, include_tests)
        marks = ",".join("?" * len(pids)) or "NULL"
        sql = f"SELECT * FROM entities WHERE name_upper=? AND profile_id IN ({marks})"
        args: list[Any] = [key, *pids]
        if storage_kind:
            sql += " AND storage_kind=?"
            args.append(storage_kind)
        if component:
            sql += " AND component LIKE ?"
            args.append(component + "%")
        rows = db.execute(sql, args).fetchall()
        base = {"generation": g.id, "commit_sha": g.meta.get("commit_sha")}
        if not rows:
            names = [r[0] for r in db.execute("SELECT DISTINCT name FROM entities")]
            return base | {"status": "not_found", "name": name,
                           "suggestions": difflib.get_close_matches(name.upper(), [n.upper() for n in names], n=5, cutoff=0.6)}
        if len(rows) > 1:
            return base | {"status": "ambiguous", "name": name, "candidates": [
                {"entity_key": r["entity_key"], "storage_kind": r["storage_kind"], "component": r["component"], "path": r["path"]}
                for r in rows], "hint": "pass storage_kind and/or component"}
        return base | {"status": "ok"} | self._entity_detail(db, rows[0])

    def _entity_detail(self, db, e) -> dict[str, Any]:
        eid = e["id"]
        prof = db.execute("SELECT * FROM schema_profiles WHERE id=?", (e["profile_id"],)).fetchone()
        mappings: dict[str, list[dict[str, Any]]] = {}
        for m in db.execute("SELECT column_name, java_type, property, source, fact_status FROM mappings WHERE entity_id=?", (eid,)):
            mappings.setdefault(m["column_name"].upper(), []).append(
                {"java_type": m["java_type"], "property": m["property"], "result_map": m["source"], "status": m["fact_status"],
                 "note": "Java type of the property is not evidence of the SQL type"})
        columns = []
        for c in db.execute("SELECT * FROM columns WHERE entity_id=? ORDER BY position", (eid,)):
            columns.append({
                "name": c["name"], "data_type": c["data_type"], "type_source": c["type_source"],
                "nullable": None if c["nullable"] is None else bool(c["nullable"]), "default": c["default_value"],
                "primary_key": bool(c["primary_key"]), "fact_status": c["fact_status"],
                "purpose": c["purpose"] or NOT_ESTABLISHED, "purpose_status": c["purpose_status"],
                "java_mappings": mappings.get(c["name"].upper(), []), "history": _j(c["history_json"], []),
                "provenance": {"path": c["path"], "line": c["line_start"]},
            })
        rels = {"declared_foreign_keys": [], "inferred_from_joins": []}
        for r in db.execute("""SELECT r.*, a.name a_name, b.name b_name FROM relationships r
                               JOIN entities a ON a.id=r.from_entity_id LEFT JOIN entities b ON b.id=r.to_entity_id
                               WHERE r.from_entity_id=? OR r.to_entity_id=?""", (eid, eid)):
            item = {"from": f"{r['a_name']}({', '.join(_j(r['from_columns_json'], []))})",
                    "to": f"{r['b_name'] or r['to_table']}({', '.join(_j(r['to_columns_json'], []))})",
                    "status": r["fact_status"], "evidence_count": r["evidence_count"]}
            rels["declared_foreign_keys" if r["kind"] == "declared_fk" else "inferred_from_joins"].append(item)
        changes = [dict(x) for x in db.execute(
            """SELECT c.changelog_path, c.changeset_id, c.change_type, c.line_start, o.outcome FROM schema_changes c
               JOIN schema_change_outcomes o ON o.change_id=c.id AND o.profile_id=?
               WHERE upper(replace(c.table_name, '${prefix}', ''))=? ORDER BY c.ord""", (e["profile_id"], e["name_upper"]))]
        usage = {r["access"]: r["n"] for r in db.execute(
            "SELECT access, COUNT(DISTINCT query_id) n FROM query_table_refs WHERE entity_id=? GROUP BY access", (eid,))}
        evidence = [dict(x) for x in db.execute(
            "SELECT path, line_start, line_end, symbol, extractor, evidence_kind, confidence, fact_status, commit_sha, repository "
            "FROM evidence WHERE subject_kind='entity' AND subject_id=?", (eid,))]
        other = [dict(x) for x in db.execute(
            """SELECT p.dialect, p.profile_key, e2.storage_kind FROM entities e2 JOIN schema_profiles p ON p.id=e2.profile_id
               WHERE e2.name_upper=? AND e2.id!=? AND e2.component=?""", (e["name_upper"], eid, e["component"]))]
        return {
            "entity": {"entity_key": e["entity_key"], "name": e["name"], "storage_kind": e["storage_kind"], "component": e["component"],
                       "schema": e["schema_name"], "is_test": bool(e["is_test"]), "attributes": _j(e["attributes_json"], {})},
            "profile": {"key": prof["profile_key"], "dialect": prof["dialect"], "parameters": _j(prof["parameters_json"], {}),
                        "completeness": _j(prof["completeness_json"], {})},
            "purpose": {"text": e["purpose"] or NOT_ESTABLISHED, "status": e["purpose_status"] or "not_established",
                        "sources": _j(e["purpose_sources_json"], [])},
            "usage_summary": {"text": e["usage_summary"], "status": "inferred"} if e["usage_summary"] else None,
            "columns": columns,
            "constraints": [{"kind": k["kind"], "name": k["name"], "columns": _j(k["columns_json"], []), "ref_table": k["ref_table"],
                             "ref_columns": _j(k["ref_columns_json"], [])}
                            for k in db.execute("SELECT * FROM constraints WHERE entity_id=?", (eid,))],
            "indexes": [{"name": i["name"], "columns": _j(i["columns_json"], []), "unique": bool(i["is_unique"])}
                        for i in db.execute("SELECT * FROM indexes WHERE entity_id=?", (eid,))],
            "relationships": rels,
            "schema_changes": changes,
            "query_usage": {"read": usage.get("read", 0), "write": usage.get("write", 0)},
            "evidence": evidence,
            "other_profiles": other,
        }

    # ------------------------------------------------------------------ queries

    def queries(self, table: str | None = None, column: str | None = None, mapper: str | None = None,
                access: str | None = None, dialect: str | None = None, language: str | None = None, text: str | None = None,
                include_tests: bool = False, include_variants: bool = False, limit: int = 20) -> dict[str, Any]:
        g = self._current()
        db = g.db
        where, args = ["1=1"], []
        if not include_tests:
            where.append("q.is_test=0")
        if table:
            sub = "SELECT query_id FROM query_table_refs WHERE table_upper=?"
            targs: list[Any] = [table.upper()]
            if access:
                sub += " AND access=?"
                targs.append(access)
            where.append(f"q.id IN ({sub})")
            args += targs
        if column:
            sub = "SELECT query_id FROM query_column_refs WHERE column_upper=?"
            cargs: list[Any] = [column.upper()]
            if table:
                sub += " AND upper(table_name)=?"
                cargs.append(table.upper())
            where.append(f"q.id IN ({sub})")
            args += cargs
        if mapper:
            where.append("(q.namespace LIKE ? OR q.symbol LIKE ?)")
            args += [f"%{mapper}%", f"%{mapper}%"]
        if language:
            where.append("q.language=?")
            args.append(language)
        if text:
            where.append("(q.raw_text LIKE ? OR q.query_key LIKE ?)")
            args += [f"%{text}%", f"%{text}%"]
        where.append("q.statement_kind!='sql_fragment'")
        total = db.execute(f"SELECT COUNT(*) FROM queries q WHERE {' AND '.join(where)}", args).fetchone()[0]
        rows = db.execute(f"SELECT q.* FROM queries q WHERE {' AND '.join(where)} ORDER BY q.query_key LIMIT ?", args + [limit]).fetchall()
        want = dialect or self.s.default_dialect
        out = []
        for q in rows:
            variants = db.execute("SELECT * FROM query_variants WHERE query_id=? ORDER BY ordinal", (q["id"],)).fetchall()
            statuses: dict[str, dict[str, int]] = {}
            for v in variants:
                d = statuses.setdefault(v["dialect"] or "unknown", {})
                d[v["parse_status"]] = d.get(v["parse_status"], 0) + 1
            rep = next((v for v in variants if v["dialect"] == want), variants[0] if variants else None)
            method = db.execute("SELECT fqn, signature, javadoc FROM symbols WHERE id=?", (q["mapper_method_symbol_id"],)).fetchone() \
                if q["mapper_method_symbol_id"] else None
            item = {
                "key": q["query_key"], "language": q["language"], "origin": q["origin"], "statement_kind": q["statement_kind"],
                "symbol": q["symbol"], "database_id": q["database_id"], "result_map": q["result_map"],
                "description": q["description"], "is_test": bool(q["is_test"]),
                "provenance": {"path": q["path"], "line_start": q["line_start"], "line_end": q["line_end"]},
                "mapper_method": dict(method) if method else None,
                "tables": [dict(r) for r in db.execute(
                    "SELECT DISTINCT table_name, access, entity_id IS NOT NULL AS in_catalog FROM query_table_refs WHERE query_id=?", (q["id"],))],
                "variant_statuses": statuses,
                "representative": None if rep is None else {
                    "dialect": rep["dialect"], "dialect_evidence": rep["dialect_evidence"], "parse_status": rep["parse_status"],
                    "conditions": _j(rep["conditions_json"], []), "sql": rep["normalized_sql"] or rep["sql_text"],
                    "unresolved": _j(rep["unresolved_json"], []), "fingerprint": rep["fingerprint"],
                    "parameters": [dict(p) for p in db.execute(
                        "SELECT name, kind, java_type, jdbc_type, type_handler, resolved_value FROM query_parameters WHERE variant_id=? ORDER BY position",
                        (rep["id"],))],
                },
            }
            if include_variants:
                item["variants"] = [{"dialect": v["dialect"], "parse_status": v["parse_status"], "partial": bool(v["partial"]),
                                     "conditions": _j(v["conditions_json"], []), "sql": v["sql_text"],
                                     "normalized_sql": v["normalized_sql"], "unresolved": _j(v["unresolved_json"], []),
                                     "diagnostics": _j(v["diagnostics_json"], [])} for v in variants]
            out.append(item)
        return {"generation": g.id, "commit_sha": g.meta.get("commit_sha"), "total": total, "returned": len(out), "queries": out}

    # ------------------------------------------------------------------ validation

    def validation_catalog(self, dialect: str, component: str | None = None) -> dict[str, Any]:
        g = self._current()
        db = g.db
        sql = "SELECT id, profile_key, completeness_json FROM schema_profiles WHERE dialect=? AND is_test=0"
        args: list[Any] = [dialect]
        if component:
            sql += " AND component LIKE ?"
            args.append(component + "%")
        else:
            sql += " AND storage_family='relational'"
        profiles = db.execute(sql, args).fetchall()
        tables: dict[str, set[str]] = {}
        for p in profiles:
            for e in db.execute("SELECT id, name_upper FROM entities WHERE profile_id=?", (p["id"],)):
                tables.setdefault(e["name_upper"], set()).update(
                    r[0] for r in db.execute("SELECT name_upper FROM columns WHERE entity_id=?", (e["id"],)))
        non_rel = {r[0]: r[1] for r in db.execute("SELECT name_upper, storage_kind FROM entities WHERE storage_kind!='relational_table'")}
        levels = {_j(p["completeness_json"], {}).get("level") for p in profiles}
        return {"profile_key": "; ".join(p["profile_key"] for p in profiles) or None,
                "completeness": "partial" if "partial" in levels else ("complete" if profiles else None),
                "tables": tables, "non_relational": non_rel}

    def validate(self, sql: str, dialect: str | None = None, component: str | None = None) -> ValidationResult:
        d = dialect or self.s.default_dialect
        res = validate(sql, d, self.validation_catalog(d, component))
        return res
