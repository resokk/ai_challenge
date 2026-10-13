"""Indexing pipeline: discovery -> extractors -> reconstruction -> catalog -> embeddings -> publish."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from camunda_rag import discovery as disc
from camunda_rag.config import Settings
from camunda_rag.extractors import docs as docs_x
from camunda_rag.extractors import java as java_x
from camunda_rag.extractors import mybatis as mb
from camunda_rag.extractors import search_schema, sql_files
from camunda_rag.extractors.liquibase import LiquibaseExtractor
from camunda_rag.extractors.vendor_properties import load_vendor_properties
from camunda_rag.models import (
    ChangeRecord, Diagnostic, DocSection, EntityDef, FileRole, FileStatus, Provenance, QueryDef, ResultMapping,
    SchemaProfileResult, StorageKind, SymbolDef,
)
from camunda_rag.schema import reconstruct
from camunda_rag.sql_analysis import PARSER, SqlAnalysis, analyze
from camunda_rag.storage import CatalogWriter, dumps, generations_dir, publish

log = logging.getLogger(__name__)
NOT_ESTABLISHED = "назначение не установлено по исходникам"
# Changelog parameters Camunda sets from vendor properties
# (db/rdbms-schema/.../LiquibaseScriptGenerator.java: liquibase.setChangeLogParameter(...)).
LIQUIBASE_VENDOR_PARAMS = {"userCharColumnSize": "userCharColumn.size", "errorMessageSize": "errorMessage.size",
                           "treePathSize": "treePath.size"}


def split_identifier(name: str) -> list[str]:
    parts = re.split(r"[_.\s/#:\-]+", name)
    out: list[str] = []
    for p in parts:
        out.extend(x.lower() for x in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", p))
    return [x for x in out if x]


def identifiers_text(names: list[str]) -> str:
    toks: list[str] = []
    for n in names:
        if not n:
            continue
        toks.append(n)
        toks.extend(split_identifier(n))
    return " ".join(dict.fromkeys(toks))


@dataclass
class Extracted:
    inventory: disc.Inventory
    changes: dict[str, list[ChangeRecord]] = field(default_factory=dict)  # component -> records
    profiles: list[SchemaProfileResult] = field(default_factory=list)
    other_entities: list[EntityDef] = field(default_factory=list)
    queries: list[QueryDef] = field(default_factory=list)
    symbols: list[SymbolDef] = field(default_factory=list)
    result_maps: list[ResultMapping] = field(default_factory=list)
    doc_sections: list[DocSection] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    analyses: dict[tuple[int, int], SqlAnalysis] = field(default_factory=dict)  # (query idx, variant idx)
    java_header: dict[str, Any] = field(default_factory=dict)


class Indexer:
    def __init__(self, settings: Settings, progress: Callable[[str], None] = lambda m: log.info(m)) -> None:
        self.s = settings
        self.progress = progress

    # ------------------------------------------------------------------ extraction

    def extract(self, skip_java: bool = False) -> Extracted:
        repo = self.s.repo_path
        t0 = time.time()
        inv = disc.discover(repo, self.s.limits.max_file_bytes)
        self.progress(f"discovery: {len(inv.files)} tracked files at {inv.commit_sha[:12]} ({time.time() - t0:.1f}s)")
        ex = Extracted(inventory=inv)
        is_test = lambda p: disc.file_role(p) == FileRole.test  # noqa: E731
        known = {f.path for f in inv.files}

        vendor = load_vendor_properties(repo, [f.path for f in inv.by_kind("vendor_properties")])
        for f in inv.by_kind("vendor_properties"):
            f.extractors.append("vendor-properties")
            if f.path.split("/")[-1].rsplit(".", 1)[0] not in vendor:
                f.status, f.reason = FileStatus.irrelevant, "test vendor properties; not used for profiles"

        # Liquibase
        lq = LiquibaseExtractor(repo, known, is_test)
        lq_files = [f.path for f in inv.by_kind("liquibase")]
        reached: set[str] = set()
        for root in lq.roots(lq_files):
            res = lq.extract_root(root)
            reached.update(res.files)
            ex.diagnostics.extend(res.diagnostics)
            ex.changes[res.component] = ex.changes.get(res.component, []) + res.changes
            dialects = self.s.dialects if not res.is_test else [self.s.default_dialect]
            for d in dialects:
                vp = vendor.get(d)
                params = {"prefix": self.s.table_prefix}
                for param, prop in LIQUIBASE_VENDOR_PARAMS.items():
                    if vp and prop in vp.values:
                        params[param] = vp.values[prop]
                ex.profiles.append(reconstruct(res.changes, res.properties, component=res.component, dialect=d,
                                               contexts=self.s.liquibase_contexts, labels=self.s.liquibase_labels,
                                               parameters=params, is_test=res.is_test))
        for f in inv.by_kind("liquibase"):
            f.extractors.append("liquibase")
            if f.path not in reached:
                f.status, f.reason = FileStatus.unsupported, "changelog not reachable from any root (parse failure)"
        self.progress(f"liquibase: {sum(len(v) for v in ex.changes.values())} changes, {len(ex.profiles)} profiles")

        # standalone SQL files
        for f in inv.by_kind("sql"):
            qs, chs, ds = sql_files.extract(repo, f.path, f.role == FileRole.test, self.s.default_dialect)
            f.extractors.append("sql-file")
            ex.queries.extend(qs)
            ex.diagnostics.extend(ds)
            if chs:
                comp = chs[0].component
                ex.changes[comp] = chs
                dialect = sql_files.dialect_of(f.path)[0] or self.s.default_dialect
                ex.profiles.append(reconstruct(chs, [], component=comp, dialect=dialect, contexts=[], labels=[],
                                               parameters={}, is_test=f.role == FileRole.test))

        # MyBatis
        reg = mb.Registry()
        for f in inv.by_kind("mybatis"):
            mb.parse_mapper(repo, f.path, f.role == FileRole.test, reg)
            f.extractors.append("mybatis")
        renderer = mb.MyBatisRenderer(reg, self.s.limits)
        ex.queries.extend(mb.statement_queries(reg, renderer, self.s.dialects, vendor, self.s.table_prefix))
        ex.result_maps.extend(reg.result_maps)
        ex.diagnostics.extend(reg.diagnostics)
        self.progress(f"mybatis: {len(reg.statements)} statements, {sum(len(v) for v in reg.fragments.values())} fragments")

        # Java
        java_files = [f.path for f in inv.by_kind("java")]
        if skip_java:
            for f in inv.by_kind("java"):
                f.status, f.reason = FileStatus.unsupported, "java analysis skipped (--skip-java)"
        else:
            all_java = [f.path for f in inv.files if f.ext == "java"]
            out = java_x.run_analyzer(self.s, repo, java_files, all_java, sorted(reg.namespaces), self.s.data_dir / "work")
            qs, syms, file_recs, ds, header = java_x.convert(java_x.read_jsonl(out), is_test, self.s.table_prefix)
            ex.queries.extend(qs)
            ex.symbols.extend(syms)
            ex.diagnostics.extend(ds)
            ex.java_header = header
            for f in inv.by_kind("java"):
                f.extractors.append("java-analyzer")
                rec = file_recs.get(f.path)
                if rec is None:
                    f.status, f.reason = FileStatus.unsupported, "no analyzer output"
                elif rec.get("status") == "parse_error":
                    f.status, f.reason = FileStatus.unsupported, f"JavaParser error: {rec.get('message', '')[:200]}"
                elif not rec.get("records"):
                    f.status, f.reason = FileStatus.irrelevant, "parsed; no query or model records"
            ex.other_entities.extend(java_x.rocksdb_entities(syms))
            self.progress(f"java: {len(qs)} query candidates, {len(syms)} symbols")

        # ES / OpenSearch mappings
        for f in inv.by_kind("search_mapping"):
            ent, ds = search_schema.extract(repo, f.path, f.role == FileRole.test)
            f.extractors.append("search-mapping")
            ex.diagnostics.extend(ds)
            if ent:
                ex.other_entities.append(ent)
            else:
                f.status, f.reason = FileStatus.irrelevant, ds[0].message if ds else "no mapping"

        # docs
        for f in inv.by_kind("doc"):
            ex.doc_sections.extend(docs_x.sections(repo, f.path, f.role))
            f.extractors.append("docs")

        self._analyze_queries(ex)
        return ex

    def _schema_for(self, ex: Extracted, dialect: str) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for p in ex.profiles:
            if p.dialect == dialect and not p.is_test:
                for e in p.entities:
                    out.setdefault(e.name.upper(), set()).update(c.name.upper() for c in e.columns)
        return out

    def _analyze_queries(self, ex: Extracted) -> None:
        schemas: dict[str, dict[str, set[str]]] = {}
        n = 0
        for qi, q in enumerate(ex.queries):
            for vi, v in enumerate(q.variants):
                if q.language != "sql" or v.partial:
                    continue
                dialect = v.dialect or self.s.default_dialect
                if dialect not in schemas:
                    schemas[dialect] = self._schema_for(ex, dialect)
                a = analyze(v.sql, dialect, schemas[dialect])
                if v.dialect is None:
                    a.diagnostics.append(f"dialect assumed: {dialect}")
                ex.analyses[(qi, vi)] = a
                n += 1
        self.progress(f"sqlglot: {n} variants analysed")

    # ------------------------------------------------------------------ catalog

    def build(self, skip_java: bool = False, embed: bool = True, embedder=None) -> str:
        ex = self.extract(skip_java=skip_java)
        gen_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        gdir = generations_dir(self.s.data_dir) / gen_id
        gdir.mkdir(parents=True)
        w = CatalogWriter(gdir / "catalog.sqlite", self.s.repository_url, ex.inventory.commit_sha)
        try:
            stats = CatalogBuilder(self.s, w, ex).write()
            stats.update(build_retrieval(self.s, w, gdir, embed=embed, progress=self.progress, doc_sections=ex.doc_sections,
                                         embedder=embedder))
            w.insert("index_generations", id=gen_id, created_at=datetime.now(timezone.utc).isoformat(), status="published",
                     commit_sha=ex.inventory.commit_sha, settings_json=self.s.model_dump_json(), stats_json=dumps(stats),
                     embedding_model=stats.get("embedding_model"), embedding_dim=stats.get("embedding_dim"),
                     catalog_schema_version=1)
            w.insert("snapshots", repository=self.s.repository_url, commit_sha=ex.inventory.commit_sha, generation_id=gen_id,
                     indexed_at=datetime.now(timezone.utc).isoformat())
        except BaseException:
            w.close()
            shutil.rmtree(gdir, ignore_errors=True)
            raise
        w.close()
        publish(self.s.data_dir, gen_id)
        self.progress(f"published generation {gen_id}: {json.dumps(stats, ensure_ascii=False)}")
        return gen_id


class CatalogBuilder:
    def __init__(self, settings: Settings, w: CatalogWriter, ex: Extracted) -> None:
        self.s = settings
        self.w = w
        self.ex = ex
        self.entity_ids: dict[tuple[int, str], int] = {}  # (profile_id, NAME) -> entity id
        self.default_profile_by_dialect: dict[str, list[int]] = defaultdict(list)
        self.entity_rows: dict[int, dict[str, Any]] = {}
        self.column_ids: dict[tuple[int, str], int] = {}

    def write(self) -> dict[str, Any]:
        self._files()
        self._profiles()
        query_ids = self._queries()
        self._symbols_and_mappings(query_ids)
        self._relationships()
        self._purposes()
        for d in self.ex.diagnostics:
            self.w.insert("diagnostics", path=d.path, line=d.line, extractor=d.extractor, severity=d.severity, code=d.code,
                          message=d.message, subject=d.subject)
        status = Counter(f.status.value for f in self.ex.inventory.files)
        return {"commit_sha": self.ex.inventory.commit_sha, "files": len(self.ex.inventory.files), "file_status": dict(status),
                "profiles": len(self.ex.profiles), "queries": len(self.ex.queries),
                "variants": sum(len(q.variants) for q in self.ex.queries), "symbols": len(self.ex.symbols),
                "entities": len(self.entity_rows), "diagnostics": len(self.ex.diagnostics),
                "java_analyzer": self.ex.java_header, "sql_parser": PARSER}

    def _files(self) -> None:
        for f in self.ex.inventory.files:
            m = f.to_model()
            self.w.insert("source_files", path=m.path, language=m.language, role=m.role.value, status=m.status.value,
                          reason=m.reason, extractors=",".join(m.extractors), size=m.size)

    def _entity(self, profile_id: int, profile_key: str, e: EntityDef) -> int:
        key = f"{profile_key}|{e.storage_kind.value}|{e.component}|{e.schema_name or ''}|{e.name}"
        eid = self.w.insert("entities", entity_key=key, profile_id=profile_id, storage_kind=e.storage_kind.value,
                            component=e.component, schema_name=e.schema_name, name=e.name, name_upper=e.name.upper(),
                            remarks=e.remarks, is_test=int(e.is_test), attributes_json=dumps(e.attributes),
                            completeness=e.attributes.get("completeness"), path=e.provenance.path,
                            line_start=e.provenance.line_start, line_end=e.provenance.line_end)
        self.w.evidence("entity", eid, e.provenance)
        self.entity_ids[(profile_id, e.name.upper())] = eid
        self.entity_rows[eid] = {"name": e.name, "profile_id": profile_id, "kind": e.storage_kind.value, "def": e}
        for c in e.columns:
            cid = self.w.insert("columns", entity_id=eid, name=c.name, name_upper=c.name.upper(), position=c.position,
                                data_type=c.data_type, type_source=c.type_source,
                                nullable=None if c.nullable is None else int(c.nullable), default_value=c.default,
                                primary_key=int(c.primary_key), remarks=c.remarks, fact_status=c.status.value,
                                history_json=dumps(c.history), path=c.provenance.path, line_start=c.provenance.line_start)
            self.w.evidence("column", cid, c.provenance, status=c.status.value)
            self.column_ids[(eid, c.name.upper())] = cid
        for k in e.constraints:
            kid = self.w.insert("constraints", entity_id=eid, kind=k.kind, name=k.name, columns_json=dumps(k.columns),
                                ref_table=k.ref_table, ref_columns_json=dumps(k.ref_columns))
            self.w.evidence("constraint", kid, k.provenance)
        for i in e.indexes:
            iid = self.w.insert("indexes", entity_id=eid, name=i.name, columns_json=dumps(i.columns), is_unique=int(i.unique))
            self.w.evidence("index", iid, i.provenance)
        return eid

    def _profiles(self) -> None:
        change_ids: dict[tuple[str, int], int] = {}
        for comp, changes in self.ex.changes.items():
            for ch in changes:
                cid = self.w.insert("schema_changes", component=comp, ord=ch.order, changelog_path=ch.changelog_path,
                                    changeset_id=ch.changeset_id, author=ch.author, change_type=ch.change_type,
                                    table_name=ch.table, details_json=dumps(ch.details), dbms=",".join(ch.dbms),
                                    contexts=ch.contexts, labels=ch.labels, preconditions_json=dumps(ch.preconditions),
                                    on_fail=ch.on_fail, comment=ch.comment, line_start=ch.provenance.line_start,
                                    line_end=ch.provenance.line_end)
                self.w.evidence("schema_change", cid, ch.provenance)
                change_ids[(comp, ch.order)] = cid
        for p in self.ex.profiles:
            # Liquibase changelogs define the product schema; standalone SQL scripts get their own family
            family = "relational_sql_script" if p.component.startswith("sqlfile:") else "relational"
            is_default = int(family == "relational" and not p.is_test and p.dialect == self.s.default_dialect)
            pid = self.w.insert("schema_profiles", profile_key=p.profile_key, component=p.component, storage_family=family,
                                dialect=p.dialect, contexts=",".join(p.contexts), labels=",".join(p.labels),
                                parameters_json=dumps(p.parameters), is_test=int(p.is_test), is_default=is_default,
                                completeness_json=dumps(p.completeness))
            if family == "relational" and not p.is_test:
                self.default_profile_by_dialect[p.dialect].append(pid)
            for order, outcome in p.applied_changes:
                self.w.insert("schema_change_outcomes", profile_id=pid, change_id=change_ids[(p.component, order)], outcome=outcome)
            for e in p.entities:
                self._entity(pid, p.profile_key, e)
        # non-relational storages: one profile per (storage kind, component)
        groups: dict[tuple[str, str, bool], list[EntityDef]] = defaultdict(list)
        for e in self.ex.other_entities:
            groups[(e.storage_kind.value, e.component, e.is_test)].append(e)
        for (kind, comp, is_test), ents in groups.items():
            key = f"{kind}:{comp}"
            pid = self.w.insert("schema_profiles", profile_key=key, component=comp, storage_family=kind, dialect=None,
                                contexts="", labels="", parameters_json="{}", is_test=int(is_test), is_default=int(not is_test),
                                completeness_json=dumps({"level": ents[0].attributes.get("completeness", "complete")}))
            seen: set[str] = set()
            for e in ents:
                if e.name.upper() in seen:
                    self.ex.diagnostics.append(Diagnostic(path=e.provenance.path, extractor="catalog", code="duplicate_entity",
                                                          message=f"{kind} {e.name} defined twice in {comp}; first kept"))
                    continue
                seen.add(e.name.upper())
                self._entity(pid, key, e)

    def _resolve_entity(self, table: str, dialect: str | None) -> int | None:
        name = table.upper()
        if self.s.table_prefix and name.startswith(self.s.table_prefix.upper()):
            name = name[len(self.s.table_prefix):]
        pids = self.default_profile_by_dialect.get(dialect or self.s.default_dialect, [])
        hits = [self.entity_ids[(pid, name)] for pid in pids if (pid, name) in self.entity_ids]
        return hits[0] if len(hits) == 1 else None

    def _queries(self) -> dict[str, int]:
        ids: dict[str, int] = {}
        for qi, q in enumerate(self.ex.queries):
            if q.key in ids:
                q = q.model_copy(update={"key": f"{q.key}#{qi}"})
            p = q.provenance
            qid = self.w.insert("queries", query_key=q.key, language=q.language, origin=q.origin, statement_kind=q.statement_kind,
                                namespace=q.namespace, statement_id=q.statement_id, database_id=q.database_id,
                                result_map=q.result_map, parameter_type=q.parameter_type, result_type=q.result_type,
                                template_json=dumps(q.template), raw_text=q.raw_text, description=q.description,
                                is_test=int(q.is_test), path=p.path, line_start=p.line_start, line_end=p.line_end, symbol=p.symbol,
                                diagnostics_json=dumps(q.diagnostics))
            self.w.evidence("query", qid, p)
            ids[q.key] = qid
            for vi, v in enumerate(q.variants):
                a = self.ex.analyses.get((qi, vi))
                if q.language != "sql":
                    status = f"not_sql:{q.language}"
                elif v.partial:
                    status = "partial"
                else:
                    status = a.status if a else "not_analysed"
                vid = self.w.insert(
                    "query_variants", query_id=qid, ordinal=vi, dialect=v.dialect, dialect_evidence=v.dialect_evidence,
                    conditions_json=dumps(v.conditions), sql_text=v.sql, partial=int(v.partial), unresolved_json=dumps(v.unresolved),
                    parse_status=status, statement_type=a.statement_type if a else None, normalized_sql=a.normalized if a else None,
                    fingerprint=a.fingerprint if a else None, ast_json=a.ast_json if a else None, parser_version=PARSER if a else None,
                    analysis_json=dumps(a.model_dump(exclude={"ast_json", "normalized"})) if a else None,
                    diagnostics_json=dumps(a.diagnostics) if a else None)
                for pos, prm in enumerate(v.parameters):
                    self.w.insert("query_parameters", variant_id=vid, position=pos, name=prm.name, kind=prm.kind,
                                  java_type=prm.java_type, jdbc_type=prm.jdbc_type, type_handler=prm.type_handler,
                                  resolved_value=prm.resolved_value)
                if not a or a.status != "parsed":
                    continue
                dialect = v.dialect or self.s.default_dialect
                for t in a.tables:
                    if t.via == "cte":
                        continue
                    self.w.insert("query_table_refs", query_id=qid, variant_id=vid, table_name=t.name, table_upper=t.name.upper(),
                                  entity_id=self._resolve_entity(t.name, dialect), schema_name=t.schema_name, alias=t.alias,
                                  access=t.access, via=t.via)
                for c in a.columns:
                    self.w.insert("query_column_refs", query_id=qid, variant_id=vid, table_name=c.table,
                                  entity_id=self._resolve_entity(c.table, dialect) if c.table else None, column_name=c.column,
                                  column_upper=c.column.upper(), clause=c.clause, resolved=int(c.resolved))
        return ids

    def _symbols_and_mappings(self, query_ids: dict[str, int]) -> None:
        sym_ids: dict[str, int] = {}
        for s in self.ex.symbols:
            p = s.provenance
            sid = self.w.insert("symbols", kind=s.kind, fqn=s.fqn, name=s.name, parent_fqn=s.attributes.get("parent"),
                                javadoc=s.javadoc, comment=s.comment, signature=s.signature, attributes_json=dumps(s.attributes),
                                is_test=int(s.is_test), path=p.path, line_start=p.line_start, line_end=p.line_end)
            self.w.evidence("symbol", sid, p)
            sym_ids.setdefault(s.fqn, sid)
        db = self.w.db
        # mapper method <-> statement (namespace = interface FQN, id = method name)
        for q in self.ex.queries:
            if q.origin != "mybatis" or q.statement_kind == "sql_fragment":
                continue
            sid = sym_ids.get(f"{q.namespace}#{q.statement_id}")
            qid = query_ids.get(q.key)
            if sid and qid:
                db.execute("UPDATE queries SET mapper_method_symbol_id=? WHERE id=?", (sid, qid))
                mid = self.w.insert("mappings", kind="mapper_method", source=f"{q.namespace}#{q.statement_id}", query_id=qid,
                                    symbol_id=sid, fact_status="recovered", confidence=1.0, path=q.provenance.path,
                                    line=q.provenance.line_start)
                self.w.evidence("mapping", mid, q.provenance.model_copy(update={"evidence_kind": "mybatis_namespace_method_match"}),
                                status="recovered")
        # resultMap columns -> entity columns. First choice: projection lineage of the statements using the map
        # (SELECT o.ORDER_KEY ... -> ORDERS.ORDER_KEY); fallback: the only table read by them that has the column.
        rm_tables: dict[str, set[int]] = defaultdict(set)
        rm_lineage: dict[tuple[str, str], set[int]] = defaultdict(set)
        rows = db.execute("""SELECT q.namespace, q.result_map, v.dialect, v.analysis_json FROM queries q
                             JOIN query_variants v ON v.query_id=q.id
                             WHERE q.result_map IS NOT NULL AND v.parse_status='parsed'""").fetchall()
        for ns, rm, dialect, analysis_json in rows:
            rm_key = rm if "." in rm else f"{ns}.{rm}"
            analysis = json.loads(analysis_json)
            for t in analysis["tables"]:
                eid = self._resolve_entity(t["name"], dialect)
                if eid is not None:
                    rm_tables[rm_key].add(eid)
            for item in analysis["lineage"]:
                if len(item["sources"]) != 1:
                    continue
                table, col = item["sources"][0].rsplit(".", 1)
                eid = self._resolve_entity(table, dialect) if table != "?" else None
                if eid is not None and (eid, col.upper()) in self.column_ids:
                    rm_lineage[(rm_key, item["output"].upper())].add(eid)
        written: set[tuple[int, str, str]] = set()
        for m in self.ex.result_maps:
            via_lineage = rm_lineage.get((m.result_map, m.column.upper()), set())
            candidates = via_lineage or {eid for eid in rm_tables.get(m.result_map, set()) if (eid, m.column.upper()) in self.column_ids}
            by_profile: dict[int, list[int]] = defaultdict(list)  # the same table exists once per dialect profile
            for eid in candidates:
                by_profile[self.entity_rows[eid]["profile_id"]].append(eid)
            for owners in by_profile.values():
                if len(owners) != 1:
                    continue  # still ambiguous: no mapping rather than a guess
                key = (owners[0], m.column.upper(), m.property)
                if key in written:
                    continue
                written.add(key)
                confidence = 0.95 if via_lineage else 0.8
                mid = self.w.insert("mappings", kind="result_map", source=m.result_map, entity_id=owners[0], column_name=m.column,
                                    java_type=m.java_type, property=m.property, symbol_id=sym_ids.get(m.java_type or ""),
                                    fact_status="recovered", confidence=confidence, path=m.provenance.path,
                                    line=m.provenance.line_start)
                self.w.evidence("mapping", mid, m.provenance, status="recovered", confidence=confidence)

    def _relationships(self) -> None:
        db = self.w.db
        for eid, row in self.entity_rows.items():
            for k in row["def"].constraints:
                if k.kind != "foreign_key" or not k.ref_table:
                    continue
                to = self.entity_ids.get((row["profile_id"], k.ref_table.upper()))
                rid = self.w.insert("relationships", profile_id=row["profile_id"], kind="declared_fk", from_entity_id=eid,
                                    from_columns_json=dumps(k.columns), to_entity_id=to, to_table=k.ref_table,
                                    to_columns_json=dumps(k.ref_columns), evidence_count=1, fact_status="declared",
                                    details_json=dumps({"constraint": k.name}))
                self.w.evidence("relationship", rid, k.provenance, status="declared")
        joins: dict[tuple[int, str, int, str], set[int]] = defaultdict(set)
        for qid, analysis_json, dialect in db.execute(
                "SELECT query_id, analysis_json, dialect FROM query_variants WHERE parse_status='parsed'"):
            for j in json.loads(analysis_json).get("joins", []):
                if not (j["left_table"] and j["right_table"] and j["left_column"] and j["right_column"]):
                    continue
                a = self._resolve_entity(j["left_table"], dialect)
                b = self._resolve_entity(j["right_table"], dialect)
                if a is None or b is None or a == b:
                    continue
                ends = sorted([(a, j["left_column"].upper()), (b, j["right_column"].upper())])
                joins[(ends[0][0], ends[0][1], ends[1][0], ends[1][1])].add(qid)
        for (a, ca, b, cb), qids in joins.items():
            rid = self.w.insert("relationships", profile_id=self.entity_rows[a]["profile_id"], kind="inferred_join",
                                from_entity_id=a, from_columns_json=dumps([ca]), to_entity_id=b,
                                to_table=self.entity_rows[b]["name"], to_columns_json=dumps([cb]), evidence_count=len(qids),
                                fact_status="recovered", details_json=dumps({"query_ids": sorted(qids)[:50]}))
            for qid in sorted(qids)[:5]:
                path, ls, le, sym = db.execute("SELECT path, line_start, line_end, symbol FROM queries WHERE id=?", (qid,)).fetchone()
                self.w.evidence("relationship", rid, Provenance(path=path, line_start=ls, line_end=le, symbol=sym,
                                                                extractor="sqlglot", evidence_kind="join_condition", confidence=0.8),
                                status="recovered")

    def _purposes(self) -> None:
        db = self.w.db
        javadoc_by_type = {s.fqn: s for s in self.ex.symbols if s.kind in {"class", "record", "interface", "enum"}}
        create_comments: dict[tuple[str, str], str] = {}
        for comp, changes in self.ex.changes.items():
            for ch in changes:
                if ch.change_type == "createTable" and ch.comment and ch.table:
                    create_comments[(comp, ch.table.upper().replace("${PREFIX}", ""))] = ch.comment
        for eid, row in self.entity_rows.items():
            e: EntityDef = row["def"]
            sources: list[dict[str, Any]] = []
            if e.remarks:
                sources.append({"status": "declared", "kind": "remarks/comment", "text": e.remarks, "path": e.provenance.path})
            cc = create_comments.get((e.component, e.name.upper()))
            if cc:
                sources.append({"status": "declared", "kind": "changelog comment", "text": cc, "path": e.provenance.path})
            for (java_type,) in db.execute("SELECT DISTINCT java_type FROM mappings WHERE entity_id=? AND kind='result_map'", (eid,)):
                sym = javadoc_by_type.get(java_type or "")
                if sym and sym.javadoc:
                    sources.append({"status": "linked", "kind": f"javadoc of mapped type {java_type}", "text": sym.javadoc,
                                    "path": sym.provenance.path})
            usage = self._usage_summary(eid, row)
            primary = sources[0] if sources else None
            db.execute("UPDATE entities SET purpose=?, purpose_status=?, purpose_sources_json=?, usage_summary=? WHERE id=?",
                       (primary["text"] if primary else NOT_ESTABLISHED, primary["status"] if primary else "not_established",
                        dumps(sources), usage, eid))
        # column purposes: remarks, else javadoc/comment of mapped Java field
        fields: dict[tuple[str, str], str] = {}
        for s in javadoc_by_type.values():
            for f in s.attributes.get("fields", []):
                txt = f.get("javadoc") or f.get("comment")
                if txt:
                    fields[(s.fqn, f["name"])] = txt
        for cid, remarks, eid, cname in db.execute("SELECT id, remarks, entity_id, name_upper FROM columns").fetchall():
            if remarks:
                db.execute("UPDATE columns SET purpose=?, purpose_status='declared' WHERE id=?", (remarks, cid))
                continue
            for java_type, prop in db.execute("SELECT java_type, property FROM mappings WHERE entity_id=? AND column_name=? COLLATE NOCASE",
                                              (eid, cname)):
                txt = fields.get((java_type or "", (prop or "").split(".")[-1]))
                if txt:
                    db.execute("UPDATE columns SET purpose=?, purpose_status='linked' WHERE id=?", (txt, cid))
                    break
            else:
                db.execute("UPDATE columns SET purpose=?, purpose_status='not_established' WHERE id=?", (NOT_ESTABLISHED, cid))

    def _usage_summary(self, eid: int, row: dict[str, Any]) -> str | None:
        if row["kind"] != StorageKind.relational_table.value:
            return None
        db = self.w.db
        acc = dict(db.execute("""SELECT access, COUNT(DISTINCT query_id) FROM query_table_refs WHERE entity_id=? GROUP BY access""",
                              (eid,)).fetchall())
        if not acc:
            return "Вывод по использованию: запросы к таблице в проиндексированных источниках не найдены."
        mappers = [r[0] for r in db.execute("""SELECT DISTINCT q.namespace FROM query_table_refs r JOIN queries q ON q.id=r.query_id
                                               WHERE r.entity_id=? AND q.namespace IS NOT NULL LIMIT 5""", (eid,))]
        filters = [r[0] for r in db.execute("""SELECT column_name FROM query_column_refs WHERE entity_id=? AND clause='where'
                                               GROUP BY column_upper ORDER BY COUNT(*) DESC LIMIT 6""", (eid,))]
        joined = [r[0] for r in db.execute("""SELECT DISTINCT e.name FROM relationships r JOIN entities e
                                              ON e.id = CASE WHEN r.from_entity_id=? THEN r.to_entity_id ELSE r.from_entity_id END
                                              WHERE (r.from_entity_id=? OR r.to_entity_id=?) LIMIT 8""", (eid, eid, eid))]
        parts = [f"читается в {acc.get('read', 0)} и изменяется в {acc.get('write', 0)} запросах"]
        if mappers:
            parts.append("мапперы: " + ", ".join(m.rsplit(".", 1)[-1] for m in mappers))
        if filters:
            parts.append("частые фильтры: " + ", ".join(filters))
        if joined:
            parts.append("связана с: " + ", ".join(joined))
        return "Вывод по использованию (детерминированно, не заявлено в источниках): " + "; ".join(parts) + "."


# ------------------------------------------------------------------ retrieval documents


def _documents(db) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    ents = db.execute("""SELECT e.*, p.is_default FROM entities e JOIN schema_profiles p ON p.id=e.profile_id
                         WHERE p.is_default=1 OR p.is_test=1""").fetchall()
    for e in ents:
        cols = db.execute("SELECT name, data_type, purpose, purpose_status FROM columns WHERE entity_id=? ORDER BY position",
                          (e["id"],)).fetchall()
        idx = db.execute("SELECT name, columns_json FROM indexes WHERE entity_id=?", (e["id"],)).fetchall()
        header = [f"{e['storage_kind']} {e['name']} (component {e['component']})",
                  f"Назначение: {e['purpose']}" + ("" if e["purpose_status"] in {"declared", "linked"} else ""),
                  e["usage_summary"] or ""]
        col_lines = [f"- {c['name']} {c['data_type'] or ''}" + (f": {c['purpose']}" if c["purpose_status"] != "not_established" else "")
                     for c in cols]
        small = len(cols) <= 40
        body = header + (["Колонки:"] + col_lines if small else [f"Колонок: {len(cols)} (см. группы колонок)"]) + \
            ([f"Индексы: " + "; ".join(f"{i['name']}({', '.join(json.loads(i['columns_json']))})" for i in idx)] if idx else [])
        docs.append({"doc_key": f"entity:{e['id']}", "doc_type": "entity", "title": f"{e['storage_kind']} {e['name']}",
                     "subject_kind": "entity", "subject_id": e["id"], "storage_kind": e["storage_kind"], "component": e["component"],
                     "is_test": e["is_test"], "path": e["path"], "line_start": e["line_start"], "line_end": e["line_end"],
                     "units": [x for x in body if x], "identifiers": [e["name"]] + [c["name"] for c in cols]})
        if not small:
            for g in range(0, len(cols), 25):
                group = cols[g:g + 25]
                docs.append({"doc_key": f"entity:{e['id']}:cols:{g}", "doc_type": "column_group",
                             "title": f"{e['name']} columns {g + 1}-{g + len(group)}", "subject_kind": "entity",
                             "subject_id": e["id"], "storage_kind": e["storage_kind"], "component": e["component"],
                             "is_test": e["is_test"], "path": e["path"], "line_start": e["line_start"], "line_end": e["line_end"],
                             "units": [f"{e['storage_kind']} {e['name']} — группа колонок"] + col_lines[g:g + 25],
                             "identifiers": [e["name"]] + [c["name"] for c in group]})
    for r in db.execute("""SELECT r.*, a.name AS a_name, b.name AS b_name, a.storage_kind, a.component, a.path
                           FROM relationships r JOIN entities a ON a.id=r.from_entity_id LEFT JOIN entities b ON b.id=r.to_entity_id
                           JOIN schema_profiles p ON p.id=a.profile_id WHERE p.is_default=1"""):
        fc, tc = json.loads(r["from_columns_json"]), json.loads(r["to_columns_json"])
        label = "объявленный внешний ключ" if r["kind"] == "declared_fk" else f"соединение JOIN в {r['evidence_count']} запросах (не FK)"
        docs.append({"doc_key": f"relationship:{r['id']}", "doc_type": "relationship",
                     "title": f"{r['a_name']}.{','.join(fc)} -> {r['b_name'] or r['to_table']}.{','.join(tc)}",
                     "subject_kind": "relationship", "subject_id": r["id"], "storage_kind": r["storage_kind"],
                     "component": r["component"], "is_test": 0, "path": r["path"], "line_start": None, "line_end": None,
                     "units": [f"Связь {r['a_name']}.{', '.join(fc)} → {r['b_name'] or r['to_table']}.{', '.join(tc)}: {label}"],
                     "identifiers": [r["a_name"], r["b_name"] or r["to_table"]] + fc + tc})
    for q in db.execute("SELECT * FROM queries"):
        v = db.execute("""SELECT * FROM query_variants WHERE query_id=? ORDER BY (dialect IS NULL), (parse_status!='parsed'), ordinal
                          LIMIT 1""", (q["id"],)).fetchone()
        tables = [r[0] for r in db.execute("SELECT DISTINCT table_name || ' (' || access || ')' FROM query_table_refs WHERE query_id=?",
                                           (q["id"],))]
        sql = (v["normalized_sql"] or v["sql_text"]) if v else re.sub(r"\s+", " ", q["raw_text"])[:3000]
        title = q["symbol"] or q["query_key"]
        units = [f"{q['language']} запрос {q['query_key']} ({q['statement_kind']}, {q['origin']})"]
        if q["description"]:
            units.append(f"Комментарий: {q['description']}")
        if tables:
            units.append("Таблицы: " + ", ".join(tables))
        units.extend(_sql_units(sql))
        docs.append({"doc_key": f"query:{q['id']}", "doc_type": "query", "title": title, "subject_kind": "query",
                     "subject_id": q["id"], "storage_kind": "relational_table" if q["language"] == "sql" else q["language"],
                     "component": q["path"].split("/src/", 1)[0], "is_test": q["is_test"], "path": q["path"],
                     "line_start": q["line_start"], "line_end": q["line_end"], "units": units,
                     "identifiers": [q["namespace"] or "", q["statement_id"] or "", q["symbol"] or ""] + [t.split(" ")[0] for t in tables]})
    for s in db.execute("SELECT * FROM symbols WHERE kind IN ('method','class','record','interface','enum')"):
        attrs = json.loads(s["attributes_json"] or "{}")
        if s["kind"] == "method":
            stmt = db.execute("SELECT query_key, id FROM queries WHERE mapper_method_symbol_id=?", (s["id"],)).fetchall()
            tables = sorted({r[0] for q in stmt for r in db.execute("SELECT DISTINCT table_name FROM query_table_refs WHERE query_id=?", (q[1],))})
            units = [f"Mapper-метод {s['fqn']}", f"Сигнатура: {s['signature']}"]
            if s["javadoc"]:
                units.append(s["javadoc"])
            if stmt:
                units.append("MyBatis statements: " + ", ".join(q[0] for q in stmt))
            if tables:
                units.append("Таблицы: " + ", ".join(tables))
            ids = [s["fqn"], s["name"]] + tables
        else:
            fields = attrs.get("fields", [])
            units = [f"Java {s['kind']} {s['fqn']}"] + ([s["javadoc"]] if s["javadoc"] else []) + \
                [f"- {f['name']}: {f['java_type']}" + (f" — {f['javadoc']}" if f.get("javadoc") else "") for f in fields]
            if attrs.get("enum_constants"):
                units += [f"- {c['name']}({', '.join(c.get('arguments', []))})" + (f" // {c['comment']}" if c.get("comment") else "")
                          for c in attrs["enum_constants"]]
            ids = [s["fqn"], s["name"]] + [f["name"] for f in fields]
        docs.append({"doc_key": f"symbol:{s['id']}", "doc_type": "mapper_method" if s["kind"] == "method" else "java_type",
                     "title": s["fqn"], "subject_kind": "symbol", "subject_id": s["id"], "storage_kind": None,
                     "component": s["path"].split("/src/", 1)[0], "is_test": s["is_test"], "path": s["path"],
                     "line_start": s["line_start"], "line_end": s["line_end"], "units": units, "identifiers": ids})
    return docs


def _sql_units(sql: str) -> list[str]:
    # break long SQL at clause keywords so chunks follow statement structure
    parts = re.split(r"\s(?=(?:FROM|WHERE|LEFT JOIN|JOIN|GROUP BY|ORDER BY|UNION|SET|VALUES|HAVING)\b)", sql)
    return [p for p in parts if p.strip()]


def build_retrieval(settings: Settings, w: CatalogWriter, gdir: Path, embed: bool, progress, doc_sections=None,
                    embedder=None) -> dict[str, Any]:
    from camunda_rag.embeddings import Embedder

    db = w.db
    db.row_factory = sqlite3.Row
    docs = _documents(db)
    for n, sec in enumerate(doc_sections or []):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec.text) if p.strip()]
        docs.append({"doc_key": f"doc:{sec.path}:{sec.line_start}:{n}", "doc_type": "documentation",
                     "title": f"{sec.path} — {sec.heading}", "subject_kind": "doc", "subject_id": None, "storage_kind": None,
                     "component": sec.path.split("/", 1)[0], "is_test": int(sec.role == FileRole.test), "path": sec.path,
                     "line_start": sec.line_start, "line_end": sec.line_end, "units": [sec.heading] + paragraphs,
                     "identifiers": [sec.heading]})
    db.row_factory = None
    emb = embedder or Embedder(settings.embedding, load_model=embed)
    texts, chunk_ids, split_notes = [], [], 0
    for d in docs:
        did = w.insert("retrieval_documents", doc_key=d["doc_key"], doc_type=d["doc_type"], title=d["title"],
                       subject_kind=d["subject_kind"], subject_id=d["subject_id"], storage_kind=d["storage_kind"],
                       component=d["component"], is_test=int(d["is_test"] or 0), path=d["path"], line_start=d["line_start"],
                       line_end=d["line_end"], text="\n".join(d["units"]), identifiers=identifiers_text(d["identifiers"]))
        reserve = emb.count(d["title"]) + 1  # continuation chunks repeat the title
        for i, ch in enumerate(emb.chunk(d["units"], reserve=reserve)):
            ident = identifiers_text(d["identifiers"]) if i == 0 else identifiers_text([d["title"]])
            text = ch.text if i == 0 else f"{d['title']}\n{ch.text}"
            cid = w.insert("chunks", document_id=did, ordinal=i, text=text, token_count=emb.count(text), identifiers=ident)
            split_notes += int(ch.split_note is not None)
            texts.append(text)
            chunk_ids.append(cid)
    w.rebuild_fts()
    stats: dict[str, Any] = {"documents": len(docs), "chunks": len(chunk_ids), "oversized_units_split": split_notes}
    if embed:
        import faiss
        import numpy as np

        progress(f"embedding {len(texts)} chunks with {emb.name} on {emb.device}")
        vecs = emb.embed(texts, show_progress=True)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb.dim))
        index.add_with_ids(vecs, np.asarray(chunk_ids, dtype="int64"))
        faiss.write_index(index, str(gdir / "vectors.faiss"))
        db.executemany("INSERT INTO embedding_records VALUES (?, ?, ?, ?, 1)", [(c, c, emb.name, emb.dim) for c in chunk_ids])
        stats.update(embedding_model=emb.name, embedding_dim=emb.dim, embedding_device=emb.device)
    else:
        stats.update(embedding_model=None, embedding_dim=None)
    return stats
