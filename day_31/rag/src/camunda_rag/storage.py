"""SQLite catalog: normalized schema, provenance, FTS5 and generation publishing.

A generation is built in data/generations/<id>/ (catalog.sqlite + vectors.faiss) and published
by atomically replacing data/CURRENT. Readers open the published generation read-only, so all
operations of a request see one snapshot.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from camunda_rag.models import Provenance

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE index_generations (
  id TEXT PRIMARY KEY, created_at TEXT, status TEXT, commit_sha TEXT, settings_json TEXT, stats_json TEXT,
  embedding_model TEXT, embedding_dim INTEGER, catalog_schema_version INTEGER);
CREATE TABLE snapshots (id INTEGER PRIMARY KEY, repository TEXT, commit_sha TEXT, generation_id TEXT, indexed_at TEXT);
CREATE TABLE source_files (
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, language TEXT, role TEXT, status TEXT, reason TEXT, extractors TEXT, size INTEGER);
CREATE TABLE evidence (
  id INTEGER PRIMARY KEY, subject_kind TEXT, subject_id INTEGER, repository TEXT, commit_sha TEXT, path TEXT,
  line_start INTEGER, line_end INTEGER, symbol TEXT, extractor TEXT, evidence_kind TEXT, confidence REAL, fact_status TEXT);
CREATE INDEX evidence_subject ON evidence(subject_kind, subject_id);
CREATE TABLE diagnostics (
  id INTEGER PRIMARY KEY, path TEXT, line INTEGER, extractor TEXT, severity TEXT, code TEXT, message TEXT, subject TEXT);
CREATE INDEX diagnostics_path ON diagnostics(path);
CREATE TABLE schema_profiles (
  id INTEGER PRIMARY KEY, profile_key TEXT UNIQUE, component TEXT, storage_family TEXT, dialect TEXT, contexts TEXT,
  labels TEXT, parameters_json TEXT, is_test INTEGER, is_default INTEGER, completeness_json TEXT);
CREATE TABLE entities (
  id INTEGER PRIMARY KEY, entity_key TEXT UNIQUE, profile_id INTEGER REFERENCES schema_profiles(id), storage_kind TEXT,
  component TEXT, schema_name TEXT, name TEXT, name_upper TEXT, remarks TEXT, purpose TEXT, purpose_status TEXT,
  purpose_sources_json TEXT, usage_summary TEXT, is_test INTEGER, attributes_json TEXT, completeness TEXT,
  path TEXT, line_start INTEGER, line_end INTEGER);
CREATE INDEX entities_name ON entities(name_upper);
CREATE TABLE columns (
  id INTEGER PRIMARY KEY, entity_id INTEGER REFERENCES entities(id), name TEXT, name_upper TEXT, position INTEGER,
  data_type TEXT, type_source TEXT, nullable INTEGER, default_value TEXT, primary_key INTEGER, remarks TEXT,
  purpose TEXT, purpose_status TEXT, fact_status TEXT, history_json TEXT, path TEXT, line_start INTEGER);
CREATE INDEX columns_entity ON columns(entity_id);
CREATE INDEX columns_name ON columns(name_upper);
CREATE TABLE constraints (
  id INTEGER PRIMARY KEY, entity_id INTEGER REFERENCES entities(id), kind TEXT, name TEXT, columns_json TEXT,
  ref_table TEXT, ref_columns_json TEXT);
CREATE TABLE indexes (
  id INTEGER PRIMARY KEY, entity_id INTEGER REFERENCES entities(id), name TEXT, columns_json TEXT, is_unique INTEGER);
CREATE TABLE schema_changes (
  id INTEGER PRIMARY KEY, component TEXT, ord INTEGER, changelog_path TEXT, changeset_id TEXT, author TEXT,
  change_type TEXT, table_name TEXT, details_json TEXT, dbms TEXT, contexts TEXT, labels TEXT, preconditions_json TEXT,
  on_fail TEXT, comment TEXT, line_start INTEGER, line_end INTEGER);
CREATE TABLE schema_change_outcomes (
  profile_id INTEGER REFERENCES schema_profiles(id), change_id INTEGER REFERENCES schema_changes(id), outcome TEXT,
  PRIMARY KEY (profile_id, change_id));
CREATE TABLE relationships (
  id INTEGER PRIMARY KEY, profile_id INTEGER, kind TEXT, from_entity_id INTEGER, from_columns_json TEXT,
  to_entity_id INTEGER, to_table TEXT, to_columns_json TEXT, evidence_count INTEGER, fact_status TEXT, details_json TEXT);
CREATE INDEX relationships_from ON relationships(from_entity_id);
CREATE INDEX relationships_to ON relationships(to_entity_id);
CREATE TABLE queries (
  id INTEGER PRIMARY KEY, query_key TEXT UNIQUE, language TEXT, origin TEXT, statement_kind TEXT, namespace TEXT,
  statement_id TEXT, database_id TEXT, result_map TEXT, parameter_type TEXT, result_type TEXT, template_json TEXT,
  raw_text TEXT, description TEXT, is_test INTEGER, path TEXT, line_start INTEGER, line_end INTEGER, symbol TEXT,
  diagnostics_json TEXT, mapper_method_symbol_id INTEGER);
CREATE TABLE query_variants (
  id INTEGER PRIMARY KEY, query_id INTEGER REFERENCES queries(id), ordinal INTEGER, dialect TEXT, dialect_evidence TEXT,
  conditions_json TEXT, sql_text TEXT, partial INTEGER, unresolved_json TEXT, parse_status TEXT, statement_type TEXT,
  normalized_sql TEXT, fingerprint TEXT, ast_json TEXT, parser_version TEXT, analysis_json TEXT, diagnostics_json TEXT);
CREATE INDEX query_variants_query ON query_variants(query_id);
CREATE TABLE query_parameters (
  id INTEGER PRIMARY KEY, variant_id INTEGER REFERENCES query_variants(id), position INTEGER, name TEXT, kind TEXT,
  java_type TEXT, jdbc_type TEXT, type_handler TEXT, resolved_value TEXT);
CREATE TABLE query_table_refs (
  id INTEGER PRIMARY KEY, query_id INTEGER, variant_id INTEGER, table_name TEXT, table_upper TEXT, entity_id INTEGER,
  schema_name TEXT, alias TEXT, access TEXT, via TEXT);
CREATE INDEX qtr_entity ON query_table_refs(entity_id);
CREATE INDEX qtr_table ON query_table_refs(table_upper);
CREATE TABLE query_column_refs (
  id INTEGER PRIMARY KEY, query_id INTEGER, variant_id INTEGER, table_name TEXT, entity_id INTEGER, column_name TEXT,
  column_upper TEXT, clause TEXT, resolved INTEGER);
CREATE INDEX qcr_entity ON query_column_refs(entity_id, column_upper);
CREATE TABLE symbols (
  id INTEGER PRIMARY KEY, kind TEXT, fqn TEXT, name TEXT, parent_fqn TEXT, javadoc TEXT, comment TEXT, signature TEXT,
  attributes_json TEXT, is_test INTEGER, path TEXT, line_start INTEGER, line_end INTEGER);
CREATE INDEX symbols_fqn ON symbols(fqn);
CREATE TABLE mappings (
  id INTEGER PRIMARY KEY, kind TEXT, source TEXT, entity_id INTEGER, column_name TEXT, java_type TEXT, property TEXT,
  query_id INTEGER, symbol_id INTEGER, fact_status TEXT, confidence REAL, path TEXT, line INTEGER);
CREATE INDEX mappings_entity ON mappings(entity_id);
CREATE TABLE retrieval_documents (
  id INTEGER PRIMARY KEY, doc_key TEXT UNIQUE, doc_type TEXT, title TEXT, subject_kind TEXT, subject_id INTEGER,
  storage_kind TEXT, component TEXT, is_test INTEGER, path TEXT, line_start INTEGER, line_end INTEGER, text TEXT,
  identifiers TEXT);
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, document_id INTEGER REFERENCES retrieval_documents(id), ordinal INTEGER, text TEXT,
  token_count INTEGER, identifiers TEXT);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text, identifiers, content='chunks', content_rowid='id', tokenize="unicode61 remove_diacritics 2 tokenchars '_'");
CREATE TABLE embedding_records (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id), faiss_id INTEGER, model TEXT, dim INTEGER, normalized INTEGER);
"""


def dumps(v: Any) -> str | None:
    return None if v is None else json.dumps(v, ensure_ascii=False, default=str)


class CatalogWriter:
    def __init__(self, path: Path, repository: str, commit_sha: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.executescript(DDL)
        self.db.execute("INSERT INTO meta VALUES ('catalog_schema_version', ?)", (str(SCHEMA_VERSION),))
        self.repository = repository
        self.commit_sha = commit_sha

    def insert(self, table: str, **values: Any) -> int:
        cols = ", ".join(values)
        qs = ", ".join("?" for _ in values)
        cur = self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(values.values()))
        return int(cur.lastrowid)

    def evidence(self, subject_kind: str, subject_id: int, p: Provenance, status: str | None = None,
                 confidence: float | None = None) -> None:
        self.insert("evidence", subject_kind=subject_kind, subject_id=subject_id, repository=self.repository,
                    commit_sha=self.commit_sha, path=p.path, line_start=p.line_start, line_end=p.line_end, symbol=p.symbol,
                    extractor=p.extractor, evidence_kind=p.evidence_kind,
                    confidence=p.confidence if confidence is None else confidence,
                    fact_status=status or p.status.value)

    def rebuild_fts(self) -> None:
        self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

    def close(self) -> None:
        self.db.commit()
        self.db.close()


# ------------------------------------------------------------------ generations


def generations_dir(data_dir: Path) -> Path:
    return data_dir / "generations"


def current_generation(data_dir: Path) -> str | None:
    p = data_dir / "CURRENT"
    return p.read_text().strip() if p.exists() else None


def publish(data_dir: Path, generation_id: str) -> None:
    tmp = data_dir / "CURRENT.tmp"
    tmp.write_text(generation_id)
    os.replace(tmp, data_dir / "CURRENT")


def open_readonly(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db
