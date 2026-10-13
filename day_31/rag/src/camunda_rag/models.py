"""Intermediate data model shared by extractors, schema reconstruction and storage.

Every fact carries a Provenance. `FactStatus` separates what the sources declare from what we
reconstruct, assume or could not establish.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    declared = "declared"  # stated verbatim by a source (DDL, changelog, mapping JSON, javadoc)
    recovered = "recovered"  # derived deterministically (changelog replay, SQL AST, include resolution)
    assumed = "assumed"  # heuristic link (name match, prefix substitution)
    unknown = "unknown"  # could not be established


class FileRole(str, Enum):
    main = "main"
    test = "test"
    doc = "doc"
    build = "build"


class FileStatus(str, Enum):
    processed = "processed"
    irrelevant = "irrelevant"
    excluded = "excluded"
    unsupported = "unsupported"


class StorageKind(str, Enum):
    relational_table = "relational_table"
    elasticsearch_index = "elasticsearch_index"
    opensearch_index = "opensearch_index"
    elasticsearch_index_template = "elasticsearch_index_template"
    opensearch_index_template = "opensearch_index_template"
    rocksdb_column_family = "rocksdb_column_family"


RELATIONAL_KINDS = {StorageKind.relational_table.value}


class Provenance(BaseModel):
    path: str
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    extractor: str
    evidence_kind: str  # e.g. liquibase_createTable, mybatis_statement, javadoc, es_mapping
    confidence: float = 1.0
    status: FactStatus = FactStatus.declared


class Diagnostic(BaseModel):
    path: str | None = None
    line: int | None = None
    extractor: str
    severity: str = "warning"  # info | warning | error
    code: str
    message: str
    subject: str | None = None


class SourceFile(BaseModel):
    path: str
    language: str
    role: FileRole
    status: FileStatus
    reason: str = ""
    extractors: list[str] = Field(default_factory=list)
    size: int = 0


# ---------------------------------------------------------------- schema (changelog level)


class ChangeRecord(BaseModel):
    """One Liquibase change (or one statement of a raw SQL change) in changelog order."""

    order: int
    component: str
    changelog_path: str
    changeset_id: str
    author: str | None = None
    change_type: str  # createTable, addColumn, ..., sql, unknown:<tag>
    table: str | None = None  # name with unresolved ${...} kept
    details: dict[str, Any] = Field(default_factory=dict)
    dbms: list[str] = Field(default_factory=list)  # empty = all
    contexts: str | None = None  # Liquibase context expression
    labels: str | None = None
    preconditions: dict[str, Any] | None = None
    on_fail: str | None = None
    comment: str | None = None
    provenance: Provenance


class ChangelogProperty(BaseModel):
    name: str
    value: str
    dbms: list[str] = Field(default_factory=list)
    context: str | None = None
    path: str


# ---------------------------------------------------------------- reconstructed schema


class ColumnDef(BaseModel):
    name: str
    data_type: str | None = None
    type_source: str = "unknown"  # liquibase_type | sql_ddl | es_mapping | java_field
    nullable: bool | None = None
    default: str | None = None
    primary_key: bool = False
    position: int = 0
    remarks: str | None = None
    status: FactStatus = FactStatus.declared
    provenance: Provenance
    history: list[str] = Field(default_factory=list)  # change descriptions in order


class ConstraintDef(BaseModel):
    kind: str  # primary_key | unique | foreign_key | not_null
    name: str | None = None
    columns: list[str]
    ref_table: str | None = None
    ref_columns: list[str] = Field(default_factory=list)
    provenance: Provenance


class IndexDef(BaseModel):
    name: str
    columns: list[str]
    unique: bool = False
    provenance: Provenance


class EntityDef(BaseModel):
    storage_kind: StorageKind
    component: str
    schema_name: str | None = None
    name: str
    columns: list[ColumnDef] = Field(default_factory=list)
    constraints: list[ConstraintDef] = Field(default_factory=list)
    indexes: list[IndexDef] = Field(default_factory=list)
    remarks: str | None = None
    provenance: Provenance
    is_test: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


class SchemaProfileResult(BaseModel):
    profile_key: str
    component: str
    dialect: str
    contexts: list[str]
    labels: list[str]
    parameters: dict[str, str]
    is_test: bool
    entities: list[EntityDef]
    applied_changes: list[tuple[int, str]]  # (order, outcome)
    completeness: dict[str, Any]


# ---------------------------------------------------------------- queries


class QueryParam(BaseModel):
    name: str
    kind: str  # bind (#{}) | substitution (${} runtime) | config_property | include_property | java_hole
    java_type: str | None = None
    jdbc_type: str | None = None
    type_handler: str | None = None
    resolved_value: str | None = None


class QueryVariant(BaseModel):
    dialect: str | None  # None = dialect independent / unknown
    dialect_evidence: str
    conditions: list[str] = Field(default_factory=list)
    sql: str
    partial: bool = False
    unresolved: list[str] = Field(default_factory=list)
    parameters: list[QueryParam] = Field(default_factory=list)


class QueryDef(BaseModel):
    key: str
    language: str  # sql | jpql | es_dsl | unknown
    origin: str  # mybatis | java | sql_file
    statement_kind: str | None = None  # select/insert/update/delete/sql-fragment/...
    namespace: str | None = None
    statement_id: str | None = None
    database_id: str | None = None
    result_map: str | None = None
    parameter_type: str | None = None
    result_type: str | None = None
    template: dict[str, Any] | None = None
    raw_text: str
    description: str | None = None  # from comments / javadoc
    provenance: Provenance
    is_test: bool = False
    variants: list[QueryVariant] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class SymbolDef(BaseModel):
    kind: str  # class | interface | record | enum | method | enum_constant
    fqn: str
    name: str
    javadoc: str | None = None
    comment: str | None = None
    signature: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    is_test: bool = False


class ResultMapping(BaseModel):
    result_map: str  # namespace.id
    java_type: str | None
    column: str
    property: str
    provenance: Provenance


class DocSection(BaseModel):
    path: str
    heading: str
    text: str
    line_start: int
    line_end: int
    role: FileRole
