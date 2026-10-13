"""HTTP adapters: REST API (validation, search, describe, queries) and the MCP server at /mcp.

Both use the same CatalogService instance, hence the same published generation.
"""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import FastAPI, HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from camunda_rag.config import SQLGLOT_DIALECTS, Settings
from camunda_rag.service import CatalogService, NoIndexError
from camunda_rag.validation import ValidationResult


class ValidateRequest(BaseModel):
    sql: str = Field(..., description="SQL text; one or more statements. Bind parameters as ? or :name.")
    dialect: str | None = Field(None, description=f"one of {sorted(SQLGLOT_DIALECTS)}; default from settings")
    component: str | None = Field(None, description="restrict schema to a component prefix")


def build_mcp(service: CatalogService) -> MCPServer:
    mcp = MCPServer("camunda-data-model", instructions="Explore the Camunda data model indexed from source: search, describe, "
                                                       "queries; check SQL against it with validate_sql.")

    @mcp.tool()
    def search(query: str, k: int = 10, storage_kind: str | None = None, component: str | None = None,
               doc_types: list[str] | None = None, include_tests: bool = False) -> dict[str, Any]:
        """Hybrid search (exact identifiers + FTS5 + BGE-M3 dense) over tables, column groups, relationships,
        queries, mapper methods, Java model types and documentation of the indexed Camunda snapshot.
        storage_kind: relational_table | elasticsearch_index | opensearch_index | *_index_template | rocksdb_column_family.
        doc_types: entity, column_group, relationship, query, mapper_method, java_type, documentation."""
        return service.search(query, k=k, storage_kind=storage_kind, component=component, doc_types=doc_types,
                              include_tests=include_tests)

    @mcp.tool()
    def describe(name: str, dialect: str | None = None, storage_kind: str | None = None, component: str | None = None,
                 include_tests: bool = False) -> dict[str, Any]:
        """Full structured description of a table / index / column family from the catalog (not from search top-k):
        columns with types, nullability, purpose and Java mappings, constraints, indexes, declared FKs vs JOIN-inferred
        relations, schema change history with outcomes for the profile, profile completeness and provenance.
        dialect selects the schema profile (postgresql, mysql, mariadb, oracle, mssql, h2)."""
        return service.describe(name, dialect=dialect, storage_kind=storage_kind, component=component, include_tests=include_tests)

    @mcp.tool()
    def queries(table: str | None = None, column: str | None = None, mapper: str | None = None, access: str | None = None,
                dialect: str | None = None, language: str | None = None, text: str | None = None, include_tests: bool = False,
                include_variants: bool = False, limit: int = 20) -> dict[str, Any]:
        """Queries touching a table/column (access: read|write), of a mapper (namespace or class substring), or by language
        (sql, jpql, es_dsl). Returns MyBatis statements / Java queries with mapper method, tables, representative variant
        for the dialect, parameters, branch conditions, parse status and provenance."""
        return service.queries(table=table, column=column, mapper=mapper, access=access, dialect=dialect, language=language,
                               text=text, include_tests=include_tests, include_variants=include_variants, limit=limit)

    @mcp.tool()
    def validate_sql(sql: str, dialect: str | None = None, component: str | None = None) -> dict[str, Any]:
        """Static SQLGlot validation of SQL against the indexed schema; the SQL is never executed.
        sql: one or more statements, bind parameters as ? or :name.
        dialect: postgresql | mysql | mariadb | oracle | mssql | h2 (default from settings) selects the schema profile;
        component restricts the schema to a component prefix.
        Returns status (valid | invalid | unsupported | parse_error), issues (unknown_table, unknown_column with
        suggestions, ambiguous_column, ...) and per statement its type, normalized SQL and tables read/written."""
        try:
            return service.validate(sql, dialect, component).model_dump(mode="json")
        except ValueError as e:  # bad dialect/component: a plain ValueError reaches the client without its message
            raise ToolError(str(e)) from e

    return mcp


def create_app(settings: Settings | None = None, service: CatalogService | None = None) -> FastAPI:
    settings = settings or Settings()
    service = service or CatalogService(settings)
    mcp = build_mcp(service)
    mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", json_response=True, stateless_http=True, host=settings.host)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="Camunda data-model explorer", lifespan=lifespan)

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except NoIndexError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            st = service.status()
            return {"status": "ok", "generation": st["generation"], "commit_sha": st["commit_sha"], "dense_search": st["dense_search"]}
        except NoIndexError as e:
            return {"status": "no_index", "detail": str(e)}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return guard(service.status)

    @app.post("/validate", response_model=ValidationResult)
    def validate(req: ValidateRequest) -> ValidationResult:
        return guard(service.validate, req.sql, req.dialect, req.component)

    @app.get("/search")
    def search(q: str, k: int = 10, storage_kind: str | None = None, component: str | None = None,
               include_tests: bool = False) -> dict[str, Any]:
        return guard(service.search, q, k=k, storage_kind=storage_kind, component=component, include_tests=include_tests)

    @app.get("/describe/{name}")
    def describe(name: str, dialect: str | None = None, storage_kind: str | None = None, component: str | None = None) -> dict[str, Any]:
        return guard(service.describe, name, dialect=dialect, storage_kind=storage_kind, component=component)

    @app.get("/queries")
    def queries(table: str | None = None, column: str | None = None, mapper: str | None = None, access: str | None = None,
                dialect: str | None = None, limit: int = 20) -> dict[str, Any]:
        return guard(service.queries, table=table, column=column, mapper=mapper, access=access, dialect=dialect, limit=limit)

    app.router.routes.extend(mcp_app.routes)  # MCP streamable HTTP endpoint at /mcp
    return app
