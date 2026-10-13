"""Command line interface (Typer)."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import typer

from camunda_rag.config import Settings
from camunda_rag.storage import current_generation, generations_dir

app = typer.Typer(add_completion=False, help="Camunda data-model explorer: index, search, describe, validate, serve.")


def _settings(repo: Optional[Path]) -> Settings:
    s = Settings()
    if repo:
        s = s.model_copy(update={"repo_path": repo.expanduser().resolve()})
    return s


def _print(obj) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


@app.command()
def index(repo: Optional[Path] = typer.Option(None, help="path to the camunda/camunda checkout"),
          skip_java: bool = typer.Option(False, help="skip the JavaParser analyzer"),
          no_embed: bool = typer.Option(False, help="build catalog + FTS only (tokenizer still used for chunking)")) -> None:
    """Index the whole repository snapshot and publish a new generation."""
    from camunda_rag.indexer import Indexer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("sqlglot").setLevel(logging.ERROR)  # fallbacks are stored as diagnostics instead
    s = _settings(repo)
    gen = Indexer(s, progress=typer.echo).build(skip_java=skip_java, embed=not no_embed)
    typer.echo(f"published {gen}")


@app.command("build-java")
def build_java() -> None:
    """Build the JavaParser analyzer jar with Maven."""
    from camunda_rag.extractors.java import ensure_jar

    typer.echo(ensure_jar(Settings(), rebuild=True))


@app.command()
def inventory(status: Optional[str] = typer.Option(None, help="processed|irrelevant|excluded|unsupported"),
              limit: int = 50) -> None:
    """Per-file processing status of the published snapshot."""
    s = Settings()
    gen = current_generation(s.data_dir)
    if gen is None:
        raise typer.Exit("no published index")
    db = sqlite3.connect(generations_dir(s.data_dir) / gen / "catalog.sqlite")
    _print({"by_status_role": db.execute("SELECT status, role, COUNT(*) FROM source_files GROUP BY 1,2 ORDER BY 1,2").fetchall()})
    if status:
        _print(db.execute("SELECT path, role, reason, extractors FROM source_files WHERE status=? LIMIT ?", (status, limit)).fetchall())


@app.command()
def search(query: str, k: int = 10, storage_kind: Optional[str] = None, include_tests: bool = False,
           no_dense: bool = typer.Option(False, help="skip loading BGE-M3 (exact + FTS only)")) -> None:
    from camunda_rag.service import CatalogService

    _print(CatalogService(Settings(), load_embedder=not no_dense).search(query, k=k, storage_kind=storage_kind,
                                                                          include_tests=include_tests))


@app.command()
def describe(name: str, dialect: Optional[str] = None, storage_kind: Optional[str] = None, component: Optional[str] = None) -> None:
    from camunda_rag.service import CatalogService

    _print(CatalogService(Settings(), load_embedder=False).describe(name, dialect=dialect, storage_kind=storage_kind,
                                                                    component=component))


@app.command()
def queries(table: Optional[str] = None, column: Optional[str] = None, mapper: Optional[str] = None,
            access: Optional[str] = None, dialect: Optional[str] = None, variants: bool = False, limit: int = 20) -> None:
    from camunda_rag.service import CatalogService

    _print(CatalogService(Settings(), load_embedder=False).queries(table=table, column=column, mapper=mapper, access=access,
                                                                   dialect=dialect, include_variants=variants, limit=limit))


@app.command()
def validate(sql: str, dialect: Optional[str] = None) -> None:
    """Static SQL validation against the catalog (SQLGlot; nothing is executed)."""
    from camunda_rag.service import CatalogService

    _print(CatalogService(Settings(), load_embedder=False).validate(sql, dialect).model_dump())


@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None,
          no_dense: bool = typer.Option(False, help="serve without loading BGE-M3")) -> None:
    """HTTP API + MCP (streamable HTTP at /mcp)."""
    import uvicorn

    from camunda_rag.http_app import create_app
    from camunda_rag.service import CatalogService

    s = Settings()
    uvicorn.run(create_app(s, CatalogService(s, load_embedder=not no_dense)), host=host or s.host, port=port or s.port)


if __name__ == "__main__":
    app()
