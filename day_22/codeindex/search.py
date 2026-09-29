"""Opening an index and querying it: shared by the CLI and the MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

from . import CodeIndexError
from .embedder import Embedder
from .store import DEFAULT_DB, Store

INDEX_ROOT = Path(__file__).resolve().parent.parent / "index"
Kind = Literal["file", "module", "class", "function", "method", "block"]
KINDS = list(get_args(Kind))
RESULT_FIELDS = ("score", "file_path", "start_line", "end_line", "kind",
                 "class_name", "symbol_name", "qualified_name", "part", "content")


def open_store(chunking: str, db: str = DEFAULT_DB, index_dir: str | None = None,
               must_exist: bool = True) -> Store:
    directory = Path(index_dir) if index_dir else INDEX_ROOT / chunking
    if must_exist and not (directory / db).exists():
        raise CodeIndexError(f"no index in {directory}; run: python -m codeindex index <repo_path> "
                             f"--chunking {chunking}")
    return Store(directory, db)


def search(store: Store, chunking: str, model: str, query: str, k: int, path_prefix: str | None = None,
           kind: Kind | None = None, class_name: str | None = None) -> list[dict]:
    """The k best chunks for query, each as a dict of RESULT_FIELDS."""
    store.check(chunking=chunking, model=model)
    results = store.search(Embedder(model).embed_query(query), k, path_prefix, kind, class_name,
                           merge_parts=chunking in ("recursive", "contextual"))
    return [{f: r[f] for f in RESULT_FIELDS} for r in results]
