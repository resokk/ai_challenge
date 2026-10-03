"""Opening an index and querying it: shared by the CLI and the MCP server.

A query runs through up to three stages: an optional LLM rewrite, vector search for fetch_k
candidates (for both the original and the rewritten query), then a second stage that drops
candidates below min_score similarity and reranks the rest before the top k are returned.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, get_args

import numpy as np

from . import CodeIndexError
from .embedder import Embedder
from .rerank import DEFAULT_LLM, Generate, heuristic_rerank, llm_rerank, rewrite_query
from .store import DEFAULT_DB, Store
from .summarizer import ollama_generate

INDEX_ROOT = Path(__file__).resolve().parent.parent / "index"
Kind = Literal["file", "module", "class", "function", "method", "block"]
KINDS = list(get_args(Kind))
Rerank = Literal["none", "heuristic", "llm"]
RERANKS = list(get_args(Rerank))
RESULT_FIELDS = ("id", "score", "file_path", "start_line", "end_line", "kind",
                 "class_name", "symbol_name", "qualified_name", "part", "content")


@dataclass(frozen=True)
class Pipeline:
    """How a query is searched; the defaults are plain vector search."""
    rewrite: bool = False  # an LLM rewrites the query first; candidates of both queries are merged
    min_score: float = 0.0  # candidates with a lower cosine similarity are dropped
    rerank: Rerank = "none"
    min_relevance: int = 5  # rerank llm: candidates the LLM rates lower (0-10) are dropped
    fetch_k: int | None = None  # candidates per query before the second stage (default: k)
    llm: str = DEFAULT_LLM


def open_store(chunking: str, db: str = DEFAULT_DB, index_dir: str | None = None,
               must_exist: bool = True) -> Store:
    directory = Path(index_dir) if index_dir else INDEX_ROOT / chunking
    if must_exist and not (directory / db).exists():
        raise CodeIndexError(f"no index in {directory}; run: python -m codeindex index <repo_path> "
                             f"--chunking {chunking}")
    return Store(directory, db)


def ollama_llm(model: str, prompt: str) -> str:
    return ollama_generate(model, prompt, temperature=0)


def run(store: Store, embed_query: Callable[[str], np.ndarray], generate: Generate, query: str, k: int,
        pipeline: Pipeline, merge_parts: bool = False, **filters) -> list[dict]:
    """The k best chunks for query after every stage pipeline enables, each as a dict of RESULT_FIELDS
    (plus rerank_score when reranked). filters are Store.search's path_prefix, kind and class_name."""
    queries = [query]
    if pipeline.rewrite:
        rewritten = rewrite_query(query, generate, pipeline.llm)
        print(f"rewritten query: {rewritten}", file=sys.stderr)
        if rewritten.lower() != query.lower():
            queries.append(rewritten)
    fetch_k = max(pipeline.fetch_k or k, k)
    candidates: dict = {}
    for q in queries:  # a chunk (or split unit, when merging parts) found twice keeps its better score
        for r in store.search(embed_query(q), fetch_k, merge_parts=merge_parts, **filters):
            key = (r["file_path"], r["qualified_name"]) if merge_parts and r["part"] is not None else r["id"]
            if key not in candidates or r["score"] > candidates[key]["score"]:
                candidates[key] = r
    results = [{f: r[f] for f in RESULT_FIELDS} for r in candidates.values() if r["score"] >= pipeline.min_score]
    results.sort(key=lambda r: -r["score"])
    # The heuristic matches the terms of both queries (a rewrite adds English identifiers);
    # the LLM judges each candidate against the user's own question.
    if pipeline.rerank == "heuristic":
        results = heuristic_rerank(" ".join(queries), results)
    elif pipeline.rerank == "llm":
        results = llm_rerank(query, results, generate, pipeline.llm, pipeline.min_relevance)
    return results[:k]


def search(store: Store, chunking: str, model: str, query: str, k: int, path_prefix: str | None = None,
           kind: Kind | None = None, class_name: str | None = None,
           pipeline: Pipeline = Pipeline()) -> list[dict]:
    """run() against Ollama, after checking that the index was built with chunking and model."""
    store.check(chunking=chunking, model=model)
    return run(store, Embedder(model).embed_query, ollama_llm, query, k, pipeline,
               merge_parts=chunking in ("recursive", "contextual"),
               path_prefix=path_prefix, kind=kind, class_name=class_name)


def add_pipeline_args(parser) -> None:
    """The Pipeline options, for the CLI and the server."""
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=False,
                        help="rewrite the query with an LLM first")
    parser.add_argument("--min-score", type=float, default=0.0, help="drop results below this similarity")
    parser.add_argument("--rerank", choices=RERANKS, default="none", help="second stage after vector search")
    parser.add_argument("--min-relevance", type=int, default=5, help="--rerank llm: drop results rated lower (0-10)")
    parser.add_argument("--fetch-k", type=int, help="candidates per query before the second stage (default: k)")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"Ollama model for --rewrite and --rerank llm "
                                                           f"(default: {DEFAULT_LLM})")


def pipeline_from(args) -> Pipeline:
    return Pipeline(rewrite=args.rewrite, min_score=args.min_score, rerank=args.rerank,
                    min_relevance=args.min_relevance, fetch_k=args.fetch_k, llm=args.llm)
