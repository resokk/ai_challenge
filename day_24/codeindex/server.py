"""MCP server over HTTP: python -m codeindex.server [--chunking <mode>] [--db <name>] ...

Serves one index, picked at startup with the same options as the CLI, through a
search_code tool, and answers questions from it with sources and quotes through ask_code.
The search pipeline options (--rewrite, --min-score, --rerank ...) are fixed
at startup too. Build the index first with python -m codeindex index.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import CodeIndexError, answer
from .chunker import MODES
from .embedder import default_model
from .search import Kind, add_pipeline_args, open_store, pipeline_from, search
from .store import DEFAULT_DB

# Loopback only, like the other MCP servers here: nothing authenticates the
# callers, and results carry the indexed source code.
HOST = "127.0.0.1"
PORT = 8768
MAX_K = 50

server = MCPServer("codeindex")
config = argparse.Namespace()  # set by main()


@server.tool()
def search_code(query: str, k: int = 10, path_prefix: str | None = None, kind: Kind | None = None,
                class_name: str | None = None) -> list[dict[str, Any]]:
    """Semantic search over the indexed source tree: the k chunks of code closest in meaning to query.

    query is a natural-language description of the code wanted, e.g. "read sensor temperature".
    path_prefix keeps only files whose repo-relative path starts with it; kind keeps one kind of
    chunk; class_name keeps chunks inside that class. Each result has its file_path, start_line,
    end_line, kind, qualified_name, similarity score and the chunk's content, plus rerank_score
    when the server reranks. Fewer than k results, or none, means the rest were not relevant enough.
    """
    if not 1 <= k <= MAX_K:
        raise ToolError(f"k must be between 1 and {MAX_K}, got {k}")
    try:
        # A store per call: its SQLite connection must stay in the calling
        # thread, and a re-run of `index` is picked up without a restart.
        store = open_store(config.chunking, config.db, config.index_dir)
        return search(store, config.chunking, config.model, query, k, path_prefix, kind, class_name,
                      config.pipeline)
    except CodeIndexError as exc:
        # Only a ToolError's message reaches the caller; anything else arrives
        # as a bare "Error executing tool".
        raise ToolError(str(exc)) from None


@server.tool()
def ask_code(question: str, k: int = 5) -> dict[str, Any]:
    """Answer a question about the indexed code from its k most relevant chunks, with proof.

    Returns status "answered" with the answer, its sources (file_path, lines, section, chunk_id) and
    verbatim quotes from them, or status "unknown" when nothing relevant enough was found: then the
    answer says "I don't know" and asks to clarify the question. Quotes are checked against the code.
    """
    if not 1 <= k <= MAX_K:
        raise ToolError(f"k must be between 1 and {MAX_K}, got {k}")
    try:
        store = open_store(config.chunking, config.db, config.index_dir)
        return answer.ask(store, config.chunking, config.model, question, k, config.pipeline, config.answer_llm)
    except CodeIndexError as exc:
        raise ToolError(str(exc)) from None


def main() -> None:
    parser = argparse.ArgumentParser(prog="codeindex.server", description=__doc__)
    parser.add_argument("--chunking", choices=list(MODES), default="ast")
    parser.add_argument("--index-dir", help="index folder (default: index/<mode>)")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file name in the index folder (default: {DEFAULT_DB})")
    parser.add_argument("--model", default=default_model(),
                        help="Ollama embedding model (default: $OLLAMA_EMBED_MODEL or nomic-embed-text)")
    add_pipeline_args(parser)
    parser.add_argument("--answer-llm", default=answer.DEFAULT_ANSWER_LLM,
                        help=f"Ollama model for ask_code (default: {answer.DEFAULT_ANSWER_LLM})")
    parser.parse_args(namespace=config)
    config.pipeline = pipeline_from(config)
    try:  # fail at startup, not on the first call, if the index is missing or was built otherwise
        open_store(config.chunking, config.db, config.index_dir).check(chunking=config.chunking, model=config.model)
    except CodeIndexError as e:
        sys.exit(f"error: {e}")
    server.run(transport="streamable-http", host=HOST, port=PORT, json_response=True, stateless_http=True)


if __name__ == "__main__":
    main()
