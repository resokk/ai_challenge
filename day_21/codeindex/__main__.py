"""CLI: python -m codeindex {index,search,stats} ... --chunking <mode>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import CodeIndexError
from .chunker import MODES
from .embedder import Embedder, default_model
from .store import DEFAULT_DB, Store
from .summarizer import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL, Summarizer

INDEX_ROOT = Path(__file__).resolve().parent.parent / "index"
KINDS = ["file", "module", "class", "function", "method", "block"]
PREVIEW_LINES = 8


def open_store(args, must_exist: bool) -> Store:
    directory = Path(args.index_dir) if args.index_dir else INDEX_ROOT / args.chunking
    if must_exist and not (directory / args.db).exists():
        raise CodeIndexError(f"no index in {directory}; run: python -m codeindex index <repo_path> "
                             f"--chunking {args.chunking}")
    return Store(directory, args.db)


def cmd_index(args) -> None:
    store = open_store(args, must_exist=False)
    summarizer = Summarizer(args.summary_model) if args.chunking == "summary" else None
    result = store.sync(Path(args.repo_path), args.chunking, MODES[args.chunking], Embedder(args.model),
                        summarizer, rebuild=args.rebuild)
    print(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in result.items()))


def cmd_search(args) -> None:
    store = open_store(args, must_exist=True)
    store.check(chunking=args.chunking, model=args.model)
    embedder = Embedder(args.model)
    results = store.search(embedder.embed_query(args.query), args.k, args.path_prefix, args.kind,
                           args.class_name, merge_parts=args.chunking in ("recursive", "contextual"))
    if args.json:
        print(json.dumps([{k: r[k] for k in ("score", "file_path", "start_line", "end_line", "kind",
                                             "class_name", "symbol_name", "qualified_name", "part", "content")}
                          for r in results], indent=2, ensure_ascii=False))
        return
    for r in results:
        name = (r["qualified_name"] or "") + (f" [part {r['part']}]" if r["part"] is not None else "")
        print(f"{r['score']:.4f}  {r['file_path']}:{r['start_line']}-{r['end_line']}  {r['kind']} {name}")
        for line in r["content"].splitlines()[:PREVIEW_LINES]:
            print(f"    {line}")
        print()


def cmd_stats(args) -> None:
    stats = open_store(args, must_exist=True).stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="codeindex", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--chunking", choices=list(MODES), default="ast")
    common.add_argument("--index-dir", help="index folder (default: index/<mode>)")
    common.add_argument("--db", default=DEFAULT_DB,
                        help=f"SQLite file name in the index folder (default: {DEFAULT_DB}); "
                             "its vectors go in <name>.faiss beside it")
    with_model = argparse.ArgumentParser(add_help=False)
    with_model.add_argument("--model", default=default_model(),
                            help="Ollama embedding model (default: $OLLAMA_EMBED_MODEL or nomic-embed-text)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", parents=[common, with_model], help="build or update the index")
    p.add_argument("repo_path")
    p.add_argument("--rebuild", action="store_true", help="start over, e.g. after changing the model")
    p.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL, help="Ollama model for summary mode")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", parents=[common, with_model], help="find chunks matching a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--path-prefix")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--class", dest="class_name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", parents=[common], help="print counts")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    try:
        args.func(args)
    except CodeIndexError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
