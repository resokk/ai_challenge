"""CLI: python -m codeindex {index,search,eval,stats} ... --chunking <mode>"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import CodeIndexError
from .chunker import MODES
from .embedder import Embedder, default_model
from .rerank import DEFAULT_LLM
from . import evaluate
from .search import KINDS, add_pipeline_args, open_store, pipeline_from, search
from .store import DEFAULT_DB
from .summarizer import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL, Summarizer

PREVIEW_LINES = 8


def cmd_index(args) -> None:
    store = open_store(args.chunking, args.db, args.index_dir, must_exist=False)
    summarizer = Summarizer(args.summary_model) if args.chunking == "summary" else None
    result = store.sync(Path(args.repo_path), args.chunking, MODES[args.chunking], Embedder(args.model),
                        summarizer, rebuild=args.rebuild)
    print(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in result.items()))


def cmd_search(args) -> None:
    store = open_store(args.chunking, args.db, args.index_dir)
    results = search(store, args.chunking, args.model, args.query, args.k, args.path_prefix, args.kind,
                     args.class_name, pipeline_from(args))
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    for r in results:
        name = (r["qualified_name"] or "") + (f" [part {r['part']}]" if r["part"] is not None else "")
        rerank = f"rerank {r['rerank_score']}  " if "rerank_score" in r else ""
        print(f"{r['score']:.4f}  {rerank}{r['file_path']}:{r['start_line']}-{r['end_line']}  {r['kind']} {name}")
        for line in r["content"].splitlines()[:PREVIEW_LINES]:
            print(f"    {line}")
        print()


def cmd_eval(args) -> None:
    store = open_store(args.chunking, args.db, args.index_dir)
    questions = evaluate.load(args.questions)
    modes = evaluate.pipelines(args.min_score, args.fetch_k, args.min_relevance)
    unknown = set(args.modes or []) - set(modes)
    if unknown:
        sys.exit(f"error: unknown modes {sorted(unknown)}; choose from {list(modes)}")
    rows = {}
    for name in args.modes or modes:
        print(f"{name}:", file=sys.stderr)
        log = (lambda line: print(line, file=sys.stderr)) if args.verbose else None
        searcher = lambda q, k, p: search(store, args.chunking, args.model, q, k, pipeline=p)  # noqa: E731
        rows[name] = evaluate.evaluate(questions, searcher, replace(modes[name], llm=args.llm), args.k, log)
    print(f"\n{len(questions)} questions, k={args.k}, min score {args.min_score}, fetch k {args.fetch_k}, "
          f"min relevance {args.min_relevance}\n")
    print(evaluate.table(rows))


def cmd_stats(args) -> None:
    stats = open_store(args.chunking, args.db, args.index_dir).stats()
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
    add_pipeline_args(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", parents=[common, with_model], help="compare search pipelines on labeled questions")
    p.add_argument("questions", help="JSON list of {query, relevant: [qualified names]}")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--modes", nargs="+", help="modes to run (default: all)")
    p.add_argument("--min-score", type=float, default=0.62, help="similarity threshold of the filtering modes")
    p.add_argument("--fetch-k", type=int, default=20, help="candidates per query before reranking")
    p.add_argument("--min-relevance", type=int, default=5, help="llm modes: drop results rated lower (0-10)")
    p.add_argument("--llm", default=DEFAULT_LLM, help="Ollama model for rewriting and llm reranking")
    p.add_argument("-v", "--verbose", action="store_true", help="print each question's outcome")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stats", parents=[common], help="print counts")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    try:
        args.func(args)
    except CodeIndexError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
