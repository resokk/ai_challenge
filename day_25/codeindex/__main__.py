"""CLI: python -m codeindex {index,search,ask,chat,chat-eval,eval,eval-answers,stats} ... --chunking <mode>"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import CodeIndexError
from .chunker import MODES
from .embedder import Embedder, default_model
from .rerank import DEFAULT_LLM
from . import answer, evaluate
from .chat import Chat, ChatStore
from .search import (INDEX_ROOT, KINDS, add_pipeline_args, open_store, pipeline_from, search,
                     without_codebase_name)
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


def cmd_ask(args) -> None:
    store = open_store(args.chunking, args.db, args.index_dir)
    result = answer.ask(store, args.chunking, args.model, args.question, args.k, pipeline_from(args),
                        args.answer_llm)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['source']}] {s['file_path']}:{s['lines']}  {s['section']}  (chunk {s['chunk_id']}, "
                  f"score {s['score']})")
    if result["quotes"]:
        print("\nQuotes:")
        for q in result["quotes"]:
            print(f"  [{q['source']}] " + q["text"].replace("\n", "\n       "))
    if result["removed_statements"]:
        print("\nRemoved, their quotes are in no source:")
        for s in result["removed_statements"]:
            print(f"  - {s['text']}")
    if result["unverified_names"]:
        print(f"\nWarning, names not in the sources: {', '.join(result['unverified_names'])}")


def cmd_eval_answers(args) -> None:
    store = open_store(args.chunking, args.db, args.index_dir)
    pipeline = pipeline_from(args)
    summary = evaluate.evaluate_answers(
        evaluate.load(args.questions),
        lambda q: answer.ask(store, args.chunking, args.model, q, args.k, pipeline, args.answer_llm),
        lambda q, result: answer.judge(q, result, answer.ollama_json, args.answer_llm),
        lambda line: print(line, flush=True))
    print("\n" + "\n".join(f"{name:16} {value}" for name, value in summary.items()))


def make_chat(args, session: str) -> Chat:
    store = open_store(args.chunking, args.db, args.index_dir)
    pipeline = pipeline_from(args)
    return Chat(ChatStore(args.chat_db), session,
                lambda q: search(store, args.chunking, args.model, without_codebase_name(q, store), args.k,
                                 pipeline=pipeline),
                lambda q: search(store, args.chunking, args.model, without_codebase_name(q, store), answer.HINTS),
                answer.ollama_json, args.answer_llm, args.answer_llm)


def print_reply(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result["sources"]:
        print("\nИсточники:")
        for s in result["sources"]:
            print(f"  [{s['source']}] {s['file_path']}:{s['lines']}  {s['section']}  (chunk {s['chunk_id']})")
    elif result["nearest"]:
        print("\nИсточники: ничего выше порога релевантности; ближайшие найденные:")
        for s in result["nearest"]:
            print(f"  {s['file_path']}:{s['lines']}  {s['section']}  (chunk {s['chunk_id']}, score {s['score']})")
    for q in result["quotes"]:
        print(f"  > [{q['source']}] " + q["text"].strip().replace("\n", "\n    "))


def print_state_changes(before: dict, after: dict) -> None:
    if before["goal"] != after["goal"]:
        print(f"\n[память] цель: {after['goal']}")
    for name, label in (("clarified", "уточнено"), ("constraints", "ограничение"), ("terms", "термин")):
        for item in after[name][len(before[name]):]:
            print(f"[память] {label}: {item}")


def cmd_chat(args) -> None:
    chat = make_chat(args, args.session)
    print(f"Сессия «{args.session}». Команды: /state — память задачи, /history — история, /new — начать заново, "
          f"/quit — выход.")
    while True:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message in ("/quit", "/exit"):
            return
        if message == "/state":
            print(chat.state.render())
        elif message == "/history":
            for m in chat.store.history(args.session):
                print(f"{m['role']}: {m['content']}")
        elif message == "/new":
            chat.store.reset(args.session)
            print("Начали заново: история и память задачи очищены.")
        elif message:
            before = asdict(chat.state)
            try:
                result = chat.send(message)
            except CodeIndexError as e:
                print(f"error: {e}")
                continue
            print_reply(result)
            if result["status"] != "noted":  # a noted message already lists what went into the memory
                print_state_changes(before, result["state"])


def cmd_chat_eval(args) -> None:
    rows = {}
    for scenario in evaluate.load(args.scenarios):
        print(f"\n##### {scenario['name']}  ({len(scenario['turns'])} messages)", flush=True)
        session = f"eval-{scenario['name']}"
        chat = make_chat(args, session)
        chat.store.reset(session)
        rows[scenario["name"]] = evaluate.evaluate_chat(scenario, chat.send, lambda line: print(line, flush=True))
    print()
    for name, row in rows.items():
        print(f"{name}:  " + "   ".join(f"{k} {v}" for k, v in row.items()))


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

    # The best pipeline of day 23 (rewrite + threshold + LLM reranking) is the default for answers.
    for name, help_ in (("ask", "answer a question with sources and quotes"),
                        ("eval-answers", "check answers' sources, quotes and grounding on questions")):
        p = sub.add_parser(name, parents=[common, with_model], help=help_)
        p.add_argument("question" if name == "ask" else "questions",
                       help="the question" if name == "ask" else "JSON list of {query, relevant: [qualified names]}")
        p.add_argument("-k", type=int, default=5, help="chunks given to the model")
        add_pipeline_args(p)
        p.set_defaults(rewrite=True, min_score=0.62, rerank="llm", fetch_k=20)
        p.add_argument("--answer-llm", default=answer.DEFAULT_ANSWER_LLM,
                       help=f"Ollama model that answers (default: {answer.DEFAULT_ANSWER_LLM})")
        if name == "ask":
            p.add_argument("--json", action="store_true")
        p.set_defaults(func=cmd_ask if name == "ask" else cmd_eval_answers)

    for name, help_ in (("chat", "chat about the code: history, task memory, sources on every answer"),
                        ("chat-eval", "run long chat scenarios and check goal, memory and sources per turn")):
        p = sub.add_parser(name, parents=[common, with_model], help=help_)
        if name == "chat-eval":
            p.add_argument("scenarios", help="JSON list of {name, goal_keywords, turns: [{user, remember?, expect?}]}")
        else:
            p.add_argument("--session", default="default", help="conversation to continue or start")
        p.add_argument("--chat-db", default=str(INDEX_ROOT / "chat.db"), help="SQLite file with chat sessions")
        p.add_argument("-k", type=int, default=5, help="chunks given to the model")
        add_pipeline_args(p)
        # The task memory already turns each message into a standalone English query.
        p.set_defaults(rewrite=False, min_score=0.62, rerank="llm", fetch_k=20)
        p.add_argument("--answer-llm", default=answer.DEFAULT_ANSWER_LLM,
                       help=f"Ollama model for the task memory and answers (default: {answer.DEFAULT_ANSWER_LLM})")
        p.set_defaults(func=cmd_chat if name == "chat" else cmd_chat_eval)

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
