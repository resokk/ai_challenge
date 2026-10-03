"""Search quality per pipeline on a labeled question set.

The set is a JSON list of {"query": ..., "relevant": [qualified names]}; an empty list marks a
question the index cannot answer, where the right result is no result at all.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import CodeIndexError
from .search import Pipeline

Searcher = Callable[[str, int, Pipeline], list[dict]]
UNKNOWN = "said don't know"


def pipelines(min_score: float, fetch_k: int, min_relevance: int) -> dict[str, Pipeline]:
    """The compared modes: plain search, then each second stage with and without query rewriting."""
    plain = {
        "baseline": Pipeline(),
        "filter": Pipeline(min_score=min_score),
        "heuristic": Pipeline(min_score=min_score, rerank="heuristic", fetch_k=fetch_k),
        "llm": Pipeline(min_score=min_score, rerank="llm", fetch_k=fetch_k, min_relevance=min_relevance),
    }
    return {**plain, **{("rewrite" if name == "baseline" else f"rewrite+{name}"): replace(p, rewrite=True)
                        for name, p in plain.items()}}


def evaluate(questions: list[dict], search: Searcher, pipeline: Pipeline, k: int,
             log: Callable[[str], None] | None = None) -> dict[str, float]:
    """hit@k, MRR and precision over answerable questions, the rejection rate over the others."""
    hits = rr = precision = rejected = returned = 0.0
    answerable = [q for q in questions if q["relevant"]]
    start = time.perf_counter()
    for q in questions:
        results = search(q["query"], k, pipeline)
        returned += len(results)
        relevant = [r.get("qualified_name") in q["relevant"] for r in results]
        if not q["relevant"]:
            rejected += not results
        elif any(relevant):
            hits += 1
            rr += 1 / (relevant.index(True) + 1)
        if q["relevant"] and results:
            precision += sum(relevant) / len(results)
        if log:
            rank = relevant.index(True) + 1 if any(relevant) else "-"
            top = results[0]["qualified_name"] if results else "(nothing)"
            log(f"  {q['query'][:48]:48}  first relevant: {rank!s:2}  results: {len(results):2}  top: {top}")
    n, m = len(answerable) or 1, (len(questions) - len(answerable)) or 1
    return {"hit@k": hits / n, "MRR": rr / n, "precision": precision / n, "rejected": rejected / m,
            "avg results": returned / len(questions), "sec/query": (time.perf_counter() - start) / len(questions)}


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def table(rows: dict[str, dict[str, float]]) -> str:
    columns = list(next(iter(rows.values())))
    lines = [f"{'mode':18}" + "".join(f"{c:>12}" for c in columns)]
    lines += [f"{name:18}" + "".join(f"{v:12.2f}" for v in row.values()) for name, row in rows.items()]
    return "\n".join(lines)


def evaluate_answers(questions: list[dict], ask: Callable[[str], dict], judge: Callable[[str, dict], dict],
                     log: Callable[[str], None] = print) -> dict[str, str]:
    """Per question: sources, quotes, whether the answer's meaning matches its quotes, and whether it
    answered or said "I don't know" as it should; a question with empty "relevant" has no answer."""
    counts = dict.fromkeys(("answered", "sources", "quotes", "nothing removed", "no invented names",
                            "meaning matches", "right source", UNKNOWN), 0)
    answerable = [q for q in questions if q["relevant"]]
    for q in questions:
        start = time.perf_counter()
        try:
            result = ask(q["query"])
        except CodeIndexError as e:  # one failed question (Ollama timing out) does not end the run
            log(f"\n=== {q['query']}\nERROR: {e}")
            continue
        seconds = time.perf_counter() - start
        log(f"\n=== {q['query']}  ({seconds:.0f}s)\n{result['status']}: {result['answer']}")
        for s in result["sources"]:
            log(f"  source {s['source']}: {s['file_path']}:{s['lines']} {s['section']} (chunk {s['chunk_id']})")
        for quote in result["quotes"]:
            log(f"  quote {quote['source']}: " + quote["text"].replace("\n", "\n      "))
        if result["unverified_names"]:
            log(f"  names not in the sources: {', '.join(result['unverified_names'])}")
        for s in result["removed_statements"]:
            log(f"  REMOVED statement, its quote is in no source: {s['text']}\n    quote: {s['quote'][:120]!r}")
        if not q["relevant"]:
            counts[UNKNOWN] += result["status"] == "unknown"
            continue
        if result["status"] != "answered":
            continue
        try:
            verdict = judge(q["query"], result)
        except CodeIndexError as e:
            log(f"  judge ERROR: {e}")
            verdict = {"supported": False, "unsupported_claims": []}
        log(f"  judge: {'supported' if verdict['supported'] else 'NOT supported'}"
            + "".join(f"\n    unsupported: {c}" for c in verdict["unsupported_claims"]))
        counts["answered"] += 1
        counts["sources"] += bool(result["sources"])
        counts["quotes"] += bool(result["quotes"])
        counts["nothing removed"] += not result["removed_statements"]
        counts["no invented names"] += not result["unverified_names"]
        counts["meaning matches"] += bool(verdict["supported"])
        counts["right source"] += any(s["section"] in q["relevant"] for s in result["sources"])
    n, m = len(answerable), len(questions) - len(answerable)
    return {name: f"{v}/{m if name == UNKNOWN else n}" for name, v in counts.items()}


def evaluate_chat(scenario: dict, send: Callable[[str], dict], log: Callable[[str], None] = print) -> dict[str, str]:
    """Run a scenario's turns through one chat session and check every turn: does the reply show
    sources (or, for "I don't know", the nearest candidates), is the goal still in the task memory,
    and is everything the user said so far ("remember" keywords) still there; "expect" names the
    sections one of the sources should be."""
    counts = dict.fromkeys(("answered", "noted", "sources shown", "goal kept", "memory kept", "right source"), 0)
    expected = remembered = 0
    must_remember: list[str] = []
    state: dict = {}
    for i, turn in enumerate(scenario["turns"], 1):
        start = time.perf_counter()
        try:
            result = send(turn["user"])
        except CodeIndexError as e:
            log(f"\n[{i}] {turn['user']}\nERROR: {e}")
            continue
        state = result["state"]
        memory = json.dumps(state, ensure_ascii=False).lower()
        must_remember += turn.get("remember", [])
        missing = [kw for kw in must_remember if kw.lower() not in memory]
        goal_kept = any(kw.lower() in state["goal"].lower() for kw in scenario["goal_keywords"])
        shown = result["sources"] or result["nearest"]
        log(f"\n[{i}] user: {turn['user']}  ({time.perf_counter() - start:.0f}s)\n    query: {result['query']}\n"
            f"    {result['status']}: {result['answer'][:300]}")
        for s in shown:
            log(f"    {'source' if result['sources'] else 'nearest'} {s['source']}: {s['file_path']}:{s['lines']}"
                f" {s['section']} (chunk {s['chunk_id']})")
        log(f"    goal: {state['goal']}{'' if goal_kept else '   <-- GOAL LOST'}")
        if missing:
            log(f"    MEMORY LOST: {', '.join(missing)}")
        counts["answered"] += result["status"] == "answered"
        counts["noted"] += result["status"] == "noted"
        counts["sources shown"] += bool(shown)
        counts["goal kept"] += goal_kept
        if must_remember:
            remembered += 1
            counts["memory kept"] += not missing
        if turn.get("expect"):
            expected += 1
            counts["right source"] += any(s["section"] in turn["expect"] for s in result["sources"])
    n = len(scenario["turns"])
    totals = {"answered": n, "noted": n, "sources shown": n, "goal kept": n, "memory kept": remembered, "right source": expected}
    log(f"\nfinal task memory: {json.dumps(state, ensure_ascii=False, indent=2)}")
    return {name: f"{v}/{totals[name]}" for name, v in counts.items()}
