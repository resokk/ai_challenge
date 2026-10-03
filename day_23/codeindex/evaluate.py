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

from .search import Pipeline

Searcher = Callable[[str, int, Pipeline], list[dict]]


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
