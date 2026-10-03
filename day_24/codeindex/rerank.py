"""The stages around vector search: query rewriting before it, relevance filtering and reranking after it.

Each takes a generate(model, prompt) -> str callable for its LLM, so tests can pass a fake.
"""

from __future__ import annotations

import re
from typing import Callable

Generate = Callable[[str, str], str]

DEFAULT_LLM = "qwen2.5-coder:3b"  # 1.5b rates relevance close to randomly

NAME_WEIGHT, CONTENT_WEIGHT = 0.08, 0.04  # heuristic: bonus for query terms in the name / the code
LLM_CODE_CHARS = 2000

REWRITE_PROMPT = """Rewrite the question below as a search query for a source code base.
Translate it to English if needed, keep its meaning, and add the Python function, method or class
names and technical keywords such code would likely use. Reply with the query only, on one line.

Question: {query}"""

RERANK_PROMPT = """You judge results of a code search engine.

Query: {query}

Result: {name} in {path}
```
{code}
```

How well does this code answer the query? Reply with one integer from 0 (unrelated) to 10 (exactly
what the query asks for), and nothing else."""

WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")  # splits snake_case and CamelCase
STOPWORDS = {"the", "and", "for", "with", "from", "into", "that", "this", "how", "what", "when", "where",
             "which", "are", "was", "get", "all", "any", "one", "can", "its", "use", "self", "def", "return"}


def rewrite_query(query: str, generate: Generate, model: str = DEFAULT_LLM) -> str:
    """The LLM's search-friendly version of query; query itself if the reply is empty."""
    lines = [line.strip().strip('"') for line in generate(model, REWRITE_PROMPT.format(query=query)).splitlines()]
    return next((line for line in lines if line), query)


def terms(text: str) -> set[str]:
    """Lowercased words of text and of the identifiers in it, plurals folded."""
    words = (w.lower() for w in WORD_RE.findall(text))
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if len(w) > 2} - STOPWORDS


def heuristic_rerank(query: str, results: list[dict]) -> list[dict]:
    """Similarity plus a bonus for the query's terms in each result's name and code, best first."""
    wanted = terms(query)
    for r in results:
        name = len(wanted & terms(r["qualified_name"] or r["file_path"])) / (len(wanted) or 1)
        code = len(wanted & terms(r["content"])) / (len(wanted) or 1)
        r["rerank_score"] = round(r["score"] + NAME_WEIGHT * name + CONTENT_WEIGHT * code, 4)
    return sorted(results, key=lambda r: -r["rerank_score"])


def llm_rerank(query: str, results: list[dict], generate: Generate, model: str = DEFAULT_LLM,
               min_relevance: int = 0) -> list[dict]:
    """Results the LLM rates at least min_relevance out of 10, best rated first, ties by similarity."""
    kept = []
    for r in results:
        reply = generate(model, RERANK_PROMPT.format(query=query, name=r["qualified_name"] or r["kind"],
                                                     path=r["file_path"], code=r["content"][:LLM_CODE_CHARS]))
        match = re.search(r"\d+", reply)
        r["rerank_score"] = min(int(match.group()), 10) if match else 0
        if r["rerank_score"] >= min_relevance:
            kept.append(r)
    return sorted(kept, key=lambda r: (-r["rerank_score"], -r["score"]))
