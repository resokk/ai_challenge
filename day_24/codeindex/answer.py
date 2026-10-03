"""Answers grounded in the retrieved chunks: every answer carries its sources and verbatim quotes.

The model sees the chunks as numbered sources [S1], [S2], ... and must reply in JSON with the
answer as a list of statements, each with the quote from a source that proves it. Nothing it says
is taken on trust: a quote counts only if it occurs word for word in a source, a statement whose
quote does not is cut from the answer, the sources listed are the ones the kept quotes come from,
and an answer with no statement left becomes "I don't know". With no chunk above the relevance
threshold the model is not asked at all.
"""

from __future__ import annotations

import builtins
import json
import keyword
import re
from typing import Callable

from . import CodeIndexError
from .embedder import post_json
from .search import Pipeline, search
from .store import Store

DEFAULT_ANSWER_LLM = "qwen2.5-coder:7b"
MIN_QUOTE_CHARS = 12  # shorter "quotes" (a bare name, "return x") prove nothing
SOURCE_CHARS = 4000
HINTS = 3
NUM_CTX, NUM_PREDICT = 16384, 2000  # Ollama's default 4096-token window cuts off a 5-source prompt
CODE_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
PYTHON_NAMES = set(keyword.kwlist) | set(dir(builtins)) | {"python"}

GenerateJSON = Callable[[str, str, dict], dict]  # (model, prompt, JSON schema) -> parsed reply

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "known": {"type": "boolean"},
        "statements": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "source": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["text", "source", "quote"]}},
        "missing": {"type": "string"},
    },
    "required": ["known", "statements", "missing"],
}

ANSWER_PROMPT = """You answer questions about a code base using ONLY the numbered sources below.

Reply with the answer split into statements. Each statement is one fact, written for the user, with
the source it comes from (e.g. "S2") and a quote proving it: an exact excerpt of that source, one or
more whole lines copied character for character (no "...", no edits, no reformatting).

Rules:
- State only what your quote shows. Do not use outside knowledge, do not guess.
- Do not write code examples of your own and do not invent parameters or names.
- Together the statements must answer the question; start with the direct answer.
- If the sources do not answer the question, set "known" to false, leave "statements" empty and say in
  "missing" what is missing and what the user could clarify.
- Write "text" and "missing" in the language of the question: {language}. Quotes stay as in the source.

Question: {question}

{sources}

Reply in JSON: {{"known": true|false, "statements": [{{"text": "...", "source": "S1", "quote": "..."}}],
"missing": ""}}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"},
                   "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["supported", "unsupported_claims"],
}

JUDGE_PROMPT = """Check an answer against the quotes it cites.

Question: {question}

Answer, statement by statement, each followed by the quote given as its proof and where the quote is
from (file and the class, method or function it is in):
{statements}

Does each quote support its statement? The place a quote is from counts as part of it: a quote from
inside NoMatch is about NoMatch. Ignore wording, translation and references like [S1]; judge only the
meaning. A code example, parameter, class or method name that appears in neither the quotes nor their
places is unsupported. List the statements the quotes do not support.
Reply in JSON: {{"supported": true|false, "unsupported_claims": ["..."]}}"""

IDK = {
    "ru": "Не знаю: в найденном коде нет надёжного ответа на этот вопрос. Уточните, пожалуйста, вопрос — "
          "например, назовите класс, функцию или модуль, о котором речь.",
    "en": "I don't know: the code found does not reliably answer this question. Please clarify it — "
          "for example, name the class, function or module you mean.",
}
NEAREST = {"ru": "Ближе всего, но ниже порога релевантности: ", "en": "Closest matches, below the relevance threshold: "}


def ollama_json(model: str, prompt: str, schema: dict) -> dict:
    """Ollama's reply constrained to schema, parsed."""
    reply = post_json("/api/generate", {"model": model, "prompt": prompt, "stream": False, "format": schema,
                                        "options": {"temperature": 0, "num_ctx": NUM_CTX,
                                                    "num_predict": NUM_PREDICT}})["response"]
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        raise CodeIndexError(f"{model} replied with invalid JSON: {reply[:200]!r}") from None


def language(text: str) -> str:
    return "ru" if re.search(r"[а-яё]", text, re.IGNORECASE) else "en"


def bare(text: str) -> str:
    """text without whitespace, so a quote reflowed onto other lines still matches."""
    return re.sub(r"\s+", "", text)


def unverified_names(text: str, chunks: list[dict]) -> list[str]:
    """Names in the answer's `code` spans that occur in none of the sources: likely invented."""
    known = set(NAME_RE.findall(" ".join(f"{c['content']} {c['qualified_name'] or ''} {c['file_path']}"
                                         for c in chunks)))
    names = {n for span in CODE_RE.findall(text) for n in NAME_RE.findall(span.strip("`"))}
    return sorted(names - known - PYTHON_NAMES)


def source_of(chunk: dict, label: str) -> dict:
    return {"source": label, "chunk_id": chunk["id"], "file_path": chunk["file_path"],
            "lines": f"{chunk['start_line']}-{chunk['end_line']}", "section": chunk["qualified_name"] or chunk["kind"],
            "score": round(chunk["score"], 4)}


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(f"[S{i}] {c['file_path']}:{c['start_line']}-{c['end_line']} {c['qualified_name'] or ''}\n"
                       f"```\n{c['content'][:SOURCE_CHARS]}\n```" for i, c in enumerate(chunks, 1))


def locate(quote: str, claimed: str, chunks: list[dict]) -> str | None:
    """The label of the source quote occurs in, trying the claimed one first; None if in none.

    Whitespace and the quote's own surrounding punctuation are ignored; a quote credited to the
    wrong source is credited to the right one.
    """
    text = bare(quote).strip(".,;:'\"")
    if len(text) < MIN_QUOTE_CHARS:
        return None
    labels = [f"S{i}" for i in range(1, len(chunks) + 1)]
    order = [claimed] + [label for label in labels if label != claimed] if claimed in labels else labels
    return next((label for label in order if text in bare(chunks[int(label[1:]) - 1]["content"])), None)


def unknown(question: str, reason: str, nearest: list[dict], detail: str = "", **extra) -> dict:
    lang = language(question)
    text = IDK[lang] + (f"\n{detail.strip()}" if detail.strip() else "")
    if nearest:
        text += "\n" + NEAREST[lang] + ", ".join(
            f"{c['qualified_name'] or c['kind']} ({c['file_path']}, {c['score']:.2f})" for c in nearest[:HINTS])
    return {"status": "unknown", "reason": reason, "answer": text, "sources": [], "quotes": [],
            "removed_statements": [], "unverified_names": [], **extra}


def answer(question: str, chunks: list[dict], generate: GenerateJSON, model: str = DEFAULT_ANSWER_LLM,
           nearest: Callable[[], list[dict]] = list) -> dict:
    """{status: answered|unknown, answer, sources, quotes, removed_statements, unverified_names[, reason]}.

    chunks are the retrieved results that passed the relevance threshold; nearest() gives the best
    candidates regardless of it, offered as hints when the answer is "I don't know".
    """
    if not chunks:
        return unknown(question, "no_relevant_context", nearest())
    lang = language(question)
    reply = generate(model, ANSWER_PROMPT.format(language="Russian" if lang == "ru" else "English",
                                                 question=question, sources=format_sources(chunks)),
                     ANSWER_SCHEMA)
    kept, removed = [], []
    for s in reply.get("statements") or []:
        text, quote = str(s.get("text", "")).strip(), str(s.get("quote", "")).strip("\n")
        label = locate(quote, str(s.get("source", "")).strip("[] "), chunks)
        if text and label:
            kept.append({"text": text, "source": label, "quote": quote})
        elif text:
            removed.append({"text": text, "source": s.get("source", ""), "quote": quote})
    if not reply.get("known", False):
        return unknown(question, "model", nearest(), str(reply.get("missing", "")), removed_statements=removed)
    if not kept:
        return unknown(question, "no_verified_quotes", nearest(), removed_statements=removed)
    chunk_of = lambda label: chunks[int(label[1:]) - 1]  # noqa: E731
    quotes = {}
    for s in kept:
        quotes.setdefault((s["source"], bare(s["quote"])), {
            "source": s["source"], "chunk_id": chunk_of(s["source"])["id"],
            "file_path": chunk_of(s["source"])["file_path"], "text": s["quote"]})
    text = " ".join(f"{s['text']} [{s['source']}]" for s in kept)
    return {"status": "answered", "answer": text, "statements": kept,
            "sources": [source_of(chunk_of(label), label) for label in dict.fromkeys(s["source"] for s in kept)],
            "quotes": list(quotes.values()), "removed_statements": removed,
            "unverified_names": unverified_names(text, chunks)}


def judge(question: str, result: dict, generate: GenerateJSON, model: str = DEFAULT_ANSWER_LLM) -> dict:
    """{supported, unsupported_claims}: does the meaning of each statement match its quote?"""
    places = {s["source"]: f"{s['file_path']}, {s['section']}" for s in result["sources"]}
    statements = "\n\n".join(f"{i}. {s['text']}\n   quote from {places[s['source']]}: {s['quote']}"
                             for i, s in enumerate(result["statements"], 1))
    return generate(model, JUDGE_PROMPT.format(question=question, statements=statements), JUDGE_SCHEMA)


def ask(store: Store, chunking: str, model: str, question: str, k: int, pipeline: Pipeline,
        answer_llm: str = DEFAULT_ANSWER_LLM, **filters) -> dict:
    """Retrieve with pipeline, then answer from what passed it; filters as for search()."""
    chunks = search(store, chunking, model, question, k, pipeline=pipeline, **filters)
    nearest = lambda: search(store, chunking, model, question, HINTS, **filters)  # noqa: E731
    return answer(question, chunks, ollama_json, answer_llm, nearest)
