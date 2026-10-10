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
import keyword
import re
from typing import Callable

from . import llm
from .llm import generate_json
from .search import Pipeline, search
from .store import Store

DEFAULT_ANSWER_LLM = "mlx-community/Qwen2.5-Coder-3B-Instruct-8bit"  # as good as bf16 here, half the memory
MIN_QUOTE_CHARS = 12  # shorter "quotes" (a bare name, "return x") prove nothing
SOURCE_CHARS = 4000
MIN_SOURCE_CHARS = 300  # a source cut shorter than this to fit the context window is left out
CHARS_PER_TOKEN = 3.5  # Qwen's tokenizer averages 4 characters of Python code per token
HINTS = 3
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

# Written for this task: a developer asks how to do something with a Python library, the sources are
# functions and classes of that library, and only verbatim quotes count. Compared with the generic
# prompt of day 28 it names what a good answer is (the method to call and its signature), asks for
# one-line quotes (a def line or a docstring line: easy to copy exactly) and shows one worked example.
ANSWER_PROMPT = '''You are a senior Python developer. A developer asks how to do something with a library;
answer from its source code below, the numbered sources, and nothing else.

A good answer names the method, function or class to use and how to call it: its parameters, from
its def line, and what it returns or does, from its docstring. Give 1-3 statements. Each has:
- "text": one fact for the developer, in {language};
- "source": the source it comes from, e.g. "S2";
- "quote": ONE line copied character for character from that source — preferably the def or class
  line, or a docstring line. No "...", no edits, never a line you wrote yourself.

Rules:
- Use only what the sources show: no outside knowledge, no code examples, no invented names.
- Prefer public methods (QuerySet, Model) over private helpers (names starting with "_").
- If no source does what the question asks (for example, the question is about a database, tool or
  feature the code never mentions), set "known" to false, "statements" to [], and say in "missing",
  in {language}, what is missing and what the developer could clarify.

Example. Question: "how do I fetch the first matching row?" with source
[S1] query.py:10-14 QuerySet.first
```
async def first(self, *args, **kwargs) -> "Model":
    """
    Gets the first row from the db ordered by primary key column ascending.
    """
```
Reply:
{{"known": true, "statements": [{{"text": "Call first() on a query set: it returns the first row ordered by primary key.", "source": "S1", "quote": "async def first(self, *args, **kwargs) -> \\"Model\\":"}}], "missing": ""}}
{context}
Question: {question}

{sources}

Write "text" and "missing" in {language}, whatever the language of the question, the example and the code.
Reply with the JSON object only: {{"known": true|false, "statements": [{{"text": "...", "source": "S1", "quote": "..."}}], "missing": ""}}'''

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


def source_header(i: int, chunk: dict) -> str:
    return f"[S{i}] {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']} {chunk['qualified_name'] or ''}"


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(f"{source_header(i, c)}\n```\n{c['content']}\n```" for i, c in enumerate(chunks, 1))


def fit_sources(chunks: list[dict], budget: int) -> list[dict]:
    """chunks, best first, each cut to SOURCE_CHARS, as many as fit in budget characters (at least one).

    The model sees only these, so a quote is checked against them too: a line cut off is no proof.
    """
    fitted = []
    for i, c in enumerate(chunks, 1):
        room = min(SOURCE_CHARS, budget - len(source_header(i, c)) - 10)
        if room < MIN_SOURCE_CHARS and fitted:
            break
        fitted.append({**c, "content": c["content"][:max(room, 0)]})
        budget -= len(source_header(i, c)) + 10 + len(fitted[-1]["content"])
    return fitted


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


CONTEXT = """
The question comes from an ongoing conversation. Use this to understand what it refers to and what the
user is after; it is not a source, so every statement still needs a quote from the sources below.
{text}
"""


def unknown(question: str, reason: str, nearest: list[dict], detail: str = "", lang: str | None = None,
            **extra) -> dict:
    lang = lang or language(question)
    text = IDK[lang] + (f"\n{detail.strip()}" if detail.strip() else "")
    if nearest:
        text += "\n" + NEAREST[lang] + ", ".join(
            f"{c['qualified_name'] or c['kind']} ({c['file_path']}, {c['score']:.2f})" for c in nearest[:HINTS])
    return {"status": "unknown", "reason": reason, "answer": text, "sources": [], "quotes": [],
            "nearest": [source_of(c, f"N{i}") for i, c in enumerate(nearest[:HINTS], 1)],
            "removed_statements": [], "unverified_names": [], **extra}


def answer(question: str, chunks: list[dict], generate: GenerateJSON, model: str = DEFAULT_ANSWER_LLM,
           nearest: Callable[[], list[dict]] = list, context: str = "", lang: str | None = None,
           template: str = ANSWER_PROMPT) -> dict:
    """{status: answered|unknown, answer, sources, quotes, removed_statements, unverified_names[, reason]}.

    chunks are the retrieved results that passed the relevance threshold; nearest() gives the best
    candidates regardless of it, offered as hints (and as "nearest") when the answer is "I don't know".
    context describes the conversation the question comes from, for a chat; lang ("ru" or "en") sets the
    language of the reply when it is not the question's own. The sources are cut to what fits the
    context window (llm.settings) next to the prompt and the reply.
    """
    if not chunks:
        return unknown(question, "no_relevant_context", nearest(), lang=lang)
    lang = lang or language(question)
    prompt = lambda sources: template.format(  # noqa: E731
        language="Russian" if lang == "ru" else "English", question=question, sources=sources,
        context=CONTEXT.format(text=context) if context else "")
    window = llm.settings.context_window - llm.settings.max_new_tokens
    chunks = fit_sources(chunks, int(window * CHARS_PER_TOKEN) - len(prompt("")))
    reply = generate(model, prompt(format_sources(chunks)), ANSWER_SCHEMA)
    kept, removed = [], []
    for s in reply.get("statements") or []:
        text, quote = str(s.get("text", "")).strip(), str(s.get("quote", "")).strip("\n")
        label = locate(quote, str(s.get("source", "")).strip("[] "), chunks)
        if text and label:
            kept.append({"text": text, "source": label, "quote": quote})
        elif text:
            removed.append({"text": text, "source": s.get("source", ""), "quote": quote})
    if not reply.get("known", False):
        return unknown(question, "model", nearest(), str(reply.get("missing", "")), lang,
                       removed_statements=removed)
    if not kept:
        return unknown(question, "no_verified_quotes", nearest(), lang=lang, removed_statements=removed)
    chunk_of = lambda label: chunks[int(label[1:]) - 1]  # noqa: E731
    quotes = {}
    for s in kept:
        quotes.setdefault((s["source"], bare(s["quote"])), {
            "source": s["source"], "chunk_id": chunk_of(s["source"])["id"],
            "file_path": chunk_of(s["source"])["file_path"], "text": s["quote"]})
    text = " ".join(f"{s['text']} [{s['source']}]" for s in kept)
    return {"status": "answered", "answer": text, "statements": kept, "nearest": [],
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
    return answer(question, chunks, generate_json, answer_llm, nearest)
