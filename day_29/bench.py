"""Day 29: the answer model before and after optimization — quality, speed and memory.

    .venv/bin/python bench.py retrieve    # once: the sources of every question in eval/answers.json
    .venv/bin/python bench.py run NAME    # one configuration, in this process
    .venv/bin/python bench.py             # every configuration, each in its own process, then the table

Retrieval is done once and cached in eval/retrieved.json, so every configuration answers from the same
sources and only the answer step differs. Each configuration runs in a fresh process: its memory peak is
its own and its model loads cold. Results go to eval/bench.json, answers to eval/bench/<name>.log.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from codeindex import answer, evaluate, llm  # noqa: E402

QUESTIONS, RETRIEVED, RESULTS, LOGS = "eval/answers.json", "eval/retrieved.json", "eval/bench.json", Path("eval/bench")
BF16 = "Qwen/Qwen2.5-Coder-3B-Instruct"
Q8, Q4 = "mlx-community/Qwen2.5-Coder-3B-Instruct-8bit", "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
Q4_7B = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

# The answer prompt of day 28, the baseline the case-specific answer.ANSWER_PROMPT is compared with.
DAY_28_PROMPT = """You answer questions about a code base using ONLY the numbered sources below.

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
{context}
Question: {question}

{sources}

Reply in JSON: {{"known": true|false, "statements": [{{"text": "...", "source": "S1", "quote": "..."}}],
"missing": ""}}"""

DAY_28 = {"max_new_tokens": 2000, "context_window": 32768}
TUNED = {}  # llm.Settings() defaults: max_new_tokens 768, context_window 4096
CONFIGS = {  # name: (model, settings, prompt)
    "before: bf16, day 28": (BF16, DAY_28, DAY_28_PROMPT),
    "bf16 + settings": (BF16, TUNED, DAY_28_PROMPT),
    "bf16 + settings + prompt": (BF16, TUNED, answer.ANSWER_PROMPT),
    "8-bit + settings + prompt": (Q8, TUNED, answer.ANSWER_PROMPT),
    "4-bit + settings + prompt": (Q4, TUNED, answer.ANSWER_PROMPT),
    "4-bit, temperature 0.7": (Q4, {**TUNED, "temperature": 0.7, "top_p": 0.9}, answer.ANSWER_PROMPT),
    "7B 4-bit + settings + prompt": (Q4_7B, TUNED, answer.ANSWER_PROMPT),
}


def retrieve() -> None:
    """Sources per question with the `ask` pipeline of day 28 (rewrite, threshold 0.62, LLM reranker)."""
    from demo import ASK, CHUNKING, K
    from codeindex.embedder import default_model
    from codeindex.search import open_store, search

    store, model = open_store(CHUNKING), default_model()
    found = {}
    for q in evaluate.load(QUESTIONS):
        chunks = search(store, CHUNKING, model, q["query"], K, pipeline=ASK)
        found[q["query"]] = [{k: v for k, v in c.items() if k != "vector"} for c in chunks]
        print(f"{len(chunks)} sources: {q['query']}", flush=True)
    Path(RETRIEVED).write_text(json.dumps(found, ensure_ascii=False, indent=1, default=float))


def run(name: str) -> dict:
    model, settings, prompt = CONFIGS[name]
    llm.configure(**settings)
    retrieved = json.loads(Path(RETRIEVED).read_text())
    start = time.perf_counter()
    llm.load(model)
    loaded = time.perf_counter() - start
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"{name.replace(' ', '_').replace(',', '').replace('+', 'and')}.log", "w") as log:
        summary = evaluate.evaluate_answers(
            evaluate.load(QUESTIONS),
            lambda q: answer.answer(q, retrieved[q], llm.generate_json, model, template=prompt),
            None, lambda line: print(line, file=log, flush=True))
    u = llm.usage
    return {"model": model, "settings": vars(llm.settings), **summary, "load s": round(loaded, 1),
            "total s": round(time.perf_counter() - start, 1), "s/answer": round(u["seconds"] / max(u["calls"], 1), 1),
            "prompt tokens/answer": round(u["prompt_tokens"] / max(u["calls"], 1)),
            "reply tokens/answer": round(u["new_tokens"] / max(u["calls"], 1)),
            "memory GB": round(u["peak_memory_gb"], 2)}


def table(results: dict) -> str:
    columns = [c for c in next(iter(results.values())) if c not in ("model", "settings")]
    lines = ["| | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    lines += [f"| {c} | " + " | ".join(str(r.get(c, "")) for r in results.values()) + " |" for c in columns]
    return "\n".join(lines)


def main() -> None:
    if sys.argv[1:2] == ["retrieve"]:
        return retrieve()
    if sys.argv[1:2] == ["run"]:
        return print(json.dumps(run(sys.argv[2]), ensure_ascii=False))
    names = sys.argv[1:] or list(CONFIGS)
    results = json.loads(Path(RESULTS).read_text()) if Path(RESULTS).exists() else {}
    for name in names:
        print(f"running {name}...", flush=True)
        out = subprocess.run([sys.executable, __file__, "run", name], capture_output=True, text=True)
        if out.returncode:
            sys.exit(out.stderr)
        results[name] = json.loads(out.stdout.splitlines()[-1])
        Path(RESULTS).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(table({n: results[n] for n in CONFIGS if n in results}))


if __name__ == "__main__":
    main()
