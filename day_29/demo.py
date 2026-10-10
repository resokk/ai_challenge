"""Demo of day 29: search, cited answers, "I don't know" and the chat on the optimized local model
(Qwen2.5-Coder-3B quantized to 8 bits on MLX, tuned generation settings, the task's own answer prompt).

    .venv/bin/python demo.py

Runs in one process, so each model is loaded once. Needs the index (python -m codeindex index ...);
the chat runs in a throwaway session and does not touch index/chat.db.
"""

import os
import sys
import tempfile
import time
from dataclasses import asdict

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from codeindex import CodeIndexError, answer, llm  # noqa: E402
from codeindex.__main__ import print_reply, print_state_changes  # noqa: E402
from codeindex.chat import Chat, ChatStore  # noqa: E402
from codeindex.embedder import default_model  # noqa: E402
from codeindex.llm import device, generate_json  # noqa: E402
from codeindex.rerank import DEFAULT_LLM  # noqa: E402
from codeindex.search import Pipeline, open_store, search, without_codebase_name  # noqa: E402

CHUNKING, K = "ast", 5
ASK = Pipeline(rewrite=True, min_score=0.62, rerank="llm", fetch_k=20)  # the defaults of `ask`
CHAT = Pipeline(min_score=0.62, rerank="llm", fetch_k=20)  # and of `chat`
CHAT_TURNS = [
    "Делаю API заказов. Как получить список заказов постранично?",
    "База у нас PostgreSQL.",
    "А как получить только заказы одного покупателя?",
]


def step(title: str) -> float:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)
    return time.monotonic()


def took(start: float) -> None:
    print(f"\n({time.monotonic() - start:.1f} с)", flush=True)


def main() -> None:
    store = open_store(CHUNKING)
    model = default_model()

    start = step("1. Модели: всё локально, LLM квантована до 8 бит (MLX)")
    print(f"устройство: {device()}\nэмбеддинги: {model}\nответы и память задачи: {answer.DEFAULT_ANSWER_LLM}\n"
          f"переписывание и реранкер: {DEFAULT_LLM}\nнастройки генерации: {llm.settings}")
    stats = store.stats()
    print(f"индекс: {stats['files']} файлов, {stats['chunks']} кусков кода из {stats['repo_root']}")

    query = "как удалить запись из базы"
    start = step(f"2. Поиск: «{query}» (переписывание + порог 0.62 + LLM-реранкер)")
    for r in search(store, CHUNKING, model, query, 3, pipeline=ASK):
        print(f"  {r['score']:.3f}  оценка {r['rerank_score']}/10  {r['qualified_name']}  "
              f"({r['file_path']}:{r['start_line']}-{r['end_line']})")
    took(start)

    for question in ("how do I paginate query results", "прогноз погоды на завтра"):
        start = step(f"3. Ответ с источниками и проверенными цитатами: «{question}»")
        print_reply(answer.ask(store, CHUNKING, model, question, K, ASK))
        took(start)

    with tempfile.TemporaryDirectory() as tmp:
        chat = Chat(ChatStore(os.path.join(tmp, "chat.db")), "demo",
                    lambda q: search(store, CHUNKING, model, without_codebase_name(q, store), K, pipeline=CHAT),
                    lambda q: search(store, CHUNKING, model, without_codebase_name(q, store), answer.HINTS),
                    generate_json)
        for i, message in enumerate(CHAT_TURNS, 1):
            start = step(f"4.{i} Чат с памятью задачи\n> {message}")
            before = asdict(chat.state)
            result = chat.send(message)
            print(f"(поисковый запрос: {result['query']})")
            print_reply(result)
            if result["status"] != "noted":
                print_state_changes(before, result["state"])
            took(start)
        step("Память задачи в конце разговора")
        print(chat.state.render())
    u = llm.usage
    step("Расход LLM за демо")
    print(f"{u['calls']} вызовов, {u['prompt_tokens']} токенов промпта, {u['new_tokens']} токенов ответа, "
          f"{u['seconds']:.0f} с генерации, пик памяти {u['peak_memory_gb']:.2f} ГБ")


if __name__ == "__main__":
    try:
        main()
    except CodeIndexError as e:
        sys.exit(f"error: {e}")
