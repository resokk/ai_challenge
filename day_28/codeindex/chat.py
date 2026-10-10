"""A chat over the index: dialogue history, task memory, and an answer with sources on every turn.

Each user message goes through three model calls. The first updates the task memory (the goal of the
conversation, what the user has clarified, their constraints and terms). The second turns the message,
which may lean on earlier ones ("and for them?"), into a standalone query in the library's terms, and
tells whether it asks anything at all; a message that only informs is acknowledged without an answer.
The third answers that query from what it retrieves, as answer.answer() does for a single question, in
the user's language and with the task memory as context. The memory's invariants are kept by code, not left to the model: the goal
changes only when the user sets a new one, and nothing recorded is dropped.

History and memory live in SQLite, one session per conversation, so a chat can be resumed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .answer import DEFAULT_ANSWER_LLM, GenerateJSON, answer, language, source_of

HISTORY_TURNS = 3  # earlier exchanges shown to the model besides the task memory
HISTORY_CHARS = 600  # per message
MEMORY_LISTS = ("clarified", "constraints", "terms")

STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "goal_changed": {"type": "boolean"},
        "clarified": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal", "goal_changed", "clarified", "constraints", "terms"],
}

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"is_question": {"type": "boolean"}, "search_query": {"type": "string"}},
    "required": ["is_question", "search_query"],
}

STATE_PROMPT = """You keep the task memory of a conversation about a code base (a Python library).

Task memory so far:
{memory}

Recent conversation:
{history}

Latest user message: {message}

Update the task memory:
- "goal": what the user wants to achieve in this conversation as a whole, in one sentence. On the first
  message, take it from that message. Later keep it as it is; set "goal_changed" to true only if the
  user explicitly says they now want something else. A side question does not change the goal.
- "clarified": facts the user has told about their own situation (their database, models, data, setup),
  one fact per item; not the goal or the questions themselves.
- "constraints": requirements and limits the user has set (e.g. "no raw SQL", "no duplicates").
- "terms": the user's own terms with their meaning, as "term: meaning".
Keep every earlier item and add the new ones from the latest message; do not add things the user did
not say. Write the goal and the items in {language}.

Reply in JSON: {{"goal": "...", "goal_changed": false, "clarified": [], "constraints": [], "terms": []}}"""

QUERY_PROMPT = """Turn the user's latest message in a conversation about a Python library into a query for
searching the library's source code.

The user's goal: {goal}
Earlier messages from the user:
{earlier}

Latest message: {message}

"is_question": false if the latest message only tells something (a fact about their setup, a constraint,
a term) and asks for nothing; true if it asks or requests anything.

"search_query": an English query for what the latest message is about, written as the library operation,
never in the user's domain words: the user's models (orders, customers, profiles...) are not in the
library's code, so say what the library does with any model instead. Resolve "it", "them", "that" from
the earlier messages. Add likely method or class names. Do not mention the library's name.
Examples:
- "How do I get only the orders of one customer?" -> "filter queryset by related model field value"
- "And load the customer in the same query?" -> "select_related load related model in the same query join"
- "Sort them newest first." -> "order_by descending sort by field"
- "How many orders are there in total?" -> "count rows matching query"
- "We use PostgreSQL." -> "database backend postgresql dialect"
- "How do I get the list of orders page by page?" -> "paginate results page size limit offset"

Reply in JSON: {{"is_question": true, "search_query": "..."}}"""

NOTED = {"ru": ("Записал в память задачи:", "Принял, в памяти задачи это уже есть."),
         "en": ("Added to the task memory:", "Noted, the task memory already has this.")}
LABELS = {"ru": {"goal": "цель", "clarified": "уточнено", "constraints": "ограничение", "terms": "термин"},
          "en": {"goal": "goal", "clarified": "clarified", "constraints": "constraint", "terms": "term"}}


@dataclass
class TaskState:
    goal: str = ""
    clarified: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Goal: {self.goal or '(not set yet)'}"]
        for name in MEMORY_LISTS:
            items = getattr(self, name)
            lines.append(f"{name.capitalize()}: " + ("; ".join(items) if items else "(none)"))
        return "\n".join(lines)


def merged(old: list[str], new: list[str]) -> list[str]:
    """old, then the items of new not already in it: the model can add to the memory, never drop from it.

    Items the model joined with ";" count one by one.
    """
    seen = {item.strip().lower() for item in old}
    out = list(old)
    for item in (part for joined in new for part in joined.split(";")):
        if item.strip() and item.strip().lower() not in seen:
            seen.add(item.strip().lower())
            out.append(item.strip())
    return out


def update_state(state: TaskState, reply: dict) -> TaskState:
    goal = str(reply.get("goal", "")).strip()
    keep_goal = state.goal and not (reply.get("goal_changed") and goal)
    return TaskState(goal=state.goal if keep_goal else goal or state.goal,
                     **{name: merged(getattr(state, name), [str(i) for i in reply.get(name) or []])
                        for name in MEMORY_LISTS})


def render_history(messages: list[dict]) -> str:
    """The last exchanges; an "I don't know" shows as such, not in full, so the model does not echo it."""
    recent = messages[-2 * HISTORY_TURNS:]
    if not recent:
        return "(this is the first message)"
    return "\n".join(f"{m['role']}: " + ("(no answer was found)" if m.get("status") == "unknown"
                                         else m["content"][:HISTORY_CHARS]) for m in recent)


def noted(message: str, before: TaskState, after: TaskState, found: list[dict], nearest: list[dict]) -> dict:
    """The reply to a message that asks nothing: what went into the task memory, and the code found on it."""
    lang = language(message)
    added = ([f"{LABELS[lang]['goal']}: {after.goal}"] if after.goal != before.goal else []) + [
        f"{LABELS[lang][name]}: {item}" for name in MEMORY_LISTS for item in getattr(after, name)[len(getattr(before, name)):]]
    text = "\n".join([NOTED[lang][0], *(f"- {a}" for a in added)]) if added else NOTED[lang][1]
    return {"status": "noted", "answer": text, "statements": [], "quotes": [], "removed_statements": [],
            "unverified_names": [], "sources": [source_of(c, f"S{i}") for i, c in enumerate(found, 1)],
            "nearest": [] if found else [source_of(c, f"N{i}") for i, c in enumerate(nearest, 1)]}


def respond(message: str, state: TaskState, history: list[dict],
            search: Callable[[str], list[dict]], nearest: Callable[[str], list[dict]],
            generate: GenerateJSON, state_llm: str = DEFAULT_ANSWER_LLM,
            answer_llm: str = DEFAULT_ANSWER_LLM) -> tuple[TaskState, dict]:
    """The task memory after message, and the answer to it (as answer.answer() returns, plus "query")."""
    reply = generate(state_llm, STATE_PROMPT.format(memory=state.render(), history=render_history(history),
                                                    message=message,
                                                    language="Russian" if language(message) == "ru" else "English"),
                     STATE_SCHEMA)
    before, state = state, update_state(state, reply)
    # A separate, single-purpose call: asked together with the memory, the model kept the user's
    # domain words ("orders") in the query, and they match unrelated code ("order_by").
    earlier = "\n".join(f"- {m['content'][:HISTORY_CHARS]}" for m in history if m["role"] == "user")
    plan = generate(state_llm, QUERY_PROMPT.format(goal=state.goal or "(not set yet)", message=message,
                                                   earlier=earlier or "(none)"), QUERY_SCHEMA)
    query = str(plan.get("search_query", "")).strip() or message
    found = search(query)
    # The model's call is not trusted on its own: it took "We use PostgreSQL" for a question and a
    # question about the weather for a statement. A "?" makes a question; without one, a message is a
    # statement if the model says so or if it added to the memory.
    informs = "?" not in message and (plan.get("is_question") is False or asdict(state) != asdict(before))
    if informs:
        return state, {**noted(message, before, state, found, [] if found else nearest(query)), "query": query}
    # The answer model gets the question in the library's terms, with references already resolved; given
    # the user's own words ("orders of a customer") and the history, it looked for those in the code,
    # found nothing, and said it did not know, or answered an earlier question instead.
    context = (f"The user's message was: {message}\nTheir task memory:\n{state.render()}\n"
               f"The user's own models and data are not in the sources: answer how the library does this "
               f"for any model.")
    result = answer(query, found, generate, answer_llm, lambda: nearest(query), context, language(message))
    return state, {**result, "query": query}


class ChatStore:
    """Sessions with their task memory and messages, in one SQLite file."""

    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, session TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT, content TEXT, status TEXT, sources TEXT);
        """)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def state(self, session: str) -> TaskState:
        row = self.conn.execute("SELECT state FROM sessions WHERE id = ?", (session,)).fetchone()
        return TaskState(**json.loads(row[0])) if row else TaskState()

    def history(self, session: str) -> list[dict]:
        rows = self.conn.execute("SELECT role, content, status, sources FROM messages WHERE session = ? ORDER BY id",
                                 (session,))
        return [{"role": r, "content": c, "status": st, "sources": json.loads(s or "[]")} for r, c, st, s in rows]

    def save_turn(self, session: str, state: TaskState, message: str, result: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET state = excluded.state",
                              (session, json.dumps(asdict(state), ensure_ascii=False)))
            self.conn.execute("INSERT INTO messages (session, role, content) VALUES (?, 'user', ?)", (session, message))
            self.conn.execute("INSERT INTO messages (session, role, content, status, sources)"
                              " VALUES (?, 'assistant', ?, ?, ?)", (session, result["answer"], result["status"],
                               json.dumps(result["sources"] or result["nearest"], ensure_ascii=False)))

    def reset(self, session: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (session,))


class Chat:
    """One session: send() answers a message and records the turn."""

    def __init__(self, store: ChatStore, session: str, search: Callable[[str], list[dict]],
                 nearest: Callable[[str], list[dict]], generate: GenerateJSON,
                 state_llm: str = DEFAULT_ANSWER_LLM, answer_llm: str = DEFAULT_ANSWER_LLM):
        self.store, self.session = store, session
        self.search, self.nearest, self.generate = search, nearest, generate
        self.state_llm, self.answer_llm = state_llm, answer_llm

    @property
    def state(self) -> TaskState:
        return self.store.state(self.session)

    def send(self, message: str) -> dict:
        state, result = respond(message, self.state, self.store.history(self.session), self.search,
                                self.nearest, self.generate, self.state_llm, self.answer_llm)
        self.store.save_turn(self.session, state, message, result)
        return {**result, "state": asdict(state)}
