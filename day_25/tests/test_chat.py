import tempfile
import unittest
from pathlib import Path

from codeindex.chat import QUERY_SCHEMA, STATE_SCHEMA, Chat, ChatStore, TaskState, update_state
from codeindex.search import without_codebase_name

CHUNK = {"id": 7, "score": 0.7, "file_path": "q.py", "start_line": 1, "end_line": 2, "kind": "method",
         "qualified_name": "QuerySet.paginate", "content": "def paginate(self, page: int, page_size: int = 20):\n    ..."}
NEAR = {**CHUNK, "id": 8, "score": 0.5, "qualified_name": "Model.save"}


class ScriptedLLM:
    """Plays the memory and query calls from a list of replies, and answers by quoting paginate's signature."""

    def __init__(self, states):
        self.states, self.prompts, self.current = list(states), [], None

    def __call__(self, model, prompt, schema):
        self.prompts.append(prompt)
        if schema is STATE_SCHEMA:
            self.current = self.states.pop(0)
            return self.current
        if schema is QUERY_SCHEMA:
            return self.current
        return {"known": True, "missing": "", "statements": [
            {"text": "Use paginate.", "source": "S1", "quote": "def paginate(self, page: int, page_size: int = 20):"}]}


def state_reply(goal="", changed=False, query="paginate", question=True, **lists):
    return {"goal": goal, "goal_changed": changed, "search_query": query, "is_question": question,
            **{k: lists.get(k, []) for k in ("clarified", "constraints", "terms")}}


class TaskStateTest(unittest.TestCase):
    def test_goal_is_kept_unless_the_user_changes_it(self):
        state = update_state(TaskState(), state_reply("paginate orders"))
        self.assertEqual(update_state(state, state_reply("weather in Moscow")).goal, "paginate orders")
        self.assertEqual(update_state(state, state_reply("", changed=True)).goal, "paginate orders")
        self.assertEqual(update_state(state, state_reply("import profiles", changed=True)).goal, "import profiles")

    def test_memory_only_grows(self):
        state = TaskState("g", clarified=["PostgreSQL"], constraints=["no raw SQL"])
        new = update_state(state, state_reply(clarified=["postgresql", "models Order, Customer"], constraints=[]))
        self.assertEqual(new.clarified, ["PostgreSQL", "models Order, Customer"])
        self.assertEqual(new.constraints, ["no raw SQL"])


class QueryTest(unittest.TestCase):
    def test_codebase_name_is_left_out_of_queries(self):
        store = type("FakeStore", (), {"meta": lambda self: {"repo_root": "/src/ormar/ormar"}})()
        self.assertEqual(without_codebase_name("pagination in ORMAR API", store), "pagination in API")
        self.assertEqual(without_codebase_name("ormar", store), "ormar")  # nothing else left: keep it
        self.assertEqual(without_codebase_name("ormarize data", store), "ormarize data")


class ChatTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "chat.db"
        self.searched = []

    def chat(self, llm, found=True):
        def search(query):
            self.searched.append(query)
            return [CHUNK] if found else []
        return Chat(ChatStore(self.db), "s", search, lambda q: [NEAR], llm)

    def test_turns_search_the_standalone_query_and_keep_history_and_memory(self):
        llm = ScriptedLLM([state_reply("paginate orders", query="QuerySet.paginate page size"),
                           state_reply("other", query="paginate select_related", clarified=["PostgreSQL"])])
        chat = self.chat(llm)
        first = chat.send("How do I page through orders?")
        self.assertEqual(self.searched, ["QuerySet.paginate page size"])
        self.assertEqual([s["chunk_id"] for s in first["sources"]], [7])
        second = chat.send("and with customers joined?")
        self.assertEqual(second["state"]["goal"], "paginate orders")
        self.assertIn("user: How do I page through orders?", llm.prompts[3])  # history reaches the memory call
        self.assertIn("- How do I page through orders?", llm.prompts[4])  # and the query call
        self.assertIn("Goal: paginate orders", llm.prompts[5])  # memory reaches the answer
        self.assertIn("Question: paginate select_related", llm.prompts[5])  # asked in the library's terms
        resumed = Chat(ChatStore(self.db), "s", None, None, None)  # a restart keeps both
        self.assertEqual(resumed.state.clarified, ["PostgreSQL"])
        self.assertEqual([m["role"] for m in resumed.store.history("s")], ["user", "assistant"] * 2)
        self.assertEqual(resumed.store.history("s")[1]["sources"][0]["chunk_id"], 7)

    def test_dont_know_still_shows_the_nearest_sources(self):
        chat = self.chat(ScriptedLLM([state_reply("paginate orders", query="weather")]), found=False)
        result = chat.send("Какая погода завтра?")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual([s["section"] for s in result["nearest"]], ["Model.save"])
        self.assertEqual(chat.store.history("s")[1]["sources"][0]["chunk_id"], 8)

    def test_a_message_that_asks_nothing_is_noted_with_sources(self):
        llm = ScriptedLLM([state_reply("paginate orders"),
                           state_reply("paginate orders", query="postgresql backend", question=False,
                                       clarified=["PostgreSQL"])])
        chat = self.chat(llm)
        chat.send("How do I page through orders?")
        result = chat.send("We use PostgreSQL.")
        self.assertEqual(result["status"], "noted")
        self.assertIn("clarified: PostgreSQL", result["answer"])
        self.assertEqual([s["chunk_id"] for s in result["sources"]], [7])
        self.assertEqual(len(llm.prompts), 5)  # no answer call for it

    def test_a_statement_that_adds_to_the_memory_is_noted_even_if_the_model_calls_it_a_question(self):
        llm = ScriptedLLM([state_reply("paginate orders"),
                           state_reply("paginate orders", clarified=["PostgreSQL"], question=True)])
        chat = self.chat(llm)
        chat.send("How do I page through orders?")
        self.assertEqual(chat.send("We use PostgreSQL.")["status"], "noted")

    def test_a_question_mark_makes_a_question_whatever_the_model_says(self):
        llm = ScriptedLLM([state_reply("paginate orders", query="weather", question=False)])
        self.assertEqual(self.chat(llm, found=False).send("Какая завтра погода?")["status"], "unknown")

    def test_the_answer_is_in_the_language_of_the_user(self):
        llm = ScriptedLLM([state_reply("пагинация заказов", query="paginate results page size")])
        result = self.chat(llm, found=False).send("Как выводить заказы по страницам?")
        self.assertTrue(result["answer"].startswith("Не знаю"))  # though the query it searched is English

    def test_past_dont_know_is_not_shown_in_full(self):
        llm = ScriptedLLM([state_reply("paginate orders", query="weather"), state_reply("paginate orders")])
        chat = self.chat(llm, found=False)
        chat.send("Какая погода завтра?")
        chat.search = lambda q: [CHUNK]
        chat.send("How do I page through orders?")
        self.assertIn("assistant: (no answer was found)", llm.prompts[2])  # memory call of turn 2
        self.assertNotIn("Не знаю", llm.prompts[2])

    def test_new_session_starts_empty(self):
        chat = self.chat(ScriptedLLM([state_reply("paginate orders")]))
        chat.send("How do I page through orders?")
        chat.store.reset("s")
        self.assertEqual((chat.state, chat.store.history("s")), (TaskState(), []))


if __name__ == "__main__":
    unittest.main()
