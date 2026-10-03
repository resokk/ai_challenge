import unittest

from codeindex.answer import answer, locate, unverified_names

CHUNKS = [
    {"id": 10, "score": 0.71, "file_path": "q.py", "start_line": 1, "end_line": 4, "kind": "method",
     "qualified_name": "QuerySet.paginate",
     "content": 'def paginate(self, page: int, page_size: int = 20):\n    """\n    Paginate the result.\n    """\n'
                "    limit_count = page_size\n    query_offset = (page - 1) * page_size"},
    {"id": 11, "score": 0.66, "file_path": "q.py", "start_line": 5, "end_line": 6, "kind": "method",
     "qualified_name": "QuerySet.count", "content": "async def count(self) -> int:\n    return await self.query()"},
]
NEAREST = [{"qualified_name": "Model.save", "kind": "method", "file_path": "m.py", "score": 0.55}]


class Scripted:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, model, prompt, schema):
        self.prompts.append(prompt)
        return self.reply


def statement(text, source, quote):
    return {"text": text, "source": source, "quote": quote}


class LocateTest(unittest.TestCase):
    def test_quotes_must_be_verbatim(self):
        self.assertEqual(locate("query_offset   = (page - 1) *\n page_size", "S1", CHUNKS), "S1")  # reflowed
        self.assertEqual(locate('"""Paginate the result."""', "S1", CHUNKS), "S1")  # docstring joined onto one line
        self.assertEqual(locate("limit_count = page_size.", "S1", CHUNKS), "S1")  # its own trailing period
        self.assertEqual(locate("async def count(self) -> int:", "S1", CHUNKS), "S2")  # credited to the wrong source
        self.assertIsNone(locate("return len(self.rows)", "S2", CHUNKS))  # invented
        self.assertIsNone(locate("page_size", "S1", CHUNKS))  # too short to prove anything

    def test_names_missing_from_sources_are_flagged(self):
        text = "Call `paginate(page, page_size)` with `True` or ```python\nMyModel.objects.paginate(page=1)\n```"
        self.assertEqual(unverified_names(text, CHUNKS), ["MyModel", "objects"])


class AnswerTest(unittest.TestCase):
    def test_statements_without_a_verbatim_quote_are_cut(self):
        llm = Scripted({"known": True, "missing": "", "statements": [
            statement("Use paginate(page, page_size).", "S1", "def paginate(self, page: int, page_size: int = 20):"),
            statement("It caches pages.", "S1", "self.cache[page] = rows"),
            statement("count returns an int.", "S1", "async def count(self) -> int:"),
        ]})
        result = answer("how to paginate", CHUNKS, llm)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Use paginate(page, page_size). [S1] count returns an int. [S2]")
        self.assertEqual([(s["source"], s["chunk_id"], s["section"], s["lines"]) for s in result["sources"]],
                         [("S1", 10, "QuerySet.paginate", "1-4"), ("S2", 11, "QuerySet.count", "5-6")])
        self.assertEqual([q["chunk_id"] for q in result["quotes"]], [10, 11])
        self.assertEqual([s["text"] for s in result["removed_statements"]], ["It caches pages."])
        self.assertIn("[S1] q.py:1-4 QuerySet.paginate", llm.prompts[0])

    def test_no_context_means_dont_know_without_asking_the_model(self):
        llm = Scripted({})
        result = answer("прогноз погоды на завтра", [], llm, nearest=lambda: NEAREST)
        self.assertEqual((result["status"], result["reason"]), ("unknown", "no_relevant_context"))
        self.assertTrue(result["answer"].startswith("Не знаю"))
        self.assertIn("Уточните", result["answer"])
        self.assertIn("Model.save", result["answer"])
        self.assertEqual(llm.prompts, [])

    def test_answer_left_without_statements_becomes_dont_know(self):
        llm = Scripted({"known": True, "missing": "", "statements": [
            statement("It uses a cursor.", "S1", "cursor.fetch(page)")]})
        result = answer("how to paginate", CHUNKS, llm)
        self.assertEqual((result["status"], result["reason"]), ("unknown", "no_verified_quotes"))
        self.assertTrue(result["answer"].startswith("I don't know"))
        self.assertEqual((result["sources"], result["quotes"]), ([], []))
        self.assertEqual(len(result["removed_statements"]), 1)

    def test_model_may_say_it_does_not_know(self):
        llm = Scripted({"known": False, "statements": [], "missing": "Nothing about MongoDB; which backend?"})
        result = answer("connect to MongoDB", CHUNKS, llm)
        self.assertEqual((result["status"], result["reason"]), ("unknown", "model"))
        self.assertTrue(result["answer"].startswith("I don't know"))
        self.assertIn("which backend", result["answer"])


if __name__ == "__main__":
    unittest.main()
