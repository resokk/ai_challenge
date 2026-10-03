import tempfile
import unittest
from pathlib import Path

from codeindex.chunker import MODES
from codeindex.evaluate import evaluate, pipelines
from codeindex.rerank import heuristic_rerank, llm_rerank, rewrite_query, terms
from codeindex.search import Pipeline, run
from codeindex.store import Store

from .fakes import FakeEmbedder
from .fixture import write_tree


def result(name, score, content=""):
    return {"qualified_name": name, "file_path": "a.py", "kind": "function", "score": score, "content": content}


class ScriptedLLM:
    """Answers rewrite prompts with a fixed query and rerank prompts by the result's name."""

    def __init__(self, rewrite="", ratings=None):
        self.rewrite, self.ratings, self.prompts = rewrite, ratings or {}, []

    def __call__(self, model, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Rewrite"):
            return self.rewrite
        return next((str(v) for name, v in self.ratings.items() if f"Result: {name} in" in prompt), "no idea")


class StagesTest(unittest.TestCase):
    def test_terms_split_identifiers(self):
        self.assertEqual(terms("QuerySet.order_by rows"), {"query", "set", "order", "row"})

    def test_rewrite_takes_first_line_or_keeps_query(self):
        self.assertEqual(rewrite_query("q", ScriptedLLM('\n"delete row"\nextra')), "delete row")
        self.assertEqual(rewrite_query("q", ScriptedLLM("")), "q")

    def test_heuristic_lifts_name_matches(self):
        ranked = heuristic_rerank("paginate results", [result("helper", 0.70), result("paginate", 0.67)])
        self.assertEqual([r["qualified_name"] for r in ranked], ["paginate", "helper"])

    def test_llm_rerank_orders_and_drops(self):
        llm = ScriptedLLM(ratings={"a": 3, "b": 9, "c": 7})
        ranked = llm_rerank("q", [result("a", 0.9), result("b", 0.6), result("c", 0.7), result("d", 0.8)], llm,
                            min_relevance=5)
        self.assertEqual([(r["qualified_name"], r["rerank_score"]) for r in ranked], [("b", 9), ("c", 7)])


class PipelineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        write_tree(repo)
        self.embedder = FakeEmbedder(dim=1024)  # few hash collisions: unrelated words score 0
        self.store = Store(Path(tmp.name) / "index")
        self.store.sync(repo, "ast", MODES["ast"], self.embedder)

    def run_query(self, query, pipeline, llm=None, k=3):
        return run(self.store, self.embedder.embed_query, llm or ScriptedLLM(), query, k, pipeline)

    def test_default_is_plain_search(self):
        results = self.run_query("helper", Pipeline())
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["qualified_name"], "helper")
        self.assertNotIn("rerank_score", results[0])

    def test_threshold_drops_everything_unrelated(self):
        self.assertEqual(self.run_query("weather forecast tomorrow", Pipeline(min_score=0.3)), [])
        self.assertTrue(self.run_query("helper", Pipeline(min_score=0.3)))

    def test_rewrite_merges_candidates_of_both_queries(self):
        llm = ScriptedLLM(rewrite="nested_method")
        names = {r["qualified_name"] for r in self.run_query("удалить", Pipeline(rewrite=True, min_score=0.3), llm)}
        self.assertIn("Outer.Inner.nested_method", names)
        self.assertEqual(len({r["qualified_name"] for r in self.run_query("helper", Pipeline(rewrite=True), llm,
                                                                          k=50)}),
                         len(self.run_query("helper", Pipeline(rewrite=True), llm, k=50)))  # no duplicates

    def test_fetch_k_widens_candidates_before_k_cuts(self):
        llm = ScriptedLLM(ratings={"long_function": 10})
        narrow = self.run_query("helper", Pipeline(rerank="llm", fetch_k=1), llm, k=1)
        wide = self.run_query("helper", Pipeline(rerank="llm", fetch_k=50), llm, k=1)
        self.assertEqual(narrow, [])
        self.assertEqual(wide[0]["qualified_name"], "long_function")

    def test_evaluate_metrics(self):
        questions = [{"query": "helper", "relevant": ["helper"]},
                     {"query": "nested_method", "relevant": ["nothing such"]},
                     {"query": "weather forecast tomorrow", "relevant": []}]
        search = lambda q, k, p: self.run_query(q, p, k=k)  # noqa: E731
        plain = evaluate(questions, search, Pipeline(), 3)
        self.assertEqual((plain["hit@k"], plain["MRR"], plain["rejected"]), (0.5, 0.5, 0.0))
        filtered = evaluate(questions, search, Pipeline(min_score=0.3), 3)
        self.assertEqual(filtered["rejected"], 1.0)

    def test_compared_modes(self):
        modes = pipelines(0.6, 20, 5)
        self.assertEqual(modes["baseline"], Pipeline())
        self.assertEqual(modes["rewrite+llm"], Pipeline(rewrite=True, min_score=0.6, rerank="llm", fetch_k=20))


if __name__ == "__main__":
    unittest.main()
