import os
import tempfile
import unittest
from pathlib import Path

from codeindex import CodeIndexError
from codeindex.chunker import MODES
from codeindex.embedder import Embedder
from codeindex.store import Store
from codeindex.summarizer import Summarizer

from .fakes import DownEmbedder, FakeEmbedder, FakeGenerate
from .fixture import write_tree


class IndexTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo, self.index_dir = Path(tmp.name) / "repo", Path(tmp.name) / "index"
        write_tree(self.repo)
        self.embedder = FakeEmbedder()

    def sync(self, mode="ast", embedder=None, **kw):
        store = Store(self.index_dir)
        return store, store.sync(self.repo, mode, MODES[mode], embedder or self.embedder, **kw)

    def assert_consistent(self, store):
        stats = store.stats()
        self.assertEqual(stats["chunks"], stats["vectors"])
        ids = {r[0] for r in store.conn.execute("SELECT id FROM chunks")}
        for i in ids:
            store.index.reconstruct(i)  # raises if the id is missing from FAISS

    def test_skips_binary_and_ignored_dirs(self):
        store, _ = self.sync()
        paths = {r[0] for r in store.conn.execute("SELECT path FROM files")}
        self.assertEqual(paths, {"pkg/app.py", "web.js"})
        self.assert_consistent(store)
        self.assertEqual(store.meta()["dim"], "64")

    def test_add_update_delete(self):
        store, first = self.sync()
        self.assertEqual(first["files_indexed"], 2)
        self.assertEqual(self.sync()[1]["chunks_embedded"], 0)  # unchanged: nothing re-embedded

        app = self.repo / "pkg" / "app.py"
        app.write_text(app.read_text().replace("return x * 2", "return x * 3"))
        (self.repo / "new.py").write_text("def added():\n    pass\n")
        self.embedder.calls.clear()
        store, result = self.sync()
        self.assertEqual(result["files_indexed"], 2)
        embedded = "\n".join(self.embedder.calls[0])
        self.assertIn("x * 3", embedded)
        self.assertNotIn("function greet", embedded)  # web.js was not re-embedded
        self.assert_consistent(store)

        (self.repo / "web.js").unlink()
        store, result = self.sync()
        self.assertEqual(result["files_deleted"], 1)
        self.assertEqual(store.conn.execute(
            "SELECT count(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = 'web.js'").fetchone()[0], 0)
        self.assert_consistent(store)

    def test_touch_without_change_does_not_reembed(self):
        self.sync()
        app = self.repo / "pkg" / "app.py"
        os.utime(app, (1, 1))
        _, result = self.sync()
        self.assertEqual(result["chunks_embedded"], 0)

    def test_search_filters_and_merges_parts(self):
        store, _ = self.sync("recursive")
        q = self.embedder.embed_query("long_function compute value padding")
        top = store.search(q, 5, merge_parts=True)
        names = [r["qualified_name"] for r in top]
        self.assertEqual(names.count("long_function"), 1)
        unmerged = [r["qualified_name"] for r in store.search(q, 5)]
        self.assertGreater(unmerged.count("long_function"), 1)
        methods = store.search(q, 10, kind="method")
        self.assertTrue(methods and all(r["kind"] == "method" for r in methods))
        inner = store.search(q, 10, class_name="Inner")
        self.assertEqual({r["class_name"] for r in inner}, {"Outer.Inner"})
        self.assertTrue(all(r["file_path"].startswith("pkg/") for r in store.search(q, 10, path_prefix="pkg/")))

    def test_multi_kind_filter(self):
        store, _ = self.sync("multi")
        kinds = store.stats()["kinds"]
        self.assertTrue({"file", "class", "method"} <= set(kinds))
        q = self.embedder.embed_query("outer method")
        self.assertEqual({r["kind"] for r in store.search(q, 10, kind="method")}, {"method"})

    def test_summary_cache(self):
        generate = FakeGenerate()
        summarizer = Summarizer("fake-llm", generate)
        store, _ = self.sync("summary", summarizer=summarizer)
        calls = len(generate.prompts)
        self.assertGreater(calls, 0)
        self.assertIn("# summary: summary 1", self.embedder.calls[0][0])
        store, _ = self.sync("summary", summarizer=summarizer, rebuild=True)
        self.assertEqual(len(generate.prompts), calls)  # every unit came from the cache
        self.assert_consistent(store)

    def test_settings_mismatch_requires_rebuild(self):
        self.sync("ast")
        with self.assertRaisesRegex(CodeIndexError, "--rebuild"):
            self.sync("lines")
        with self.assertRaisesRegex(CodeIndexError, "--rebuild"):
            self.sync("ast", FakeEmbedder(model="other"))
        store, _ = self.sync("lines", rebuild=True)
        self.assertEqual(store.meta()["chunking"], "lines")
        self.assert_consistent(store)

    def test_named_databases_share_a_folder(self):
        docs = Store(self.index_dir, "docs.db")
        docs.sync(self.repo, "lines", MODES["lines"], self.embedder)
        store, _ = self.sync("ast")
        self.assertTrue((self.index_dir / "docs.faiss").exists())
        self.assertTrue((self.index_dir / "vectors.faiss").exists())
        self.assertEqual(Store(self.index_dir, "docs.db").meta()["chunking"], "lines")
        self.assertEqual(store.meta()["chunking"], "ast")
        self.assert_consistent(Store(self.index_dir, "docs.db"))

    def test_inconsistent_stores_require_rebuild(self):
        store, _ = self.sync()
        store.conn.execute("DELETE FROM chunks WHERE id = (SELECT max(id) FROM chunks)")
        store.conn.commit()
        with self.assertRaisesRegex(CodeIndexError, "inconsistent"):
            self.sync()

    def test_embedding_failure_leaves_index_untouched(self):
        self.sync()
        (self.repo / "new.py").write_text("def added():\n    pass\n")
        before = {p.name: p.read_bytes() for p in self.index_dir.iterdir()}
        with self.assertRaisesRegex(CodeIndexError, "not reachable"):
            self.sync(embedder=DownEmbedder())
        self.assertEqual({p.name: p.read_bytes() for p in self.index_dir.iterdir()}, before)

    def test_unreachable_ollama_message(self):
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
        self.addCleanup(os.environ.pop, "OLLAMA_HOST")
        with self.assertRaisesRegex(CodeIndexError, "not reachable.*ollama serve"):
            Embedder("nomic-embed-text").embed_query("x")


if __name__ == "__main__":
    unittest.main()
