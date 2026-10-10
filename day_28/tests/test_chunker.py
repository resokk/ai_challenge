import importlib.util
import unittest

from codeindex import chunker
from codeindex.chunker import MAX_CHARS, MODES

from .fixture import APP_JS, APP_PY

PATH = "pkg/app.py"


def line_of(fragment, text=APP_PY):
    return next(i for i, line in enumerate(text.splitlines(), 1) if fragment in line)


def by_name(chunks):
    return {c.qualified_name: c for c in chunks if c.part is None}


class AstTest(unittest.TestCase):
    def test_units_and_metadata(self):
        chunks = by_name(MODES["ast"](PATH, APP_PY))
        m = chunks["Outer.Inner.nested_method"]
        self.assertEqual((m.kind, m.class_name, m.symbol_name), ("method", "Outer.Inner", "nested_method"))
        self.assertEqual((m.start_line, m.end_line), (line_of("def nested_method"), line_of("os.getpid")))
        self.assertEqual(chunks["Outer.Inner"].kind, "class")
        self.assertEqual(chunks["Outer"].class_name, "Outer")
        self.assertEqual(chunks["helper"].kind, "function")
        self.assertIsNone(chunks["helper"].class_name)
        module = chunks["pkg.app"]
        self.assertEqual(module.kind, "module")
        self.assertIn("LIMIT = 3", module.content)
        self.assertNotIn("def helper", module.content)
        self.assertEqual(module.end_line, line_of("print(helper(LIMIT))"))
        self.assertEqual(len(m.content_hash), 64)
        self.assertTrue(m.embed_text().startswith("# file: pkg/app.py\n# symbol: Outer.Inner.nested_method\n"))

    def test_non_python_falls_back_to_lines(self):
        chunks = MODES["ast"]("web.js", APP_JS)
        self.assertEqual([(c.kind, c.qualified_name, c.language) for c in chunks], [("block", None, "javascript")])


class LinesAndTokensTest(unittest.TestCase):
    def test_lines_windows_overlap(self):
        chunks = MODES["lines"](PATH, APP_PY)
        self.assertEqual([(c.start_line, c.end_line) for c in chunks[:2]], [(1, 60), (51, 110)])
        self.assertEqual(chunks[-1].end_line, len(APP_PY.splitlines()))
        self.assertTrue(all(c.kind == "block" and c.symbol_name is None for c in chunks))

    def test_tokens_windows_snap_to_lines(self):
        lines = APP_PY.splitlines()
        chunks = MODES["tokens"](PATH, APP_PY)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c.content, "\n".join(lines[c.start_line - 1:c.end_line]))
        for a, b in zip(chunks, chunks[1:]):
            self.assertLess(b.start_line, a.end_line + 1)  # overlap
            self.assertGreater(b.start_line, a.start_line)
        self.assertEqual(chunks[-1].end_line, len(lines))


class RecursiveTest(unittest.TestCase):
    def test_long_function_splits_into_consecutive_parts(self):
        parts = [c for c in MODES["recursive"](PATH, APP_PY) if c.qualified_name == "long_function"]
        self.assertGreater(len(parts), 1)
        self.assertEqual([p.part for p in parts], list(range(1, len(parts) + 1)))
        self.assertEqual(parts[0].start_line, line_of("def long_function"))
        self.assertEqual(parts[-1].end_line, line_of("return value_119"))
        for a, b in zip(parts, parts[1:]):
            self.assertEqual(b.start_line, a.end_line + 1)
        self.assertTrue(all(len(p.content) <= MAX_CHARS for p in parts))
        self.assertTrue(all(p.kind == "function" for p in parts))
        short = by_name(MODES["recursive"](PATH, APP_PY))["helper"]
        self.assertIsNone(short.part)


class ContextualTest(unittest.TestCase):
    def test_context_is_embedded_not_stored(self):
        m = by_name(MODES["contextual"](PATH, APP_PY))["Outer.Inner.nested_method"]
        self.assertIn("# imports: import os; from pathlib import Path", m.context)
        self.assertIn("# class: class Inner()", m.context)
        self.assertIn("# calls: helper, os.getpid", m.context)
        self.assertNotIn("# imports", m.content)
        self.assertIn(m.context, m.embed_text())
        outer = by_name(MODES["contextual"](PATH, APP_PY))["Outer.outer_method"]
        self.assertIn("# class docstring: Outer docstring.", outer.context)


class MultiTest(unittest.TestCase):
    def test_levels(self):
        chunks = MODES["multi"](PATH, APP_PY)
        kinds = {(c.kind, c.qualified_name) for c in chunks}
        self.assertIn(("file", "pkg.app"), kinds)
        self.assertIn(("class", "Outer"), kinds)
        self.assertIn(("method", "Outer.outer_method"), kinds)
        self.assertNotIn("module", {c.kind for c in chunks})
        outer = by_name(chunks)["Outer"]
        self.assertIn("def outer_method", outer.content)  # class level holds the whole class
        self.assertEqual(by_name(chunks)["pkg.app"].content, APP_PY)


@unittest.skipUnless(importlib.util.find_spec("tree_sitter_language_pack"), "tree-sitter not installed")
class TreeSitterTest(unittest.TestCase):
    def test_python_and_javascript(self):
        py = by_name(MODES["treesitter"](PATH, APP_PY))
        m = py["Outer.Inner.nested_method"]
        self.assertEqual((m.kind, m.class_name, m.start_line), ("method", "Outer.Inner", line_of("def nested_method")))
        js = by_name(MODES["treesitter"]("web.js", APP_JS))
        self.assertEqual(js["greet"].kind, "function")
        self.assertEqual((js["Greeter.hello"].kind, js["Greeter.hello"].class_name), ("method", "Greeter"))

    def test_unknown_language_falls_back_to_lines(self):
        chunks = MODES["treesitter"]("notes.txt", "a\nb\n")
        self.assertEqual([c.kind for c in chunks], ["block"])


class SummaryTest(unittest.TestCase):
    def test_summary_mode_uses_ast_units(self):
        self.assertEqual(MODES["summary"](PATH, APP_PY), MODES["ast"](PATH, APP_PY))

    def test_every_mode_registered(self):
        self.assertEqual(set(MODES), {"ast", "lines", "treesitter", "recursive", "tokens", "contextual",
                                      "multi", "summary"})
        self.assertIs(MODES["lines"], chunker.chunk_lines)


if __name__ == "__main__":
    unittest.main()
