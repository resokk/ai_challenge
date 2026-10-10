"""Chunking modes. Every mode is chunk(path, text) -> list[Chunk]; MODES maps names to them.

Modes that need a Python AST fall back to fixed line windows for other files.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from . import CodeIndexError

MAX_CHARS = 6000  # recursive: units longer than this are split
CLASS_BODY_CHARS = 1500  # ast: a class shorter than this is chunked with its body
LINES_WINDOW, LINES_OVERLAP = 60, 10
TOKENS_WINDOW, TOKENS_OVERLAP = 512, 64
TOKEN_RE = re.compile(r"\w+|[^\s\w]")

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust", ".java": "java",
    ".kt": "kotlin", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".sh": "bash", ".lua": "lua", ".sql": "sql", ".html": "html", ".css": "css",
    ".md": "markdown", ".toml": "toml", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
}


def language_of(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


@dataclass
class Chunk:
    file_path: str
    language: str
    kind: str  # file | module | class | function | method | block
    start_line: int
    end_line: int
    content: str
    class_name: str | None = None
    symbol_name: str | None = None
    qualified_name: str | None = None
    part: int | None = None
    context: str = ""  # embedded but never stored or shown (contextual mode)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()

    def header(self) -> str:
        return f"# file: {self.file_path}\n# symbol: {self.qualified_name or ''}\n"

    def embed_text(self) -> str:
        return self.header() + self.context + self.content


# --- lines and tokens: structure-free windows --------------------------------


def chunk_lines(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    language, out = language_of(path), []
    for start in range(0, len(lines), LINES_WINDOW - LINES_OVERLAP):
        end = min(start + LINES_WINDOW, len(lines))
        content = "\n".join(lines[start:end])
        if content.strip():
            out.append(Chunk(path, language, "block", start + 1, end, content))
        if end == len(lines):
            break
    return out


def chunk_tokens(path: str, text: str) -> list[Chunk]:
    """~512-token windows with ~64 tokens of overlap, snapped to whole lines."""
    lines = text.splitlines()
    counts = [len(TOKEN_RE.findall(line)) for line in lines]
    language, out, start = language_of(path), [], 0
    while start < len(lines):
        end, total = start, 0
        while end < len(lines) and (end == start or total + counts[end] <= TOKENS_WINDOW):
            total += counts[end]
            end += 1
        content = "\n".join(lines[start:end])
        if content.strip():
            out.append(Chunk(path, language, "block", start + 1, end, content))
        if end == len(lines):
            break
        # Start the next window far enough back to repeat ~TOKENS_OVERLAP tokens.
        nxt, back = end, 0
        while nxt > start + 1 and back < TOKENS_OVERLAP:
            nxt -= 1
            back += counts[nxt]
        start = nxt
    return out


# --- Python AST units ---------------------------------------------------------


@dataclass
class _Unit:
    chunk: Chunk
    line_nos: list[int]  # file line number of each content line
    stmts: list[ast.AST]  # statements whose starts are split points (recursive mode)
    walk: list[ast.AST]  # nodes that make up the unit (contextual mode's calls)
    classes: list[ast.ClassDef]  # enclosing classes, outermost first


def _start(node: ast.AST) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _uncovered(lines: list[str], covered: set[int]) -> list[int]:
    """Line numbers outside definitions, without leading or trailing blank lines."""
    keep = [n for n in range(1, len(lines) + 1) if n not in covered]
    while keep and not lines[keep[0] - 1].strip():
        keep.pop(0)
    while keep and not lines[keep[-1] - 1].strip():
        keep.pop()
    return keep


def _py_units(path: str, text: str, full_classes: bool = False) -> tuple[ast.Module, list[_Unit]] | None:
    """One unit per function, method and class, plus one for module-level code.

    Returns None for non-Python files and files that don't parse.
    """
    if language_of(path) != "python":
        return None
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    lines = text.splitlines()
    units: list[_Unit] = []

    def add(kind, start, end, classes, node, name, qualified, class_name):
        content = "\n".join(lines[start - 1:end])
        chunk = Chunk(path, "python", kind, start, end, content, class_name, name, qualified)
        units.append(_Unit(chunk, list(range(start, end + 1)), node.body, [node], classes))

    def visit(body, classes):
        for node in body:
            owner = ".".join(c.name for c in classes) or None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{owner}.{node.name}" if owner else node.name
                kind = "method" if classes else "function"
                add(kind, _start(node), node.end_lineno, classes, node, node.name, qualified, owner)
            elif isinstance(node, ast.ClassDef):
                start, end = _start(node), node.end_lineno
                if not full_classes and len("\n".join(lines[start - 1:end])) > CLASS_BODY_CHARS:
                    first = node.body[0]  # keep the header and docstring only
                    has_doc = ast.get_docstring(node) is not None
                    end = first.end_lineno if has_doc else max(node.lineno, first.lineno - 1)
                qualified = f"{owner}.{node.name}" if owner else node.name
                add("class", start, end, classes, node, node.name, qualified, qualified)
                visit(node.body, classes + [node])

    visit(tree.body, [])
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    line_nos = _uncovered(lines, {n for d in defs for n in range(_start(d), d.end_lineno + 1)})
    if line_nos:
        stmts = [n for n in tree.body if n not in defs]
        content = "\n".join(lines[n - 1] for n in line_nos)
        chunk = Chunk(path, "python", "module", line_nos[0], line_nos[-1], content,
                      qualified_name=_module_name(path))
        units.append(_Unit(chunk, line_nos, stmts, stmts, []))
    return tree, units


def chunk_ast(path: str, text: str) -> list[Chunk]:
    parsed = _py_units(path, text)
    if parsed is None:
        return chunk_lines(path, text)
    return [u.chunk for u in parsed[1]]


# --- recursive: split long units at nested blocks, blank lines, then lines ----


def _size(lines: list[str], a: int, b: int) -> int:
    return sum(len(line) + 1 for line in lines[a:b])


def _children(node: ast.AST) -> list[ast.AST]:
    out = []
    for field in ("body", "orelse", "finalbody", "handlers"):
        out += [c for c in getattr(node, field, None) or [] if isinstance(c, (ast.stmt, ast.ExceptHandler))]
    for case in getattr(node, "cases", []):
        out += case.body
    return out


def _atoms(lines, a, b, stmts, index_of) -> list[tuple[int, int]]:
    """Consecutive ranges covering lines[a:b], each within MAX_CHARS where possible."""
    if _size(lines, a, b) <= MAX_CHARS:
        return [(a, b)]
    starts = {index_of[_start(s)]: s for s in stmts if a < index_of.get(_start(s), a) < b}
    if starts:
        bounds = [a, *sorted(starts), b]
        return [r for x, y in zip(bounds, bounds[1:])
                for r in _atoms(lines, x, y, _children(starts[x]) if x in starts else [], index_of)]
    cuts = [i for i in range(a + 1, b) if not lines[i - 1].strip()]
    bounds = [a, *cuts, b]
    return [r for x, y in zip(bounds, bounds[1:])
            for r in ([(x, y)] if _size(lines, x, y) <= MAX_CHARS else [(i, i + 1) for i in range(x, y)])]


def _split(unit: _Unit) -> list[Chunk]:
    lines = unit.chunk.content.split("\n")
    index_of = {n: i for i, n in enumerate(unit.line_nos)}
    pieces: list[tuple[int, int]] = []
    for x, y in _atoms(lines, 0, len(lines), unit.stmts, index_of):
        if pieces and _size(lines, pieces[-1][0], y) <= MAX_CHARS:
            pieces[-1] = (pieces[-1][0], y)
        else:
            pieces.append((x, y))
    if len(pieces) == 1:
        return [unit.chunk]
    return [replace(unit.chunk, content="\n".join(lines[x:y]), start_line=unit.line_nos[x],
                    end_line=unit.line_nos[y - 1], part=i)
            for i, (x, y) in enumerate(pieces, 1)]


def chunk_recursive(path: str, text: str) -> list[Chunk]:
    parsed = _py_units(path, text)
    if parsed is None:
        return chunk_lines(path, text)
    return [c for u in parsed[1] for c in _split(u)]


# --- contextual: recursive chunks with imports, class and calls embedded ------


def _call_names(nodes: list[ast.AST], first: int, last: int) -> list[str]:
    names = {ast.unparse(n.func) for root in nodes for n in ast.walk(root)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
             and first <= n.lineno <= last}
    return sorted(names)


def chunk_contextual(path: str, text: str) -> list[Chunk]:
    parsed = _py_units(path, text)
    if parsed is None:
        return chunk_lines(path, text)
    tree, units = parsed
    imports = "; ".join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    out = []
    for unit in units:
        context = f"# imports: {imports}\n" if imports else ""
        if unit.classes:
            cls = unit.classes[-1]
            bases = ", ".join(ast.unparse(b) for b in cls.bases + cls.keywords)
            context += f"# class: class {cls.name}({bases})\n"
            if doc := ast.get_docstring(cls):
                context += f"# class docstring: {doc}\n"
        for chunk in _split(unit):
            calls = _call_names(unit.walk, chunk.start_line, chunk.end_line)
            chunk.context = context + (f"# calls: {', '.join(calls)}\n" if calls else "")
            out.append(chunk)
    return out


# --- multi: file, class and function/method levels from the same source -------


def chunk_multi(path: str, text: str) -> list[Chunk]:
    parsed = _py_units(path, text, full_classes=True)
    if parsed is None:
        return chunk_lines(path, text)
    whole = Chunk(path, "python", "file", 1, max(len(text.splitlines()), 1), text,
                  qualified_name=_module_name(path))
    return [whole] + [u.chunk for u in parsed[1] if u.chunk.kind != "module"]


# --- treesitter: functions, methods and classes in many languages -------------

TS_FUNCTIONS = {
    "function_definition", "function_declaration", "method_definition", "method_declaration",
    "function_item", "constructor_declaration", "method", "singleton_method",
}
TS_CLASSES = {
    "class_definition", "class_declaration", "class_specifier", "struct_specifier",
    "interface_declaration", "struct_item", "impl_item", "trait_item", "class", "module",
}


def _ts_name(node) -> str | None:
    name = node.child_by_field_name("name")
    declarator = node.child_by_field_name("declarator")  # C/C++: nested declarators
    while name is None and declarator is not None:
        if declarator.type in ("identifier", "field_identifier", "qualified_identifier"):
            name = declarator
        declarator = declarator.child_by_field_name("declarator")
    name = name or node.child_by_field_name("type")  # Rust impl blocks
    return name.text.decode() if name is not None else None


def chunk_treesitter(path: str, text: str) -> list[Chunk]:
    try:
        from tree_sitter_language_pack import Error, get_parser
    except ImportError:
        raise CodeIndexError(
            "treesitter mode needs: pip install tree-sitter tree-sitter-language-pack") from None
    language = language_of(path)
    try:
        parser = get_parser(language)
    except Error:  # unsupported language
        return chunk_lines(path, text)
    lines = text.splitlines()
    out: list[Chunk] = []
    covered: set[int] = set()

    def visit(node, classes):
        for child in node.children:
            target = child.child_by_field_name("definition") if child.type == "decorated_definition" else child
            name = _ts_name(target) if target is not None and target.type in TS_FUNCTIONS | TS_CLASSES else None
            if name is None:
                visit(child, classes)
                continue
            start, end = child.start_point[0] + 1, child.end_point[0] + 1
            covered.update(range(start, end + 1))
            owner = ".".join(classes) or None
            qualified = f"{owner}.{name}" if owner else name
            if target.type in TS_FUNCTIONS:
                kind = "method" if classes else "function"
                out.append(Chunk(path, language, kind, start, end, "\n".join(lines[start - 1:end]),
                                 owner, name, qualified))
                continue
            body = target.child_by_field_name("body")
            if body is not None and len("\n".join(lines[start - 1:end])) > CLASS_BODY_CHARS:
                end = min(end, max(start, body.start_point[0] + 1))
            out.append(Chunk(path, language, "class", start, end, "\n".join(lines[start - 1:end]),
                             qualified, name, qualified))
            visit(body if body is not None else target, classes + [name])

    visit(parser.parse(text.encode()).root_node, [])
    if not out:
        return chunk_lines(path, text)
    line_nos = _uncovered(lines, covered)
    if line_nos:
        out.append(Chunk(path, language, "module", line_nos[0], line_nos[-1],
                         "\n".join(lines[n - 1] for n in line_nos), qualified_name=_module_name(path)))
    return out


def chunk_summary(path: str, text: str) -> list[Chunk]:
    """The ast units; store.sync replaces their embedded text with an LLM summary."""
    return chunk_ast(path, text)


MODES = {
    "ast": chunk_ast,
    "lines": chunk_lines,
    "treesitter": chunk_treesitter,
    "recursive": chunk_recursive,
    "tokens": chunk_tokens,
    "contextual": chunk_contextual,
    "multi": chunk_multi,
    "summary": chunk_summary,
}
