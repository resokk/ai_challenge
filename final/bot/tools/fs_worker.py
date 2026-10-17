"""File operations on a user's working copy, run as that user's OS account.

Standalone on purpose: only the standard library, nothing from the bot, no
secrets. Reads one JSON request {"op", "root", "args"} from stdin and writes
one JSON reply {"ok": true, "result": ...} or {"ok": false, "error": ...}.

OS permissions are the real boundary; `resolve_path` is the second layer that
gives readable errors and keeps `.git/` out of reach."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_FILE_BYTES = 10_000_000
SEARCH_FILE_MAX_BYTES = 2_000_000
SEARCH_LINE_MAX_CHARS = 300


class WorkerError(Exception):
    pass


OUTSIDE = "путь вне рабочей копии"


def resolve_path(root: Path, path: str) -> Path:
    """The absolute path `path` names inside `root`, with symlinks followed.
    Absolute paths, escapes through `..` or symlinks, and `.git/` are refused
    without revealing where anything lives on the server."""
    if not isinstance(path, str) or not path or "\0" in path:
        raise WorkerError(OUTSIDE)
    if Path(path).is_absolute():
        raise WorkerError(OUTSIDE)
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise WorkerError(OUTSIDE)
    relative = target.relative_to(root)
    if relative.parts and relative.parts[0] == ".git":
        raise WorkerError(f"{OUTSIDE}: каталог .git закрыт")
    return target


def _relative(root: Path, target: Path) -> str:
    return target.relative_to(root).as_posix() or "."


def glob_regex(pattern: str) -> re.Pattern:
    """`**` crosses directories, `*` and `?` do not. A pattern without a slash
    matches file names at any depth, so `*.py` finds every Python file."""
    if "/" not in pattern:
        pattern = "**/" + pattern
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def project_files(root: Path, base: Path, pattern: str) -> list[str]:
    """Files under `base`, relative to `root`, honouring `.gitignore`."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, stdin=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        names = [name for name in result.stdout.decode(errors="replace").split("\0") if name]
    else:
        names = []
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != ".git")
            names += [_relative(root, Path(directory) / f) for f in filenames]
    prefix = _relative(root, base)
    regex = glob_regex(pattern)
    files = []
    for name in sorted(set(names)):
        if prefix != "." and not name.startswith(prefix + "/"):
            continue
        if name.split("/")[0] == ".git" or not regex.match(name):
            continue
        if (root / name).is_file():
            files.append(name)
    return files


def read_text(target: Path) -> tuple[str, str]:
    """A text file's content with `\\n` line endings, and its encoding."""
    if not target.is_file():
        raise WorkerError("файл не найден")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise WorkerError("файл слишком большой")
    data = target.read_bytes()
    if b"\0" in data[:8192]:
        raise WorkerError("бинарный файл, читать можно только текстовые")
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        return data.decode(encoding), encoding
    except UnicodeDecodeError:
        raise WorkerError("файл не в UTF-8 (похоже на бинарный)") from None


def atomic_write(target: Path, text: str, encoding: str) -> None:
    mode = target.stat().st_mode & 0o777 if target.exists() else None
    fd, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
        if mode is not None:
            os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _numbered(lines: list[str], first: int) -> str:
    return "\n".join(f"{first + i:>5}\t{line}" for i, line in enumerate(lines))


# operations


def op_resolve(root: Path, path: str, kind: str = "file") -> str:
    target = resolve_path(root, path)
    if kind == "file" and not target.is_file():
        raise WorkerError(f"файл не найден: {path}")
    return _relative(root, target)


def op_list_files(root: Path, path: str = ".", glob: str = "**/*", max_results: int = 200) -> str:
    base = resolve_path(root, path)
    if not base.is_dir():
        raise WorkerError(f"каталог не найден: {path}")
    files = project_files(root, base, glob)
    shown = files[:max_results]
    lines = [f"{name}  ({(root / name).stat().st_size} B)" for name in shown]
    if not files:
        return "Файлов не найдено."
    if len(files) > len(shown):
        lines.append(f"[показано {len(shown)} из {len(files)}]")
    return "\n".join(lines)


def op_read_file(root: Path, path: str, start_line: int = 1, end_line: int | None = None,
                 max_chars: int = 20000) -> str:
    text, _ = read_text(resolve_path(root, path))
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total = len(lines)
    start = max(start_line, 1)
    end = min(end_line or total, total)
    if start > total:
        return f"В файле {total} строк, запрошена строка {start}."
    out, size, last = [], 0, start - 1
    for number in range(start, end + 1):
        line = f"{number:>5}\t{lines[number - 1]}"
        if out and size + len(line) + 1 > max_chars:
            break
        out.append(line[:max_chars])
        size += len(line) + 1
        last = number
    result = "\n".join(out)
    if last < end:
        result += f"\n[обрезано: показаны строки {start}-{last} из {total}; дальше запросите start_line={last + 1}]"
    return result


def op_search_files(root: Path, query: str, regex: bool = False, glob: str = "**/*",
                    case_sensitive: bool = False, max_results: int = 100) -> str:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query if regex else re.escape(query), flags)
    except re.error as error:
        raise WorkerError(f"невалидное регулярное выражение: {error}") from None
    groups: list[str] = []
    found = shown = 0
    for name in project_files(root, root, glob):
        target = root / name
        if target.stat().st_size > SEARCH_FILE_MAX_BYTES:
            continue
        try:
            text, _ = read_text(target)
        except WorkerError:
            continue
        hits = []
        for number, line in enumerate(text.split("\n"), 1):
            if pattern.search(line):
                found += 1
                if shown < max_results:
                    shown += 1
                    hits.append(f"{name}:{number}: {line.strip()[:SEARCH_LINE_MAX_CHARS]}")
        if hits:
            groups.append("\n".join(hits))
    if not found:
        return "Совпадений нет."
    summary = f"Совпадений: {found}" + (f" (показано {shown})" if shown < found else "")
    return "\n\n".join(groups) + "\n\n" + summary


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _doc(node) -> str:
    doc = ast.get_docstring(node)
    return f"  — {doc.strip().splitlines()[0]}" if doc and doc.strip() else ""


def _outline(body: list[ast.stmt], depth: int) -> list[str]:
    lines = []
    indent = "  " * depth
    for node in body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            lines.append(f"{indent}{node.lineno}: class {node.name}" + (f"({bases})" if bases else "") + _doc(node))
            lines += _outline(node.body, depth + 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"{indent}{node.lineno}: {_signature(node)}{_doc(node)}")
    return lines


def op_python_outline(root: Path, path: str) -> str:
    text, _ = read_text(resolve_path(root, path))
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as error:
        return f"Синтаксическая ошибка в строке {error.lineno}: {error.msg}"
    imports = [f"  {node.lineno}: {ast.unparse(node)}" for node in tree.body
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    parts = []
    doc = ast.get_docstring(tree)
    if doc:
        parts.append("Модуль: " + doc.strip().splitlines()[0])
    if imports:
        parts.append("Импорты:\n" + "\n".join(imports))
    definitions = _outline(tree.body, 1)
    parts.append("Определения:\n" + "\n".join(definitions) if definitions else "Классов и функций нет.")
    return "\n\n".join(parts)


def op_write_file(root: Path, path: str, content: str, max_bytes: int = 1_000_000) -> str:
    target = resolve_path(root, path)
    if len(content.encode()) > max_bytes:
        raise WorkerError(f"содержимое больше лимита {max_bytes} байт")
    if target.is_dir():
        raise WorkerError(f"{path} — каталог")
    existed = target.exists()
    encoding, newline = "utf-8", "\n"
    if existed:
        try:
            old, encoding = read_text(target)
            raw = target.read_bytes()
            if b"\r\n" in raw:
                newline = "\r\n"
        except WorkerError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    text = content.replace("\r\n", "\n")
    atomic_write(target, text.replace("\n", newline), encoding)
    lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    return f"Файл {'перезаписан' if existed else 'создан'}: {_relative(root, target)} ({lines} строк)"


def op_edit_file(root: Path, path: str, old_str: str, new_str: str) -> str:
    target = resolve_path(root, path)
    text, encoding = read_text(target)
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    old = old_str.replace("\r\n", "\n")
    new = new_str.replace("\r\n", "\n")
    if not old:
        raise WorkerError("old_str пустой")
    count = text.count(old)
    if count == 0:
        raise WorkerError("old_str не найден в файле")
    if count > 1:
        raise WorkerError(f"old_str найден {count} раз — добавьте окружающий контекст, чтобы он был уникален")
    position = text.index(old)
    updated = text.replace(old, new, 1)
    atomic_write(target, updated.replace("\n", "\r\n") if crlf else updated, encoding)
    first = text.count("\n", 0, position) + 1
    lines = updated.split("\n")
    start = max(first - 2, 1)
    end = min(first + new.count("\n") + 2, len(lines))
    return f"Изменён {_relative(root, target)}:\n" + _numbered(lines[start - 1:end], start)


def op_reset_workspace(root: Path) -> str:
    """Leave the working copy empty, ready for a fresh clone."""
    if root.is_symlink() or root.is_file():
        root.unlink()
    elif root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)
    return "ok"


OPERATIONS = {
    "resolve": op_resolve,
    "list_files": op_list_files,
    "read_file": op_read_file,
    "search_files": op_search_files,
    "python_outline": op_python_outline,
    "write_file": op_write_file,
    "edit_file": op_edit_file,
    "reset_workspace": op_reset_workspace,
}


def handle(request: dict) -> dict:
    try:
        operation = OPERATIONS[request["op"]]
        root = Path(request["root"])
        if request["op"] != "reset_workspace":
            if not root.is_dir():
                raise WorkerError("рабочей копии нет — выберите задачу: /tasks <id>")
            root = root.resolve()
        return {"ok": True, "result": operation(root, **request.get("args", {}))}
    except WorkerError as error:
        return {"ok": False, "error": str(error)}
    except (OSError, TypeError, KeyError, ValueError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {getattr(error, 'strerror', None) or error}"}


def main() -> None:
    request = json.loads(sys.stdin.buffer.read() or b"{}")
    sys.stdout.write(json.dumps(handle(request), ensure_ascii=False))


if __name__ == "__main__":
    main()
