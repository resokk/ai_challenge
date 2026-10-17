"""File tools through fs_worker, run by the current-user run_as."""

from __future__ import annotations

import os

import pytest

import config
from conftest import git


@pytest.fixture
def ws(make_ctx):
    ctx = make_ctx()
    root = ctx.workspace
    git(root, "init", "-q")
    (root / ".gitignore").write_text("secret.txt\nbuild/\n")
    (root / "main.py").write_text('import os\n\n\ndef hello(name: str) -> str:\n    """Greet."""\n    return name\n')
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_text("class Helper:\n    def run(self, x=1):\n        return hello(x)\n")
    (root / "secret.txt").write_text("hello secret\n")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("hello = 1\n")
    return ctx


async def call(registry, ctx, name, **arguments):
    return (await registry.call_tool(ctx, name, arguments)).text


@pytest.mark.parametrize("path", ["../x", "/etc/passwd", "pkg/../../x", ".git/config", ".git"])
async def test_sandbox_rejects(registry, ws, path):
    text = await call(registry, ws, "read_file", path=path)
    assert "путь вне рабочей копии" in text
    assert str(ws.home) not in text


async def test_symlink_outside_is_rejected(registry, ws, tmp_path):
    (tmp_path / "outside.txt").write_text("nope")
    os.symlink(tmp_path / "outside.txt", ws.workspace / "link.txt")
    assert "путь вне рабочей копии" in await call(registry, ws, "read_file", path="link.txt")
    assert "путь вне рабочей копии" in await call(registry, ws, "write_file", path="link.txt", content="x")


async def test_users_do_not_see_each_other(registry, make_ctx, ws):
    other = make_ctx(telegram_id=2)
    (other.workspace / "theirs.py").write_text("x = 1\n")
    listing = await call(registry, ws, "list_files")
    assert "theirs.py" not in listing
    assert "путь вне" in await call(registry, ws, "read_file", path="../../home-2/.workspace/theirs.py")


async def test_no_task_hint(registry, make_ctx):
    ctx = make_ctx(task_id=None)
    assert "/tasks" in await call(registry, ctx, "list_files")


async def test_list_files_honours_gitignore(registry, ws):
    text = await call(registry, ws, "list_files")
    assert "main.py" in text and "pkg/util.py" in text and ".gitignore" in text
    assert "secret.txt" not in text and "build/" not in text and ".git/" not in text
    assert "pkg/util.py" in await call(registry, ws, "list_files", glob="*.py")
    limited = await call(registry, ws, "list_files", max_results=1)
    assert "показано 1 из 3" in limited


async def test_search_files_honours_gitignore(registry, ws):
    text = await call(registry, ws, "search_files", query="hello")
    assert "main.py:4: def hello(name: str) -> str:" in text
    assert "pkg/util.py:3:" in text
    assert "secret.txt" not in text and "build" not in text
    assert "Совпадений: 2" in text
    assert "невалидное регулярное" in await call(registry, ws, "search_files", query="(", regex=True)
    assert "Совпадений: 1" in await call(registry, ws, "search_files", query="HELLO\\(", regex=True,
                                         case_sensitive=False, glob="main.py")


async def test_read_file_range_and_truncation(registry, ws, monkeypatch):
    (ws.workspace / "long.txt").write_text("".join(f"line {i}\n" for i in range(1, 101)))
    text = await call(registry, ws, "read_file", path="long.txt", start_line=10, end_line=12)
    assert text.splitlines() == ["   10\tline 10", "   11\tline 11", "   12\tline 12"]
    monkeypatch.setattr(config, "FILE_READ_MAX_CHARS", 60)
    text = await call(registry, ws, "read_file", path="long.txt")
    assert "обрезано" in text and "start_line=" in text


async def test_read_binary_is_rejected(registry, ws):
    (ws.workspace / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert "бинарный" in await call(registry, ws, "read_file", path="blob.bin")


async def test_edit_file_needs_exactly_one_match(registry, ws):
    path = ws.workspace / "main.py"
    before = path.read_text()
    assert "не найден" in await call(registry, ws, "edit_file", path="main.py", old_str="missing", new_str="x")
    assert "найден 2 раз" in await call(registry, ws, "edit_file", path="main.py", old_str="name", new_str="x")
    assert path.read_text() == before
    text = await call(registry, ws, "edit_file", path="main.py", old_str="return name", new_str="return name.upper()")
    assert "return name.upper()" in text and "6\t" in text
    assert "return name.upper()" in path.read_text()


async def test_edit_keeps_crlf_and_bom(registry, ws):
    path = ws.workspace / "win.txt"
    path.write_bytes(b"\xef\xbb\xbfa\r\nb\r\n")
    await call(registry, ws, "edit_file", path="win.txt", old_str="b", new_str="c\nd")
    assert path.read_bytes() == b"\xef\xbb\xbfa\r\nc\r\nd\r\n"


async def test_write_file_creates_dirs_and_respects_limit(registry, ws, monkeypatch):
    text = await call(registry, ws, "write_file", path="new/deep/mod.py", content="a\nb\n")
    assert "создан" in text and "2 строк" in text
    assert (ws.workspace / "new/deep/mod.py").read_text() == "a\nb\n"
    assert "перезаписан" in await call(registry, ws, "write_file", path="new/deep/mod.py", content="c\n")
    monkeypatch.setattr(config, "FILE_WRITE_MAX_BYTES", 10)
    assert "лимит" in await call(registry, ws, "write_file", path="big.txt", content="x" * 11)
    assert not (ws.workspace / "big.txt").exists()


async def test_python_outline(registry, ws):
    text = await call(registry, ws, "python_outline", path="main.py")
    assert "1: import os" in text and "4: def hello(name: str) -> str  — Greet." in text
    text = await call(registry, ws, "python_outline", path="pkg/util.py")
    assert "1: class Helper" in text and "2: def run(self, x=1)" in text
    (ws.workspace / "bad.py").write_text("def broken(:\n")
    assert "Синтаксическая ошибка в строке 1" in await call(registry, ws, "python_outline", path="bad.py")
