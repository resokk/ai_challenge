"""venv, pip_install and run_python; pip itself is mocked."""

from __future__ import annotations

import os
import sys
import time

import pytest

import bot
import config
import git_ops
import llm
import run_as
from conftest import run_as_current_user
from run_as import RunResult
from tools import runner


@pytest.fixture
def pip_calls(monkeypatch):
    """Mock pip: every `-m pip install` is recorded, not run."""
    calls: list[list[str]] = []

    async def fake(ctx, argv, cwd, timeout, *, env=None, input=None):
        if argv[1:4] == ["-m", "pip", "install"]:
            calls.append(argv[4:])
            return RunResult(0, 0.1, "Successfully installed\n", "")
        return await run_as_current_user(ctx, argv, cwd, timeout, env=env, input=input)

    monkeypatch.setattr(run_as, "run_as", fake)
    return calls


@pytest.fixture
def quick_venv(monkeypatch):
    """A stand-in venv whose python is the test interpreter, without the
    seconds a real one takes to create."""

    async def ensure(ctx):
        (ctx.venv / "bin").mkdir(parents=True, exist_ok=True)
        python = ctx.venv / "bin" / "python"
        if not python.exists():
            python.symlink_to(sys.executable)

    monkeypatch.setattr(runner, "ensure_venv", ensure)


async def test_venv_lives_in_home_and_survives_task_switch(make_ctx, remotes, pip_calls):
    remotes("acme/demo")
    ctx = make_ctx()
    await git_ops.switch_workspace(ctx)
    await runner.ensure_venv(ctx)
    assert (ctx.home / ".venv" / "bin" / "python").exists()
    assert pip_calls == [["--upgrade", "pip"]]
    remotes("acme/other")
    await git_ops.switch_workspace(make_ctx(repo_url="https://github.com/acme/other"))
    assert (ctx.home / ".venv" / "bin" / "python").exists()
    assert (ctx.workspace / "README.md").exists()
    await runner.ensure_venv(ctx)
    assert len(pip_calls) == 1  # not recreated


@pytest.mark.parametrize(
    "spec",
    ["--index-url=http://evil", "-e", "-e .", "../x", "./pkg", "/abs/pkg", "git+https://github.com/a/b",
     "https://x/y.whl", "pkg @ file:///tmp/x", "a;b", ""],
)
async def test_pip_install_rejects_options_paths_urls(registry, make_ctx, pip_calls, quick_venv, spec):
    reply = await registry.call_tool(make_ctx(), "pip_install", {"packages": ["requests", spec]})
    assert reply.text.startswith("Ошибка")
    assert pip_calls == []


async def test_pip_install_accepts_specs(registry, make_ctx, pip_calls, quick_venv):
    specs = ["requests", "uvicorn[standard]", "httpx==0.27.2", "pydantic>=2,<3", "Django ~= 5.0"]
    reply = await registry.call_tool(make_ctx(), "pip_install", {"packages": specs})
    assert reply.run is not None and reply.run.exit_code == 0
    assert pip_calls == [specs]


async def test_pip_install_requirements_go_through_sandbox(registry, make_ctx, pip_calls, quick_venv):
    ctx = make_ctx()
    (ctx.workspace / "requirements.txt").write_text("requests\n")
    await registry.call_tool(ctx, "pip_install", {"requirements": "requirements.txt"})
    assert pip_calls == [["-r", "requirements.txt"]]
    reply = await registry.call_tool(ctx, "pip_install", {"requirements": "../outside.txt"})
    assert "путь вне рабочей копии" in reply.text
    assert "packages или requirements" in (await registry.call_tool(ctx, "pip_install", {})).text


async def test_run_python_returns_streams_separately(registry, make_ctx, quick_venv):
    ctx = make_ctx()
    (ctx.workspace / "main.py").write_text("import sys\nprint('out', sys.argv[1:])\nprint('err', file=sys.stderr)\nsys.exit(3)\n")
    reply = await registry.call_tool(ctx, "run_python", {"target": "main.py", "args": ["a b", "c"]})
    assert reply.run.exit_code == 3
    assert reply.run.stdout == "out ['a b', 'c']\n"
    assert reply.run.stderr == "err\n"
    assert "exit_code: 3" in reply.text and "stdout:\nout" in reply.text and "stderr:\nerr" in reply.text


async def test_run_python_module(registry, make_ctx, quick_venv):
    reply = await registry.call_tool(make_ctx(), "run_python", {"target": "json.tool", "module": True, "args": ["--help"]})
    assert reply.run.exit_code == 0 and "usage" in reply.run.stdout
    bad = await registry.call_tool(make_ctx(), "run_python", {"target": "os; rm", "module": True})
    assert "не имя модуля" in bad.text


@pytest.mark.parametrize("target", ["../x.py", "/etc/hosts", "missing.py"])
async def test_script_outside_workspace_is_rejected(registry, make_ctx, quick_venv, target):
    reply = await registry.call_tool(make_ctx(), "run_python", {"target": target})
    assert reply.run is None and reply.text.startswith("Ошибка")


async def test_infinite_loop_is_killed_with_children(registry, make_ctx, quick_venv, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUN_TIMEOUT_SEC", 1)
    ctx = make_ctx()
    pid_file = tmp_path / "child.pid"
    (ctx.workspace / "loop.py").write_text(
        "import subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'while True: pass'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "while True: pass\n"
    )
    started = time.monotonic()
    reply = await registry.call_tool(ctx, "run_python", {"target": "loop.py"})
    assert time.monotonic() - started < 10
    assert reply.run.timed_out and "таймаут" in reply.text
    child = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


async def test_bot_secrets_do_not_reach_user_code(registry, make_ctx, quick_venv, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
    ctx = make_ctx()
    (ctx.workspace / "env.py").write_text("import os\nprint(sorted(os.environ))\n")
    reply = await registry.call_tool(ctx, "run_python", {"target": "env.py"})
    assert "TELEGRAM_BOT_TOKEN" not in reply.run.stdout and "DEEPSEEK_API_KEY" not in reply.run.stdout
    assert "VIRTUAL_ENV" in reply.run.stdout and "HOME" in reply.run.stdout


async def test_long_output_keeps_head_and_tail(registry, make_ctx, quick_venv, monkeypatch):
    monkeypatch.setattr(config, "RUN_OUTPUT_MAX_CHARS", 100)
    ctx = make_ctx()
    (ctx.workspace / "big.py").write_text("print('START' + 'x' * 5000 + 'END')\n")
    reply = await registry.call_tool(ctx, "run_python", {"target": "big.py"})
    stdout_section = reply.text.split("stdout:\n")[1].split("\nstderr:")[0]
    assert stdout_section.startswith("START") and stdout_section.rstrip().endswith("END")
    assert "вырезано" in stdout_section
    assert len(reply.run.stdout) > 5000


async def test_user_run_sends_streams_and_full_file(tg, approved, store, quick_venv, monkeypatch, tmp_path):
    approved(1)
    store.upsert_task(1, "t", "https://github.com/acme/demo")
    store.set_current_task(1, "t")
    monkeypatch.setattr(config, "RUN_OUTPUT_MAX_CHARS", 100)
    workspace = tmp_path / "home-1" / ".workspace"  # the home the tg fixture gives user 1
    workspace.mkdir(parents=True)
    (workspace / "main.py").write_text("import sys\nprint('A' * 500)\nprint('oops', file=sys.stderr)\n")
    message = await tg(bot.run_command, "/run main.py")
    assert message.replies[1].startswith("python main.py\nкод выхода 0")
    stdout_reply = next(r for r in message.replies if "stdout:" in r)
    stderr_reply = next(r for r in message.replies if "stderr:" in r)
    assert "вырезано" in stdout_reply and "oops" in stderr_reply
    assert [name for name, _ in message.documents] == ["stdout.txt"]
    assert store.history(1) == []


@pytest.fixture
def run_user(tg, approved, store, quick_venv, tmp_path):
    """User 1 with an active task; returns a function running a script via /run."""
    approved(1)
    store.upsert_task(1, "t", "https://github.com/acme/demo")
    store.set_current_task(1, "t")
    workspace = tmp_path / "home-1" / ".workspace"
    workspace.mkdir(parents=True)

    async def run(script: str):
        (workspace / "main.py").write_text(script)
        return await tg(bot.run_command, "/run main.py")

    return run


async def test_long_stream_is_summarized_by_llm(run_user, store, monkeypatch):
    requests = []

    async def fake(messages, temperature=None):
        requests.append(messages)
        return "Напечатано 2000 символов A."

    monkeypatch.setattr(llm, "text_reply", fake)
    message = await run_user("import sys\nprint('A' * 2000)\nprint('oops', file=sys.stderr)\n")
    assert len(requests) == 1  # only stdout is over 1 KB
    request = requests[0][-1]["content"]
    assert "python main.py" in request and "stdout" in request and "A" * 2000 in request
    assert any(r.startswith("stdout (кратко") and "Напечатано 2000 символов A." in r for r in message.replies)
    assert not any("A" * 100 in r for r in message.replies)
    assert any("stderr:" in r and "oops" in r for r in message.replies)
    assert message.documents == [("stdout.txt", ("A" * 2000 + "\n").encode())]
    assert store.history(1) == []


async def test_short_streams_skip_llm(run_user, monkeypatch):
    async def fail(*args, **kwargs):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm, "text_reply", fail)
    message = await run_user("print('x' * 1000)\n")  # 1001 bytes with the newline
    assert any("x" * 1000 in r for r in message.replies)
    assert message.documents == []


async def test_summary_failure_falls_back_to_clipped_output(run_user, monkeypatch):
    async def down(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr(llm, "text_reply", down)
    monkeypatch.setattr(config, "RUN_OUTPUT_MAX_CHARS", 100)
    message = await run_user("print('START' + 'x' * 2000 + 'END')\n")
    stdout_reply = next(r for r in message.replies if "stdout:" in r)
    assert "START" in stdout_reply and "END" in stdout_reply and "вырезано" in stdout_reply
    assert [name for name, _ in message.documents] == ["stdout.txt"]


async def test_run_and_pip_use_the_same_tools_as_tool(tg, approved, registry, monkeypatch):
    approved(1)
    calls = []

    async def record(ctx, name, arguments):
        calls.append((name, arguments))
        from tools.registry import ToolReply
        return ToolReply("ok")

    monkeypatch.setattr(registry, "call_tool", record)
    await tg(bot.run_command, "/run main.py a 'b c'")
    await tg(bot.run_command, "/run -m pytest -q")
    await tg(bot.pip_command, "/pip requests 'httpx==0.27.2'")
    await tg(bot.pip_command, "/pip -r requirements.txt")
    await tg(bot.pip_command, "/pip")
    await tg(bot.tool_command, '/tool run_python {"target": "main.py", "args": ["a", "b c"]}')
    assert calls == [
        ("run_python", {"target": "main.py", "args": ["a", "b c"]}),
        ("run_python", {"target": "pytest", "module": True, "args": ["-q"]}),
        ("pip_install", {"packages": ["requests", "httpx==0.27.2"]}),
        ("pip_install", {"requirements": "requirements.txt"}),
        ("pip_list", {}),
        ("run_python", {"target": "main.py", "args": ["a", "b c"]}),
    ]
