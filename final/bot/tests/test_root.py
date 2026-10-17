"""Real OS accounts. Root on Linux only - run in Docker, see README."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import pwd
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import admin
import bot
import config
import git_ops
import os_users
import run_as
from storage import Storage

pytestmark = pytest.mark.root


def user_row(telegram_id: int):
    store = Storage(config.DB_PATH)
    try:
        return store.get_user(telegram_id)
    finally:
        store.close()


def ctx_for(telegram_id: int):
    return dataclasses.replace(os_users.user_context(user_row(telegram_id)), task_id="t",
                               repo_url="https://github.com/acme/demo")


@pytest.fixture
def two_users(monkeypatch):
    monkeypatch.setattr(config, "SYSTEM_PYTHON", sys.executable)
    monkeypatch.setattr(admin, "notify", lambda telegram_id, text: None)
    if not config.ENV_PATH.exists():
        config.ENV_PATH.write_text("TELEGRAM_BOT_TOKEN=test\n")
    ids = random.sample(range(10**8, 10**9), 2)
    store = Storage(config.DB_PATH)
    for telegram_id in ids:
        store.register(telegram_id, "u", "U")
    store.close()
    os_users.secure_bot_files()
    for telegram_id in ids:
        assert admin.main(["approve", str(telegram_id)]) == 0
    yield ids
    for telegram_id in ids:
        admin.main(["delete", str(telegram_id)])


@pytest.fixture
def bare_repo(monkeypatch):
    """A repository user A can clone without a network."""
    root = Path(tempfile.mkdtemp(dir="/tmp"))
    os.chmod(root, 0o755)
    seed = root / "seed"
    seed.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], cwd=seed, check=True)
    (seed / "README.md").write_text("demo\n")
    subprocess.run([*git, "add", "-A"], cwd=seed, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=seed, check=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(root / "demo.git")], check=True)
    monkeypatch.setattr(git_ops, "clone_url", lambda repo_url, ssh: str(root / "demo.git"))
    yield root
    subprocess.run(["rm", "-rf", str(root)])


async def test_isolation_and_ownership(two_users, bare_repo):
    a, b = two_users
    ctx_a, ctx_b = ctx_for(a), ctx_for(b)
    uid_a = pwd.getpwnam(ctx_a.os_user).pw_uid
    home_a = Path(ctx_a.home)
    assert home_a.stat().st_uid == uid_a and home_a.stat().st_mode & 0o777 == 0o700

    # git clone runs as A: the clone is A's. The bare repo must be readable by A.
    subprocess.run(["chown", "-R", ctx_a.os_user, str(bare_repo)], check=True)
    assert (await git_ops.switch_workspace(ctx_a)).error is None
    assert (ctx_a.workspace / "README.md").stat().st_uid == uid_a

    registry = bot.build_registry()
    await git_ops.public_key(ctx_b)
    reply = await registry.call_tool(ctx_b, "write_file", {"path": "secret.py", "content": "token = 1\n"})
    assert "создан" in reply.text, reply.text

    probe = f"""
import json, os
targets = {{"env": {str(config.ENV_PATH)!r}, "db": {str(config.DB_PATH)!r},
           "b_ws": {str(ctx_b.workspace / 'secret.py')!r}, "b_ssh": {str(ctx_b.ssh_key)!r}}}
result = {{"uid": os.getuid()}}
for name, path in targets.items():
    try:
        open(path).read()
        result[name] = "read"
    except PermissionError:
        result[name] = "denied"
for name, path in {{"bot_dir": {str(config.BOT_DIR)!r}, "b_home": {str(ctx_b.home)!r}}}.items():
    try:
        os.listdir(path)
        result[name] = "listed"
    except PermissionError:
        result[name] = "denied"
print(json.dumps(result))
"""
    reply = await registry.call_tool(ctx_a, "write_file", {"path": "probe.py", "content": probe})
    assert "создан" in reply.text, reply.text
    assert (ctx_a.workspace / "probe.py").stat().st_uid == uid_a

    reply = await registry.call_tool(ctx_a, "run_python", {"target": "probe.py"})
    assert reply.run is not None and reply.run.exit_code == 0, reply.text
    result = json.loads(reply.run.stdout)
    assert result == {"uid": uid_a, "env": "denied", "db": "denied", "b_ws": "denied", "b_ssh": "denied",
                      "bot_dir": "denied", "b_home": "denied"}

    # pip runs as A too: everything in the venv is A's.
    reply = await registry.call_tool(ctx_a, "pip_install", {"packages": ["pip"]})
    assert reply.run is not None and reply.run.exit_code == 0, reply.text
    for directory, _, names in os.walk(ctx_a.venv):
        for name in names:
            assert os.lstat(os.path.join(directory, name)).st_uid == uid_a


async def test_block_kills_processes(two_users):
    a, _ = two_users
    ctx_a = ctx_for(a)
    sleeper = asyncio.create_task(run_as.run_as(ctx_a, ["sleep", "300"], ctx_a.home, 600))
    for _ in range(50):
        if subprocess.run(["pgrep", "-u", ctx_a.os_user], capture_output=True).returncode == 0:
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("sleep did not start")
    assert admin.main(["block", str(a)]) == 0
    result = await asyncio.wait_for(sleeper, 10)
    assert result.exit_code != 0
    assert subprocess.run(["pgrep", "-u", ctx_a.os_user], capture_output=True).returncode == 1
    assert user_row(a)["status"] == "blocked"
