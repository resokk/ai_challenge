"""Root checks, OS accounts (system commands mocked) and run_as."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import admin
import bot
import config
import os_users
import run_as
from run_as import UserContext


class FakeSystem:
    """Stands in for the system commands and the passwd database: useradd
    creates an entry and a home, userdel removes both."""

    def __init__(self, tmp_path: Path, fail_on: str | None = None):
        self.tmp_path = tmp_path
        self.fail_on = fail_on
        self.calls: list[list[str]] = []
        self.accounts: dict[str, SimpleNamespace] = {}

    def add(self, name: str, gecos: str) -> None:
        home = self.tmp_path / "homes" / name
        home.mkdir(parents=True, exist_ok=True)
        self.accounts[name] = SimpleNamespace(pw_name=name, pw_uid=os.getuid(), pw_gid=os.getgid(),
                                              pw_dir=str(home), pw_gecos=gecos)

    def __call__(self, argv, ok_codes=(0,)):
        self.calls.append(argv)
        if argv[0] == self.fail_on:
            raise os_users.OsUserError(f"{argv[0]} failed")
        if argv[0] == "useradd":
            self.add(argv[-1], argv[argv.index("--comment") + 1])
        elif argv[0] == "userdel":
            shutil.rmtree(self.accounts.pop(argv[-1]).pw_dir)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def getpwnam(self, name):
        return self.accounts[name]


@pytest.fixture
def system(tmp_path, monkeypatch):
    fake = FakeSystem(tmp_path)
    monkeypatch.setattr(os_users, "system", fake)
    monkeypatch.setattr(os_users, "pwd", SimpleNamespace(getpwnam=fake.getpwnam, struct_passwd=object))
    monkeypatch.setattr(os_users, "require_root", lambda: None)
    monkeypatch.setattr(admin, "notify", lambda telegram_id, text: notified.append((telegram_id, text)))
    notified: list = []
    fake.notified = notified
    return fake


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "admin.db")
    from storage import Storage

    def open_store():
        return Storage(config.DB_PATH)

    return open_store


def _register(db, telegram_id=100):
    store = db()
    store.register(telegram_id, "alice", "Alice")
    user = store.get_user(telegram_id)
    store.close()
    return user


# startup checks


@pytest.mark.parametrize(
    "platform, euid, which, message",
    [
        ("darwin", 0, "/usr/bin/x", "только на Linux"),
        ("linux", 1000, "/usr/bin/x", "нужны права root"),
        ("linux", 0, None, "не найдены команды: useradd"),
    ],
)
def test_bot_refuses_to_start(monkeypatch, capsys, platform, euid, which, message):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    monkeypatch.setattr(shutil, "which", lambda name: which)
    with pytest.raises(SystemExit) as exit_info:
        bot.main()
    assert exit_info.value.code == 1
    assert message in capsys.readouterr().err


def test_environment_ok(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    assert os_users.environment_problems() == []


def test_admin_checks_root_except_for_list(monkeypatch, db):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert admin.main(["list"]) == 0
    with pytest.raises(SystemExit) as exit_info:
        admin.main(["approve", "1"])
    assert exit_info.value.code == 1


# approve / block / delete


def test_approve_creates_os_user_then_approves(system, db):
    user = _register(db)
    assert admin.main(["approve", "100"]) == 0
    name = f"user-{user['id']}"
    assert system.calls[0] == ["useradd", "--user-group", "--create-home", "--shell", "/usr/sbin/nologin",
                               "--comment", f"devbot account {user['id']}", name]
    home = Path(system.accounts[name].pw_dir)
    assert (home / ".workspace").is_dir() and (home / ".ssh").is_dir()
    assert oct(home.stat().st_mode & 0o777) == "0o700"
    assert db().get_user(100)["status"] == "approved"
    assert system.notified and system.notified[0][0] == 100


def test_useradd_failure_keeps_pending(system, db):
    _register(db)
    system.fail_on = "useradd"
    assert admin.main(["approve", "100"]) != 0
    assert db().get_user(100)["status"] == "pending"
    assert system.notified == []


def test_foreign_os_user_is_refused(system, db):
    user = _register(db)
    system.add(f"user-{user['id']}", "someone else")
    assert admin.main(["approve", "100"]) != 0
    assert not any(call[0] in ("useradd", "usermod") for call in system.calls)
    assert db().get_user(100)["status"] == "pending"


def test_reapprove_unlocks_existing_account(system, db):
    user = _register(db)
    system.add(f"user-{user['id']}", f"devbot account {user['id']}")
    assert admin.main(["approve", "100"]) == 0
    assert system.calls == [["usermod", "-U", f"user-{user['id']}"]]


def test_block_locks_and_kills(system, db):
    user = _register(db)
    admin.main(["approve", "100"])
    system.calls.clear()
    assert admin.main(["block", "100"]) == 0
    name = f"user-{user['id']}"
    assert system.calls == [["usermod", "-L", name], ["pkill", "-KILL", "-u", name]]
    assert db().get_user(100)["status"] == "blocked"
    assert system.notified[-1] == (100, admin.NOTIFICATIONS["blocked"])


def test_delete_removes_account_home_and_rows(system, db):
    user = _register(db)
    admin.main(["approve", "100"])
    store = db()
    store.upsert_task(100, "a", "https://github.com/o/r")
    store.add_message(100, "user", "hi")
    store.close()
    home = Path(system.accounts[f"user-{user['id']}"].pw_dir)
    system.calls.clear()
    assert admin.main(["delete", "100"]) == 0
    name = f"user-{user['id']}"
    assert system.calls == [["pkill", "-KILL", "-u", name], ["userdel", "--remove", name]]
    assert not home.exists()
    store = db()
    assert store.get_user(100) is None and store.list_tasks(100) == [] and store.history(100) == []


@pytest.mark.parametrize("command", ["approve", "block", "delete"])
def test_unknown_telegram_id_fails(system, db, command, capsys):
    assert admin.main([command, "999"]) != 0
    assert "999" in capsys.readouterr().err


def test_notify_failure_still_changes_status(system, db, monkeypatch, capsys):
    _register(db)
    monkeypatch.setattr(admin, "notify", _real_notify)

    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", offline)
    assert admin.main(["approve", "100"]) == 0
    assert db().get_user(100)["status"] == "approved"
    assert "Предупреждение" in capsys.readouterr().err


# run_as

# Captured before the autouse fixture swaps it for the current-user version.
_real_run_as = run_as.run_as
_real_notify = admin.notify


async def test_run_as_spawns_as_the_user(monkeypatch, tmp_path):
    captured = {}

    async def fake_exec(*argv, **kwargs):
        captured.update(kwargs, argv=argv)
        raise OSError(2, "stop here")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    ctx = UserContext(7, "user-3", 1003, 1004, tmp_path, None, None)
    await _real_run_as(ctx, ["id"], tmp_path, 5, env={"EXTRA": "1"})
    assert captured["user"] == 1003 and captured["group"] == 1004
    assert captured["extra_groups"] == [] and captured["umask"] == 0o077
    assert captured["start_new_session"] is True and captured["stdin"] == asyncio.subprocess.DEVNULL
    assert captured["preexec_fn"] is run_as._limits
    assert captured["env"] == {"HOME": str(tmp_path), "USER": "user-3", "LOGNAME": "user-3",
                               "PATH": run_as.DEFAULT_PATH, "LANG": "C.UTF-8", "EXTRA": "1"}


