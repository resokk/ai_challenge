"""Working copies and /git against local bare repositories."""

from __future__ import annotations

import pytest

import bot
import git_ops
import llm
from conftest import FakeQuery, git


@pytest.fixture
def user(tg, approved, remotes, tmp_path):
    """User 1 on task `a` with a fresh clone of acme/demo."""
    approved(1)
    remotes("acme/demo", {"README.md": "demo\n", ".gitignore": "*.log\n", "app.py": "print('hi')\n"})

    async def setup():
        message = await tg(bot.tasks, "/tasks a acme/demo")
        assert "Рабочая копия готова" in message.replies[-1], message.replies
        return tmp_path / "home-1" / ".workspace"

    return setup


@pytest.fixture
def no_llm(monkeypatch):
    async def fail(*args, **kwargs):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm, "text_reply", fail)


async def test_switch_recreates_workspace(user, tg, remotes):
    workspace = await user()
    assert (workspace / "app.py").exists()
    (workspace / "scratch.txt").write_text("unsaved")
    remotes("acme/other", {"other.py": "x = 1\n"})
    message = await tg(bot.tasks, "/tasks b acme/other")
    assert "незакоммиченные изменения" in message.replies[-1]
    assert sorted(p.name for p in workspace.iterdir()) == [".git", "other.py"]
    message = await tg(bot.tasks, "/tasks a")
    assert "незакоммиченные" not in message.replies[-1]
    assert (workspace / "app.py").exists() and not (workspace / "scratch.txt").exists()


async def test_clone_failure_still_switches(tg, approved, store, remotes):
    approved(1)
    message = await tg(bot.tasks, "/tasks a acme/missing")
    assert "Клонировать не удалось" in message.replies[-1] and "/git ssh" in message.replies[-1]
    assert store.get_user(1)["current_task_id"] == "a"


async def test_push_url_is_ssh(tg, approved, remotes, monkeypatch, tmp_path):
    approved(1)
    bare = remotes("acme/demo")
    monkeypatch.setattr(git_ops, "clone_url", lambda url, ssh: f"ssh:{url}" if ssh else str(bare))
    await tg(bot.tasks, "/tasks a acme/demo")
    workspace = tmp_path / "home-1" / ".workspace"
    assert git(workspace, "remote", "get-url", "origin").strip() == str(bare)
    assert git(workspace, "remote", "get-url", "--push", "origin").strip() == "ssh:https://github.com/acme/demo"


def test_real_clone_url():
    assert git_ops.clone_url("https://github.com/acme/demo", ssh=True) == "git@github.com:acme/demo.git"
    assert git_ops.clone_url("https://github.com/acme/demo", ssh=False) == "https://github.com/acme/demo.git"


async def test_status_clean_and_changed(user, tg):
    workspace = await user()
    message = await tg(bot.git_command, "/git")
    assert "Ветка: main → origin/main" in message.replies[-1] and "чистая" in message.replies[-1]
    (workspace / "app.py").write_text("print('bye')\n")
    (workspace / "new.py").write_text("")
    (workspace / "README.md").unlink()
    text = (await tg(bot.git_command, "/git")).replies[-1]
    assert "изменён: app.py" in text and "новый: new.py" in text and "удалён: README.md" in text


async def test_git_without_task(tg, approved):
    approved(1)
    assert "/tasks" in (await tg(bot.git_command, "/git")).replies[-1]
    assert "/git commit" in (await tg(bot.git_command, "/git bogus")).replies[-1]


async def test_commit_respects_gitignore_and_empty(user, tg, no_llm):
    workspace = await user()
    assert "Нечего коммитить" in (await tg(bot.git_command, "/git commit msg")).replies[-1]
    (workspace / "debug.log").write_text("noise")
    (workspace / "feature.py").write_text("x = 1\n")
    text = (await tg(bot.git_command, "/git commit Add feature")).replies[-1]
    assert "Add feature" in text and "feature.py" in text and "debug.log" not in text
    files = git(workspace, "show", "--name-only", "--format=", "HEAD").split()
    assert files == ["feature.py"]
    author = git(workspace, "log", "-1", "--format=%an <%ae>").strip()
    assert author == "User1 <1@users.noreply.telegram>"


async def test_commit_without_message_asks_llm(user, tg, monkeypatch):
    workspace = await user()
    seen = []

    async def fake(messages, temperature=None):
        seen.append(messages[-1]["content"])
        return "```\nAdd greeting module\n```"

    monkeypatch.setattr(llm, "text_reply", fake)
    (workspace / "greet.py").write_text("def greet(): ...\n")
    await tg(bot.git_command, "/git commit")
    assert "greet.py" in seen[0]
    assert git(workspace, "log", "-1", "--format=%s").strip() == "Add greeting module"


async def test_push_needs_key_then_reaches_remote(user, tg, tmp_path):
    workspace = await user()
    (workspace / "f.py").write_text("y = 2\n")
    await tg(bot.git_command, "/git commit Add f")
    assert "/git ssh" in (await tg(bot.git_command, "/git push")).replies[-1]
    await tg(bot.git_command, "/git ssh")
    text = (await tg(bot.git_command, "/git push")).replies[-1]
    assert "отправлены" in text
    bare = tmp_path / "remotes" / "acme__demo.git"
    assert git(bare, "log", "-1", "--format=%s").strip() == "Add f"


async def test_push_rejection_is_explained(user, tg, tmp_path):
    workspace = await user()
    await tg(bot.git_command, "/git ssh")
    other = tmp_path / "other-clone"
    git(tmp_path, "clone", "-q", str(tmp_path / "remotes" / "acme__demo.git"), str(other))
    (other / "x.txt").write_text("x")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "remote change")
    git(other, "push", "-q", "origin", "HEAD")
    (workspace / "y.txt").write_text("y")
    await tg(bot.git_command, "/git commit local change")
    text = (await tg(bot.git_command, "/git push")).replies[-1]
    assert "push не удался" in text and "принудительный push не поддерживается" in text


async def test_diff_without_changes_skips_llm(user, tg, no_llm):
    await user()
    assert "Изменений нет" in (await tg(bot.git_command, "/git diff")).replies[-1]


async def test_diff_reviews_changes(user, tg, monkeypatch, store):
    workspace = await user()
    seen = []

    async def fake(messages, temperature=None):
        seen.append(messages)
        return "app.py: текст изменён"

    monkeypatch.setattr(llm, "text_reply", fake)
    (workspace / "app.py").write_text("print('changed')\n")
    (workspace / "untracked.py").write_text("")
    text = (await tg(bot.git_command, "/git diff")).replies[-1]
    assert "app.py: текст изменён" in text
    diff = seen[0][-1]["content"]
    assert "print('changed')" in diff and "untracked.py" in diff
    assert store.history(1) == []


async def test_reset_requires_confirmation(user, tg, store):
    workspace = await user()
    (workspace / "app.py").write_text("broken\n")
    (workspace / "junk.py").write_text("")
    (workspace / "keep.log").write_text("ignored")
    message = await tg(bot.git_command, "/git reset")
    assert "app.py" in message.replies[-1] and "junk.py" in message.replies[-1]
    assert (workspace / "app.py").read_text() == "broken\n"  # nothing yet

    # Someone else's press, and a press after switching task, are refused.
    stale = FakeQuery("gitreset:1:a")
    await tg(bot.git_reset_button, user=2, query=stale)
    store.set_current_task(1, None)
    await tg(bot.git_reset_button, query=stale)
    assert "не ваша" in stale.answers[-1]
    store.set_current_task(1, "a")
    assert (workspace / "junk.py").exists()

    cancel = FakeQuery("gitcancel:1:a")
    await tg(bot.git_reset_button, query=cancel)
    assert (workspace / "junk.py").exists()

    confirm = FakeQuery("gitreset:1:a")
    await tg(bot.git_reset_button, query=confirm)
    assert "откачены" in confirm.edited[-1]
    assert (workspace / "app.py").read_text() == "print('hi')\n"
    assert not (workspace / "junk.py").exists()
    assert (workspace / "keep.log").exists()


async def test_ssh_key_created_once(tg, approved, tmp_path):
    approved(1)
    message = await tg(bot.git_command, "/git ssh")
    key = tmp_path / "home-1" / ".ssh" / "id_ed25519"
    assert oct(key.stat().st_mode & 0o777) == "0o600"
    public = next(r for r in message.replies if "ssh-ed25519" in r)
    assert "devbot-1" in public and "PRIVATE" not in "".join(message.replies)
    again = await tg(bot.git_command, "/git ssh")
    assert next(r for r in again.replies if "ssh-ed25519" in r) == public
    renewed = await tg(bot.git_command, "/git ssh new")
    assert "старый ключ" in renewed.replies[0]
    assert next(r for r in renewed.replies if "ssh-ed25519" in r) != public
