from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR))

import bot  # noqa: E402
import config  # noqa: E402
import git_ops  # noqa: E402
import os_users  # noqa: E402
import run_as  # noqa: E402
from run_as import UserContext  # noqa: E402
from storage import Storage  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "root: integration test, needs root on Linux (run in Docker)")


def pytest_collection_modifyitems(config, items):
    if sys.platform == "linux" and os.geteuid() == 0:
        return
    skip = pytest.mark.skip(reason="needs root on Linux")
    for item in items:
        if "root" in item.keywords:
            item.add_marker(skip)


async def run_as_current_user(ctx, argv, cwd, timeout, *, env=None, input=None):
    """`run_as` with the same signature, running as whoever runs the tests."""
    return await run_as.execute(argv, cwd, timeout, run_as.environment(ctx, env), {}, input)


@pytest.fixture(autouse=True)
def as_current_user(request, monkeypatch):
    if request.node.get_closest_marker("root"):
        return
    monkeypatch.setattr(run_as, "run_as", run_as_current_user)
    monkeypatch.setattr(config, "SYSTEM_PYTHON", sys.executable)


@pytest.fixture
def store(tmp_path):
    storage = Storage(tmp_path / "bot.db")
    yield storage
    storage.close()


@pytest.fixture
def make_ctx(tmp_path):
    """A context whose home is a fresh directory under tmp_path."""

    def make(telegram_id: int = 1, task_id: str | None = "t", repo_url: str | None = None) -> UserContext:
        home = tmp_path / f"home-{telegram_id}"
        (home / ".workspace").mkdir(parents=True, exist_ok=True)
        (home / ".ssh").mkdir(exist_ok=True)
        return UserContext(telegram_id, getpass.getuser(), os.getuid(), os.getgid(), home, task_id,
                           repo_url or "https://github.com/acme/demo")

    return make


@pytest.fixture
def registry():
    return bot.build_registry()


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def remotes(tmp_path, monkeypatch):
    """Local bare repositories standing in for GitHub: `remotes("acme/demo",
    {"file": "text"})` creates one, and clone_url resolves to it."""
    root = tmp_path / "remotes"

    def bare(name: str) -> Path:
        return root / (name.replace("/", "__") + ".git")

    def create(name: str, files: dict[str, str] | None = None) -> Path:
        seed = tmp_path / "seed" / name.replace("/", "__")
        seed.mkdir(parents=True)
        git(seed, "init", "-q")
        for path, text in (files or {"README.md": "demo\n"}).items():
            (seed / path).parent.mkdir(parents=True, exist_ok=True)
            (seed / path).write_text(text)
        git(seed, "add", "-A")
        git(seed, "commit", "-q", "-m", "init")
        git(tmp_path, "clone", "-q", "--bare", str(seed), str(bare(name)))
        return bare(name)

    monkeypatch.setattr(git_ops, "clone_url", lambda repo_url, ssh: str(bare(repo_url.removeprefix("https://github.com/"))))
    return create


# Telegram stand-ins: just enough of Update, Message and Context for the handlers.


class Sent:
    def __init__(self, text):
        self.text = text
        self.deleted = False

    async def edit_text(self, text, **kwargs):
        self.text = text

    async def delete(self):
        self.deleted = True


class FakeMessage:
    def __init__(self, text: str = "", document=None):
        self.text = text
        self.document = document
        self.replies: list[str] = []
        self.sent: list[Sent] = []
        self.documents: list[tuple[str, bytes]] = []
        self.chat = FakeChat()

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)
        sent = Sent(text)
        self.sent.append(sent)
        return sent

    async def reply_document(self, document, **kwargs):
        self.documents.append((document.filename, document.input_file_content))


class FakeChat:
    async def send_action(self, action):
        pass


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.answers: list[str | None] = []
        self.edited: list[str] = []

    async def answer(self, text=None, **kwargs):
        self.answers.append(text)

    async def edit_message_text(self, text, **kwargs):
        self.edited.append(text)


@pytest.fixture
def tg(store, registry, tmp_path, monkeypatch):
    """Call a handler as a Telegram user would: `await tg(handler, "/cmd a b", user=5)`.
    Approved users get an OS context under tmp_path."""

    def user_context(user, task=None):
        home = tmp_path / f"home-{user['telegram_id']}"
        for sub in (".workspace", ".ssh"):
            (home / sub).mkdir(parents=True, exist_ok=True)
        return UserContext(user["telegram_id"], getpass.getuser(), os.getuid(), os.getgid(), home,
                           task["task_id"] if task else None, task["repo_url"] if task else None)

    monkeypatch.setattr(os_users, "user_context", user_context)
    bot_data = {"store": store, "registry": registry}

    async def call(handler, text: str = "", user: int = 1, document=None, query=None):
        message = FakeMessage(text, document)
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=user, username=f"u{user}", first_name=f"User{user}"),
            message=message,
            effective_message=message,
            effective_chat=FakeChat(),
            callback_query=query,
        )
        context = SimpleNamespace(args=text.split()[1:], bot_data=bot_data)
        await handler(update, context)
        return message

    return call


@pytest.fixture
def approved(store):
    def approve(telegram_id: int = 1):
        store.register(telegram_id, f"u{telegram_id}", f"User{telegram_id}")
        store.set_status(telegram_id, "approved")

    return approve
