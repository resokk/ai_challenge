"""Root checks, the per-user OS accounts and the permissions that keep them
apart from the bot and from each other."""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import config
import run_as
from run_as import UserContext

REQUIRED_COMMANDS = ("useradd", "usermod", "userdel", "pkill", "git", "ssh-keygen")


class OsUserError(Exception):
    pass


def environment_problems() -> list[str]:
    """Everything that stops the bot from running here; empty when it can."""
    if sys.platform != "linux":
        return [f"бот работает только на Linux (сейчас {sys.platform})"]
    problems = []
    if os.geteuid() != 0:
        problems.append("нужны права root: запустите от root")
    missing = [name for name in REQUIRED_COMMANDS if shutil.which(name) is None]
    if missing:
        problems.append("не найдены команды: " + ", ".join(missing))
    return problems


def require_root() -> None:
    problems = environment_problems()
    if problems:
        for problem in problems:
            print(f"Ошибка: {problem}", file=sys.stderr)
        sys.exit(1)


def system(argv: list[str], ok_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    """Every useradd/usermod/userdel/pkill/chown goes through here."""
    result = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode not in ok_codes:
        raise OsUserError(f"{' '.join(argv)}: {result.stderr.strip() or f'код {result.returncode}'}")
    return result


def os_username(user_id: int) -> str:
    return f"user-{user_id}"


def marker(user_id: int) -> str:
    """The comment that marks an account as this bot's. No colon: it is the
    field separator of /etc/passwd, and useradd refuses it."""
    return f"devbot account {user_id}"


def lookup(user_id: int) -> pwd.struct_passwd | None:
    try:
        return pwd.getpwnam(os_username(user_id))
    except KeyError:
        return None


def _owned(user_id: int) -> pwd.struct_passwd | None:
    """The account if it exists, refusing one this bot did not create."""
    entry = lookup(user_id)
    if entry is not None and entry.pw_gecos != marker(user_id):
        raise OsUserError(f"пользователь ОС {entry.pw_name} уже существует и не принадлежит боту")
    return entry


def user_context(user, task=None) -> UserContext:
    """The call context for a `users` row and, optionally, its active task."""
    entry = pwd.getpwnam(os_username(user["id"]))
    return UserContext(
        telegram_id=user["telegram_id"],
        os_user=entry.pw_name,
        uid=entry.pw_uid,
        gid=entry.pw_gid,
        home=Path(entry.pw_dir),
        task_id=task["task_id"] if task else None,
        repo_url=task["repo_url"] if task else None,
    )


async def ensure_os_user(user) -> bool:
    """Make the account of a `users` row usable: created, unlocked, its home
    private and its working directories in place. True when it was created."""
    user_id = user["id"]
    name = os_username(user_id)
    created = _owned(user_id) is None
    if created:
        system(["useradd", "--user-group", "--create-home", "--shell", "/usr/sbin/nologin",
                "--comment", marker(user_id), name])
    else:
        system(["usermod", "-U", name])
    ctx = user_context(user)
    os.chmod(ctx.home, 0o700)
    result = await run_as.run_as(ctx, ["mkdir", "-p", "-m", "700", ".workspace", ".ssh"], ctx.home, 30)
    if result.exit_code != 0:
        raise OsUserError(f"не удалось создать каталоги {name}: {result.stderr.strip()}")
    return created


def block_os_user(user_id: int) -> None:
    if _owned(user_id) is None:
        return
    name = os_username(user_id)
    system(["usermod", "-L", name])
    system(["pkill", "-KILL", "-u", name], ok_codes=(0, 1))


def delete_os_user(user_id: int) -> None:
    if _owned(user_id) is None:
        return
    name = os_username(user_id)
    system(["pkill", "-KILL", "-u", name], ok_codes=(0, 1))
    system(["userdel", "--remove", name])


def secure_bot_files() -> None:
    """Root owns the bot; OS accounts may pass through its directory to reach
    fs_worker, but not list it or read its secrets and database."""
    bot_dir = config.BOT_DIR
    system(["chown", "root:root", str(bot_dir), str(bot_dir / "tools"), str(config.FS_WORKER_PATH)])
    os.chmod(bot_dir, 0o711)
    os.chmod(bot_dir / "tools", 0o711)
    os.chmod(config.FS_WORKER_PATH, 0o644)
    db = config.DB_PATH
    for path in (config.ENV_PATH, db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        if path.exists():
            system(["chown", "root:root", str(path)])
            os.chmod(path, 0o600)
