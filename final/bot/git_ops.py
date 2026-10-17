"""Working copies, SSH keys and the `/git` commands. Every git and ssh-keygen
process runs as the user's OS account through `run_as`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable

import config
import run_as
from run_as import RunResult, UserContext
from tools.files import worker

REPO_PATTERN = re.compile(
    r"(?:https://github\.com/)?(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<name>[A-Za-z0-9._-]{1,100}?)(?:\.git)?/?"
)
NO_WORKSPACE = "Нет рабочей копии — выберите задачу: /tasks <id>"
NO_KEY = "SSH-ключа нет — создайте его: /git ssh, и добавьте в GitHub."


class GitError(Exception):
    """A failure to show the user as is."""


def normalize_repo(text: str) -> str | None:
    """`owner/name` or a GitHub URL, as `https://github.com/owner/name`."""
    match = REPO_PATTERN.fullmatch(text.strip())
    if not match or match["name"] in (".", ".."):
        return None
    return f"https://github.com/{match['owner']}/{match['name']}"


def clone_url(repo_url: str, ssh: bool) -> str:
    """Where `repo_url` is cloned from and pushed to. Tests swap this for a
    local bare repository."""
    path = repo_url.removeprefix("https://github.com/")
    return f"git@github.com:{path}.git" if ssh else f"{repo_url}.git"


def _env(ctx: UserContext) -> dict[str, str]:
    known_hosts = ctx.ssh_dir / "known_hosts"
    return {
        "GIT_SSH_COMMAND": (
            f"ssh -i {ctx.ssh_key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "
            f"-o UserKnownHostsFile={known_hosts}"
        ),
        "GIT_TERMINAL_PROMPT": "0",
    }


async def git(ctx: UserContext, *args: str, cwd=None) -> RunResult:
    return await run_as.run_as(ctx, ["git", *args], cwd or ctx.workspace, config.GIT_TIMEOUT_SEC, env=_env(ctx))


async def _check(ctx: UserContext, *args: str) -> str:
    result = await git(ctx, *args)
    if result.exit_code != 0:
        raise GitError(f"git {args[0]}: {_tail(result)}")
    return result.stdout


def _tail(result: RunResult, limit: int = 1500) -> str:
    text = (result.stderr.strip() or result.stdout.strip() or f"код {result.exit_code}")
    return text[-limit:]


async def _test(ctx: UserContext, flag: str, path) -> bool:
    return (await run_as.run_as(ctx, ["test", flag, str(path)], ctx.home, 10)).exit_code == 0


async def has_key(ctx: UserContext) -> bool:
    return await _test(ctx, "-f", ctx.ssh_key)


async def has_repo(ctx: UserContext) -> bool:
    return ctx.task_id is not None and await _test(ctx, "-d", ctx.workspace / ".git")


async def require_repo(ctx: UserContext) -> None:
    if not await has_repo(ctx):
        raise GitError(NO_WORKSPACE)


# SSH key


async def public_key(ctx: UserContext, regenerate: bool = False) -> tuple[str, bool]:
    """The user's public key, generating the pair if there is none (or if
    `regenerate`). Returns the key and whether it was generated now. The
    private key never leaves the user's account."""
    created = False
    if regenerate or not await has_key(ctx):
        steps = [
            ["mkdir", "-p", "-m", "700", str(ctx.ssh_dir)],
            ["rm", "-f", str(ctx.ssh_key), f"{ctx.ssh_key}.pub"],
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"devbot-{ctx.telegram_id}", "-f", str(ctx.ssh_key)],
        ]
        for argv in steps:
            result = await run_as.run_as(ctx, argv, ctx.home, 30)
            if result.exit_code != 0:
                raise GitError(f"не удалось создать ключ: {_tail(result)}")
        created = True
    result = await run_as.run_as(ctx, ["ssh-keygen", "-y", "-f", str(ctx.ssh_key)], ctx.home, 30)
    if result.exit_code != 0:
        raise GitError(f"не удалось прочитать ключ: {_tail(result)}")
    return result.stdout.strip(), created


# working copy


@dataclass
class SwitchResult:
    lost_changes: bool
    error: str | None = None


async def switch_workspace(ctx: UserContext) -> SwitchResult:
    """Recreate `<home>/.workspace` as a fresh clone of the task's repository.
    Uncommitted work of the previous copy is lost; the result says if there was any."""
    lost = await has_repo(ctx) and bool((await git(ctx, "status", "--porcelain")).stdout.strip())
    await worker(ctx, "reset_workspace")
    ssh = await has_key(ctx)
    result = await git(ctx, "clone", "-q", clone_url(ctx.repo_url, ssh=ssh), ".")
    if result.exit_code != 0:
        return SwitchResult(lost, _tail(result))
    await git(ctx, "remote", "set-url", "--push", "origin", clone_url(ctx.repo_url, ssh=True))
    return SwitchResult(lost)


STATUS_LABELS = {"M": "изменён", "A": "добавлен", "D": "удалён", "R": "переименован", "C": "скопирован",
                 "U": "конфликт", "T": "изменён тип", "?": "новый"}


async def status(ctx: UserContext) -> str:
    await require_repo(ctx)
    lines = (await _check(ctx, "status", "--porcelain=v1", "--branch")).splitlines()
    header = lines[0].removeprefix("## ") if lines else ""
    branch, _, rest = header.partition(" ")
    local, _, upstream = branch.partition("...")
    out = [f"Ветка: {local}" + (f" → {upstream}" if upstream else " (без upstream)")]
    ahead = re.search(r"ahead (\d+)", rest)
    behind = re.search(r"behind (\d+)", rest)
    if upstream:
        out.append(f"Впереди: {ahead[1] if ahead else 0}, позади: {behind[1] if behind else 0}")
    files = lines[1:]
    if not files:
        out.append("Рабочая копия чистая.")
    else:
        out.append("Изменения:")
        for line in files:
            code = (line[:2].strip() or "M")[0]
            out.append(f"  {STATUS_LABELS.get(code, code)}: {line[3:]}")
    return "\n".join(out)


async def commit(
    ctx: UserContext,
    message: str | None,
    author: str,
    make_message: Callable[[str], Awaitable[str]],
) -> str:
    """`git add -A` and commit, with `make_message(diff)` writing the message
    when the user gave none."""
    await require_repo(ctx)
    await _check(ctx, "add", "-A")
    if (await git(ctx, "diff", "--cached", "--quiet")).exit_code == 0:
        return "Нечего коммитить: рабочая копия чистая."
    if not message:
        diff = await _check(ctx, "diff", "--cached")
        message = (await make_message(diff[: config.GIT_DIFF_MAX_CHARS])).strip()
    email = f"{ctx.telegram_id}@users.noreply.telegram"
    await _check(ctx, "-c", f"user.name={author}", "-c", f"user.email={email}", "commit", "-q", "-m", message)
    short = (await _check(ctx, "log", "-1", "--format=%h")).strip()
    files = (await _check(ctx, "show", "--name-status", "--format=", "HEAD")).strip()
    return f"Коммит {short}\n\n{message}\n\nФайлы:\n{files}"


async def push(ctx: UserContext) -> str:
    await require_repo(ctx)
    if not await has_key(ctx):
        raise GitError(NO_KEY)
    result = await git(ctx, "push", "origin", "HEAD")
    if result.exit_code == 0:
        return "Изменения отправлены.\n" + _tail(result, 1000)
    text = _tail(result)
    if "non-fast-forward" in text or "rejected" in text or "fetch first" in text:
        hint = "В удалённой ветке есть коммиты, которых нет у вас; принудительный push не поддерживается."
    elif "Permission denied" in text or "denied" in text or "Could not read from remote" in text:
        hint = "Нет доступа: добавьте ключ из /git ssh в GitHub (SSH keys или Deploy key с правом записи)."
    else:
        hint = "Проверьте доступ к репозиторию и повторите."
    raise GitError(f"push не удался:\n{text}\n\n{hint}")


async def diff_for_review(ctx: UserContext) -> str | None:
    """`git diff HEAD` plus the untracked files, cut to the limit; None when
    there is nothing to review."""
    await require_repo(ctx)
    has_head = (await git(ctx, "rev-parse", "--verify", "-q", "HEAD")).exit_code == 0
    diff = await _check(ctx, "diff", "HEAD") if has_head else await _check(ctx, "diff")
    untracked = (await _check(ctx, "ls-files", "--others", "--exclude-standard")).strip()
    if not diff.strip() and not untracked:
        return None
    text = diff
    if untracked:
        text += "\n\nНеотслеживаемые файлы:\n" + untracked
    if len(text) > config.GIT_DIFF_MAX_CHARS:
        text = text[: config.GIT_DIFF_MAX_CHARS] + "\n[diff обрезан]"
    return text


async def pending_changes(ctx: UserContext) -> list[str]:
    """What `/git reset` would throw away."""
    await require_repo(ctx)
    return (await _check(ctx, "status", "--porcelain")).splitlines()


async def reset(ctx: UserContext) -> None:
    """Back to HEAD: tracked changes reverted, untracked files removed,
    ignored files left alone."""
    await require_repo(ctx)
    await _check(ctx, "reset", "-q", "--hard", "HEAD")
    await _check(ctx, "clean", "-fdq")
