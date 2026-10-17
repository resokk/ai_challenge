"""The one place that starts processes on behalf of a user: every git, pip,
venv, fs_worker and user-code process goes through `run_as`, so each of them
runs as the user's own OS account, with a clean environment and limits."""

from __future__ import annotations

import asyncio
import os
import resource
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import config

DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
# How much of each output stream is kept in memory: the head and the tail of
# this size each, so a script printing forever cannot exhaust the bot's memory.
CAPTURE_BYTES = 1_000_000


@dataclass(frozen=True)
class UserContext:
    """Whose call this is: the Telegram user, their OS account and their
    active task. Tools take everything user-specific from here."""

    telegram_id: int
    os_user: str
    uid: int
    gid: int
    home: Path
    task_id: str | None = None
    repo_url: str | None = None

    @property
    def workspace(self) -> Path:
        return self.home / ".workspace"

    @property
    def venv(self) -> Path:
        return self.home / ".venv"

    @property
    def ssh_dir(self) -> Path:
        return self.home / ".ssh"

    @property
    def ssh_key(self) -> Path:
        return self.ssh_dir / "id_ed25519"


@dataclass
class RunResult:
    exit_code: int
    duration: float
    stdout: str
    stderr: str
    timed_out: bool = False


def _limits() -> None:
    """Runs in the child before exec, so the limits bind the user's process only."""
    mb = 1024 * 1024
    for limit, value in (
        (resource.RLIMIT_CPU, config.RUN_CPU_SEC),
        (resource.RLIMIT_AS, config.RUN_MEMORY_MB * mb),
        (resource.RLIMIT_FSIZE, config.RUN_FILE_MB * mb),
        # Counted per real uid, so this caps the user's account as a whole.
        (resource.RLIMIT_NPROC, config.RUN_NPROC),
    ):
        resource.setrlimit(limit, (value, value))


def identity(ctx: UserContext) -> dict:
    """The Popen arguments that make the process the user's: their uid and gid,
    no supplementary groups inherited from root, and their resource limits."""
    return {"user": ctx.uid, "group": ctx.gid, "extra_groups": [], "preexec_fn": _limits}


def environment(ctx: UserContext, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Built from scratch: nothing from the bot's own environment (and so none
    of its secrets) reaches the user's process."""
    return {
        "HOME": str(ctx.home),
        "USER": ctx.os_user,
        "LOGNAME": ctx.os_user,
        "PATH": DEFAULT_PATH,
        "LANG": "C.UTF-8",
        **(extra or {}),
    }


async def run_as(
    ctx: UserContext,
    argv: list[str],
    cwd: Path | str,
    timeout: float,
    *,
    env: dict[str, str] | None = None,
    input: bytes | None = None,
) -> RunResult:
    return await execute(argv, cwd, timeout, environment(ctx, env), identity(ctx), input)


async def execute(
    argv: list[str],
    cwd: Path | str,
    timeout: float,
    env: dict[str, str],
    spawn: dict,
    input: bytes | None = None,
) -> RunResult:
    """Run `argv` without a shell in its own session, so a timeout can kill the
    whole process group - children included."""
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            umask=0o077,
            **spawn,
        )
    except OSError as error:
        return RunResult(127, time.monotonic() - started, "", f"не удалось запустить {argv[0]}: {error.strerror}")

    async def feed() -> None:
        if input is not None:
            try:
                process.stdin.write(input)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()

    readers = asyncio.gather(_capture(process.stdout), _capture(process.stderr), feed(), process.wait())
    timed_out = False
    try:
        stdout, stderr, _, _ = await asyncio.wait_for(asyncio.shield(readers), timeout)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr, _, _ = await asyncio.wait_for(readers, 5)
        except asyncio.TimeoutError:
            # A descendant that left the group still holds the pipes open.
            stdout, stderr = b"", b""
            if process.returncode is None:
                await process.wait()
    stderr_text = stderr.decode(errors="replace")
    if timed_out:
        stderr_text += f"\n[таймаут: процесс убит через {timeout:g} с]"
    return RunResult(
        exit_code=process.returncode,
        duration=time.monotonic() - started,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr_text,
        timed_out=timed_out,
    )


async def _capture(stream: asyncio.StreamReader) -> bytes:
    head = bytearray()
    tail = bytearray()
    dropped = 0
    while chunk := await stream.read(65536):
        if len(head) < CAPTURE_BYTES:
            take = CAPTURE_BYTES - len(head)
            head += chunk[:take]
            chunk = chunk[take:]
        tail += chunk
        if len(tail) > CAPTURE_BYTES:
            dropped += len(tail) - CAPTURE_BYTES
            del tail[: len(tail) - CAPTURE_BYTES]
    if dropped:
        return bytes(head) + f"\n[... не сохранено {dropped} байт ...]\n".encode() + bytes(tail)
    return bytes(head + tail)
