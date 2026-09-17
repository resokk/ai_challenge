from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path("/tmp/bot")

# The user whose update is being handled right now. One update is handled in one
# task, so binding it once at the edge carries it to everything that update
# touches - the replies going back, and the DeepSeek calls underneath them -
# without a user id being threaded through every call that might want to log.
_current_user: ContextVar[int | None] = ContextVar("protocol_log_user", default=None)


def bind(user_id: int) -> None:
    """Send everything logged while handling this update to that user's file."""
    _current_user.set(user_id)


def record(direction: str, text: str) -> None:
    """Append one exchange to the current user's log, or drop it if nothing is
    bound - a log entry with no one to attribute it to is worse than none.

    The log is a side channel, so a failure to write it must never take down
    the exchange it is recording: it is reported to the application log instead."""
    user_id = _current_user.get()
    if user_id is None:
        return

    entry = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {direction}\n{text}\n\n"
    try:
        # 0700: these files hold whole conversations, and /tmp is shared.
        LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (LOG_DIR / f"{user_id}.log").open("a", encoding="utf-8") as log_file:
            log_file.write(entry)
    except OSError:
        logger.exception("Could not write the protocol log for user %s", user_id)
