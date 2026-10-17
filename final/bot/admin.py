"""User management from the console:

    python admin.py list [--status pending|approved|blocked]
    python admin.py approve <telegram_id>
    python admin.py block <telegram_id>
    python admin.py delete <telegram_id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

import config
import os_users
from storage import Storage

NOTIFICATIONS = {
    "approved": "Ваша заявка одобрена — можно работать. /help — список команд.",
    "blocked": "Ваш доступ к боту заблокирован администратором.",
}


def notify(telegram_id: int, text: str) -> None:
    """Tell the user through the Bot API; a failure is a warning, not an error."""
    try:
        response = httpx.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": telegram_id, "text": text},
            timeout=15,
        )
        response.raise_for_status()
    except Exception as error:
        print(f"Предупреждение: уведомление не отправлено: {type(error).__name__}", file=sys.stderr)


def cmd_list(store: Storage, status: str | None) -> int:
    rows = store.list_users(status)
    if not rows:
        print("Пользователей нет.")
        return 0
    print(f"{'id':>4}  {'telegram_id':>12}  {'status':<9}  {'os user':<10}  name")
    for row in rows:
        name = " ".join(filter(None, [row["first_name"], f"@{row['username']}" if row["username"] else None]))
        print(f"{row['id']:>4}  {row['telegram_id']:>12}  {row['status']:<9}  "
              f"{os_users.os_username(row['id']):<10}  {name}")
    return 0


def cmd_approve(store: Storage, user) -> None:
    asyncio.run(os_users.ensure_os_user(user))
    store.set_status(user["telegram_id"], "approved")
    print(f"Одобрен: {user['telegram_id']} (пользователь ОС {os_users.os_username(user['id'])})")
    notify(user["telegram_id"], NOTIFICATIONS["approved"])


def cmd_block(store: Storage, user) -> None:
    os_users.block_os_user(user["id"])
    store.set_status(user["telegram_id"], "blocked")
    print(f"Заблокирован: {user['telegram_id']}")
    notify(user["telegram_id"], NOTIFICATIONS["blocked"])


def cmd_delete(store: Storage, user) -> None:
    os_users.delete_os_user(user["id"])
    store.delete_user(user["telegram_id"])
    print(f"Удалён: {user['telegram_id']} вместе с пользователем ОС, задачами и историей")


ACTIONS = {"approve": cmd_approve, "block": cmd_block, "delete": cmd_delete}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Управление пользователями бота")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="список пользователей")
    listing.add_argument("--status", choices=["pending", "approved", "blocked"])
    for name in ACTIONS:
        commands.add_parser(name).add_argument("telegram_id", type=int)
    args = parser.parse_args(argv)

    if args.command != "list":
        os_users.require_root()
    store = Storage(config.DB_PATH)
    try:
        if args.command == "list":
            return cmd_list(store, args.status)
        user = store.get_user(args.telegram_id)
        if user is None:
            print(f"Ошибка: пользователя с telegram_id {args.telegram_id} нет", file=sys.stderr)
            return 2
        try:
            ACTIONS[args.command](store, user)
        except os_users.OsUserError as error:
            print(f"Ошибка: {error}", file=sys.stderr)
            return 1
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
