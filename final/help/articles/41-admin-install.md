---
id: admin-install
title: Установка и настройка бота: .env и mcp.json
category: admin
commands: []
related: [admin-users, admin-systemd, admin-isolation, admin-help-server]
---

# Установка и настройка бота: .env и mcp.json

Статья для администратора: как поставить бота, заполнить `.env` и `mcp.json` и
запустить его. Бот работает только на Linux и запускается от root.

## Установка

```bash
cd final/bot
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Нужны системные `git`, `ssh-keygen`, `useradd`, `usermod`, `userdel`, `pkill`
и `/usr/bin/python3` (или другой путь в `SYSTEM_PYTHON`). Из системного Python
создаются venv пользователей, поэтому он должен быть доступен их учётным
записям — venv самого бота им недоступен.

## .env

Обязательны `TELEGRAM_BOT_TOKEN` и `DEEPSEEK_API_KEY`. Остальное имеет
значения по умолчанию:

- `DEEPSEEK_MODEL` (`deepseek-chat`), `DB_PATH` (`bot.db`),
  `MCP_CONFIG_PATH` (`mcp.json`) — пути считаются от каталога бота;
- контекст: `CONTEXT_LIMIT_TOKENS` (64000), `AGENTS_MD_MAX_CHARS` (20000),
  `MEMORY_MAX_CHARS` (4000);
- инструменты: `TOOL_TIMEOUT_SEC` (30), `TOOL_RESULT_MAX_CHARS` (20000),
  `MAX_TOOL_ITERATIONS` (8), `FILE_READ_MAX_CHARS` (20000),
  `FILE_WRITE_MAX_BYTES` (1000000);
- git: `GIT_TIMEOUT_SEC` (60), `GIT_DIFF_MAX_CHARS` (20000);
- запуск: `PIP_TIMEOUT_SEC` (300), `RUN_TIMEOUT_SEC` (60),
  `RUN_OUTPUT_MAX_CHARS` (3000), `RUN_MEMORY_MB` (1024), `RUN_CPU_SEC` (600),
  `RUN_FILE_MB` (512), `RUN_NPROC` (128).

## mcp.json

```json
{
  "servers": {
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {}},
    "help":  {"url": "http://127.0.0.1:8770/mcp"}
  }
}
```

stdio-сервер задаётся `command`/`args`/`env`, streamable HTTP — `url`.
Инструменты сервера получают префикс `<server>__`. Сервер, который не
поднялся, пропускается с записью в лог. Серверы запускаются процессом бота, то
есть от root, — подключайте только то, чему доверяете.

## Запуск

```bash
sudo .venv/bin/python bot.py
```

При старте бот проверяет Linux, root и наличие утилит, выставляет права на
свои файлы и проверяет учётные записи ОС всех одобренных пользователей
(недостающие создаёт с предупреждением в логе). Затем подключает MCP-серверы.

## Что может пойти не так

Бот завершается с кодом 1 и сообщением в stderr:

- `Ошибка: нужны права root: запустите от root`;
- `Ошибка: бот работает только на Linux (сейчас darwin)`;
- `Ошибка: не найдены команды: useradd, pkill` — поставьте пакеты с этими
  утилитами;
- `Ошибка: в .env не заданы TELEGRAM_BOT_TOKEN` — заполните `.env`.
