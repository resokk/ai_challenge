---
id: admin-systemd
title: Запуск бота как systemd-сервиса
category: admin
commands: []
related: [admin-install, admin-help-server, admin-users]
---

# Запуск бота как systemd-сервиса

Статья для администратора. Чтобы бот работал постоянно и поднимался после
перезагрузки и сбоев, запустите его как сервис systemd.

## Unit-файл

Создайте `/etc/systemd/system/devbot.service` (пути — под ваше размещение):

```ini
[Unit]
Description=devbot Telegram assistant
After=network-online.target
Wants=network-online.target

[Service]
User=root
WorkingDirectory=/opt/devbot/final/bot
ExecStart=/opt/devbot/final/bot/.venv/bin/python bot.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Бот должен работать от root: он создаёт пользователей ОС и запускает процессы
от их имени. Пути к `bot.db`, `.env`, `mcp.json` и `prompts/` бот считает от
своего каталога, поэтому каталог запуска на них не влияет.

## Как запустить

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now devbot
journalctl -u devbot -f
```

Последняя команда показывает лог бота: подключение MCP-серверов, ошибки
модели, предупреждения о созданных учётных записях.

## Обновление и перезапуск

```bash
sudo systemctl restart devbot
```

Перезапуск нужен после изменения `.env`, `mcp.json` и системного промпта
`prompts/system.md`. MCP-серверы подключаются только при старте, поэтому
новый сервер или сервер, который был недоступен, появится после перезапуска.

## Вместе со справкой

Если вы подключили сервер справки, запускайте его отдельным сервисом раньше
бота — иначе бот при старте его не увидит и пропустит. Пример unit-файла — в
README справки, подробности — в статье о подключении справки.

## Что может пойти не так

- Сервис сразу останавливается — смотрите `journalctl -u devbot`. Типичные
  причины: не root (`Ошибка: нужны права root: запустите от root`), не
  заполнен `.env` (`Ошибка: в .env не заданы …`), нет утилит
  (`Ошибка: не найдены команды: …`).
- В логе `MCP server … is unavailable, skipping it` — MCP-сервер не поднялся;
  бот работает без него.
- Бот не отвечает пользователям — проверьте токен и доступ сервера к
  api.telegram.org.
