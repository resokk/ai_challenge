# help — справка по боту devbot

50 статей о том, как пользоваться Telegram-ботом из `final/bot/`, семантический
индекс по ним на локальной модели `intfloat/multilingual-e5-small` и MCP-сервер
с одним инструментом `help`. Инструмент находит подходящие фрагменты статей,
отдаёт их DeepSeek на суммаризацию и возвращает короткий ответ со ссылками на
статьи. В боте он виден как `help__help`.

Справка — самостоятельная часть: бот её не импортирует и общается с ней только
по MCP. Все пути считаются от каталога `help/`, а не от текущей директории.

```
articles/        50 статей NN-slug.md
articles.py      загрузка и проверка статей, разбиение на фрагменты
embedder.py      e5 через transformers: embed_passages, embed_query
index.py         построение/загрузка индекса, поиск; CLI
answer.py        поиск + суммаризация через DeepSeek; CLI
server.py        MCP-сервер, инструмент help
prompts/answer.md  системный промпт суммаризации
index/           генерируется: vectors.npy, chunks.json (не в git)
```

## Установка

```bash
cd final/help
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # DEEPSEEK_API_KEY, при желании DEEPSEEK_MODEL
```

Модель эмбеддингов (~470 МБ) скачивается с Hugging Face при первом построении
индекса, дальше берётся из кэша без сети. Офлайн можно указать локальную копию
модели переменной `HELP_EMBED_MODEL` (путь к каталогу модели); после смены
модели индекс перестраивается сам.

## Индекс

```bash
.venv/bin/python index.py build                                   # перестроить принудительно
.venv/bin/python index.py search "как добавить ssh ключ в github" -k 5
.venv/bin/python index.py search "бот не пушит" --category git
```

Индекс (`index/vectors.npy`, `index/chunks.json`) хранит имя модели и sha256
содержимого всех статей. При старте сервера и в CLI он перестраивается, если
его нет, модель другая или статьи изменились; иначе загружается.

## Ответ на вопрос

```bash
.venv/bin/python answer.py "как закоммитить изменения"
```

Печатает ответ модели и строку `Статьи: title (id), …`. Если DeepSeek
недоступен (сеть, таймаут 60 с, ключ, лимит), причина пишется в лог, а вместо
ответа возвращаются найденные фрагменты с пометкой
`Суммаризация недоступна, найденные фрагменты справки:`.

## MCP-сервер

```bash
.venv/bin/python server.py
```

Streamable HTTP на `127.0.0.1:8770` (только loopback), путь `/mcp`. Индекс,
модель и `.env` загружаются до приёма запросов; без `DEEPSEEK_API_KEY` сервер не
стартует (сообщение в stderr, код 1). Инструмент один — `help(question: str)`;
пустой вопрос или вопрос длиннее 1000 символов — `ToolError`.

## Подключение к боту

В `final/bot/mcp.json`:

```json
"help": {"url": "http://127.0.0.1:8770/mcp"}
```

Сервер справки должен работать к моменту запуска бота — MCP-серверы бот
подключает только при старте. Проверка: в `/tools` есть `help__help`, а
`/tool help__help {"question": "как подключить приватный репозиторий"}`
возвращает ответ. Системный промпт бота велит модели звать `help__help` на
вопросы о самом боте. Если сервер не запущен, бот пропускает его и работает
без справки.

## systemd

Справка — отдельный сервис, который стартует раньше бота. Ей не нужен root.

```ini
# /etc/systemd/system/devbot-help.service
[Unit]
Description=devbot help MCP server
After=network-online.target
Wants=network-online.target
Before=devbot.service

[Service]
User=devbot-help
WorkingDirectory=/opt/devbot/final/help
ExecStart=/opt/devbot/final/help/.venv/bin/python server.py
Environment=HF_HOME=/opt/devbot/final/help/.cache/huggingface
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Пользователь `devbot-help` должен иметь права на запись в `index/` и в
`HF_HOME`. В `devbot.service` добавьте `Wants=devbot-help.service` и
`After=devbot-help.service`.

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now devbot-help
sudo systemctl restart devbot
journalctl -u devbot-help -f
```

## Тесты

```bash
.venv/bin/python -m pytest
```

Обычные тесты работают без сети: модель заменена детерминированным мешком
слов, DeepSeek — фейковым клиентом. Проверяются статьи (ровно 50, объём,
`related`, покрытие каждой команды из `CommandHandler(...)` в `bot/bot.py` и
подкоманд `/git`), разбиение на фрагменты, индекс, `answer` и сервер.
Тесты `@pytest.mark.model` прогоняют настоящую e5 на реальных вопросах и
пропускаются, если модели нет в кэше Hugging Face.

## Как добавить или изменить статью

1. Файл `articles/NN-slug.md`; `slug` совпадает с `id` во front matter:

```markdown
---
id: slug
title: Заголовок
category: start | tasks | git | files | run | chat | tools | admin | troubleshooting | examples
commands: [/команда, /git подкоманда]
related: [id-другой-статьи]
---

# Заголовок

Вступление.

## Раздел
…
```

2. 150–500 слов; каждая секция `##` становится отдельным фрагментом поиска.
   Команды и тексты сообщений бота — дословно из кода бота.
3. Чтобы сохранить ровно 50 статей, замените существующую, а не добавляйте
   новую (иначе тест упадёт).
4. `pytest`, затем перезапустите сервер — индекс перестроится при старте.

Новая команда бота без статьи с ней в `commands` роняет тест покрытия.
