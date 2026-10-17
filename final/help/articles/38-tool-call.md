---
id: tool-call
title: Вызвать инструмент самому: /tool name {json}
category: tools
commands: [/tool]
related: [tools-list, reference-tools, mcp-tools, run-output]
---

# Вызвать инструмент самому: /tool name {json}

`/tool` вызывает любой инструмент из каталога напрямую, без модели. Результат
приходит только вам и в историю диалога не попадает.

## Как сделать

Формат: `/tool <name> [json-аргументы]`. Аргументы — JSON-объект по схеме
параметров из `/tools`; если аргументов нет, их можно не писать.

Примеры:

- `/tool current_datetime`
- `/tool pypi_package_info {"name": "httpx"}`
- `/tool read_file {"path": "app/main.py", "start_line": 1, "end_line": 40}`
- `/tool search_files {"query": "TODO", "glob": "*.py"}`
- `/tool python_outline {"path": "app/db.py"}`
- `/tool run_python {"target": "pytest", "module": true, "args": ["-q"]}`
- `/tool help__help {"question": "как подключить приватный репозиторий"}`

Пока инструмент работает, бот показывает `выполняется <name>…`. Результат
`run_python`, `pip_install` и `pip_list` приходит так же, как у `/run` и
`/pip`: заголовок, stdout и stderr.

## Чем /tool отличается от просьбы модели

- `/tool` — результат видите только вы; модель не вызывается, история и
  счётчик токенов не меняются.
- Просьба модели («прочитай main.py») — инструмент вызывает модель, результат
  видит только она, а вы получаете её ответ.

`/run` и `/pip` — короткие формы тех же вызовов: `/run main.py` — то же, что
`/tool run_python {"target": "main.py"}`.

## Что может пойти не так

- `/tool` без аргументов — бот покажет подсказку:
  `Использование: /tool <name> [json-аргументы]` с примером и ссылкой на
  `/tools`.
- `Аргументы — JSON-объект, например: /tool pypi_package_info {"name":
  "httpx"}` — аргументы не JSON-объект: проверьте фигурные скобки и двойные
  кавычки.
- `Ошибка: инструмента «readfile» нет. Доступны: …` — опечатка в имени;
  бот перечислит все доступные.
- `Ошибка в аргументах read_file (аргументы): 'path' is a required property` —
  не хватает обязательного аргумента или он не того типа.
- `Ошибка: read_file не ответил за 30 с` — инструмент не уложился во время.
