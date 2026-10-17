---
id: reference-tools
title: Справочные инструменты: PyPI, документация Python, время
category: tools
commands: [/tool]
related: [tools-list, tool-call, pip]
---

# Справочные инструменты: PyPI, документация Python, время

Три встроенных инструмента дают справочные сведения и не работают с файлами
проекта. Модель зовёт их сама, когда нужен точный факт, а вы можете вызвать
их через `/tool`.

## pypi_package_info — пакет на PyPI

`/tool pypi_package_info {"name": "httpx"}`

Ответ: имя и последняя версия, краткое описание, требуемый Python и ссылки:

```
httpx 0.27.2
The next generation HTTP client.
Требуемый Python: >=3.8
Ссылки:
- PyPI: https://pypi.org/project/httpx/
- …
```

Полезно перед `/pip`: узнать актуальную версию и поддерживает ли пакет вашу
версию Python.

## python_docs_link — документация Python

`/tool python_docs_link {"topic": "json.loads"}`

Ответ — ссылка на раздел docs.python.org:
`https://docs.python.org/3/library/json.html#json.loads`.

Работает для модулей и функций стандартной библиотеки (`pathlib`,
`asyncio.gather`), встроенных функций (`len` →
`https://docs.python.org/3/library/functions.html#len`) и встроенных
исключений (`ValueError` → страница `exceptions.html`). Скобки в конце можно
писать: `len()`.

## current_datetime — текущее время

`/tool current_datetime` — текущие дата и время UTC с днём недели, например
`2026-10-17 12:00:00 UTC (Saturday)`.

## Модель и справочные инструменты

Модели запрещено выдумывать версии и API. Если вы спрашиваете «какая последняя
версия pydantic и какой Python ей нужен?», модель вызовет
`pypi_package_info` и ответит по нему.

## Что может пойти не так

- `Ошибка: пакет X на PyPI не найден` — проверьте имя пакета (как в `pip
  install`).
- `Ошибка: «requests» не найден в стандартной библиотеке` — `python_docs_link`
  работает только со стандартной библиотекой; для сторонних пакетов
  используйте `pypi_package_info` и ссылки из его ответа.
- `Ошибка при вызове pypi_package_info: …` — нет связи с PyPI; повторите
  позже.
