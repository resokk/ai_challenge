---
id: example-fix-test
title: Пример: найти и исправить падающий тест
category: examples
commands: [/pip, /run, /git diff, /git commit]
related: [run-module, model-reads-code, model-edits-files, git-diff]
---

# Пример: найти и исправить падающий тест

Сценарий: в проекте падает тест. Запускаем тесты, поручаем модели найти
причину и исправить, проверяем и коммитим.

## 1. Подготовка

Задача уже выбрана (`/tasks api`). Ставим pytest и зависимости проекта:

Вы: /pip -r requirements.txt
Вы: /pip pytest

## 2. Запуск тестов

Вы: /run -m pytest -q
Бот: python -m pytest -q
код выхода 1, 1.8 с
Бот: stdout (кратко, полный вывод — в файле):
Прошло 23 теста, упал 1: tests/test_parser.py::test_empty_line —
AssertionError: assert None == [] (tests/test_parser.py:41).
Бот: (файл stdout.txt)

## 3. Поручение модели

Вы: Упал tests/test_parser.py::test_empty_line. Найди причину, исправь код
(не тест) и запусти pytest снова.

Модель работает сама — вы видите статусы:

Бот: вызываю read_file…
Бот: вызываю search_files…
Бот: вызываю edit_file…
Бот: вызываю run_python…

Бот: Причина: `parse_line` в app/parser.py возвращает None для пустой строки,
а тест ждёт пустой список. Исправил: для пустой строки функция возвращает
`[]`. pytest: 24 passed. Проверьте изменения: /git diff, затем /git commit.

## 4. Проверка и коммит

Вы: /git diff
Бот: app/parser.py — в `parse_line` добавлена обработка пустой строки.
Рисков не видно.

Вы: /git commit
Бот:
```
Коммит 8c1d2e0

Return empty list for empty lines in parse_line

Файлы:
M	app/parser.py
```

Сообщение коммита написала модель, потому что вы его не указали. Осталось
`/git push`.

## Советы

- Называйте конкретный тест и место: так модель быстрее найдёт причину.
- Явно скажите, что менять нельзя («не меняй тест»), — модель учтёт.
- Если правка не понравилась — `/git reset` откатит её, и можно попросить
  иначе.
- Тесты долгие? Запускайте один файл: `/run -m pytest tests/test_parser.py`
  — лимит на запуск 60 секунд.
