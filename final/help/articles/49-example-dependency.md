---
id: example-dependency
title: Пример: добавить зависимость и новую функцию с тестами
category: examples
commands: [/pip, /run, /tool, /git commit]
related: [pip, reference-tools, model-edits-files, run-module]
---

# Пример: добавить зависимость и новую функцию с тестами

Сценарий: нужна новая функция, которой понадобится сторонний пакет. Узнаём
версию, ставим пакет, просим модель написать функцию с тестом, проверяем и
коммитим.

## 1. Узнать версию пакета

Вы: /tool pypi_package_info {"name": "httpx"}
Бот: httpx 0.27.2
The next generation HTTP client.
Требуемый Python: >=3.8
…

## 2. Поставить пакет

Вы: /pip httpx pytest
Бот: pip install httpx pytest
код выхода 0, 5.2 с
…

Модель могла бы поставить пакет и сама (`pip_install`), но так вы контролируете
версию.

## 3. Поручение модели

Вы: Напиши в app/http.py функцию fetch_status(url: str) -> int на httpx,
которая возвращает код ответа. Добавь тест в tests/test_http.py с
httpx.MockTransport, без сети. Добавь httpx в requirements.txt и запусти
pytest.

Бот: вызываю list_files…
Бот: вызываю read_file…
Бот: вызываю write_file…
Бот: вызываю edit_file…
Бот: вызываю run_python…

Бот: Создал app/http.py с fetch_status и tests/test_http.py (2 теста), добавил
httpx в requirements.txt. pytest: 2 passed. Проверьте: /git diff, затем
/git commit.

## 4. Проверка

Вы: /run -m pytest -q tests/test_http.py
Бот: python -m pytest -q tests/test_http.py
код выхода 0, 0.9 с
Бот: stdout:
2 passed in 0.12s

Вы: /git diff — разбор изменений моделью.

## 5. Коммит

Вы: /git commit Add fetch_status helper with tests
Вы: /git push

## Что учесть

- venv общий для всех ваших задач: поставленный пакет будет виден и в других
  проектах.
- Записывайте зависимости в `requirements.txt` проекта — в другом окружении
  их поставит `/pip -r requirements.txt`.
- Если модель создаёт файлы, она не удаляет их; лишний файл откатит
  `/git reset` (вместе со всеми незакоммиченными изменениями).
- Сеть из вашего кода доступна, но в тестах лучше обходиться без неё, как в
  примере.
