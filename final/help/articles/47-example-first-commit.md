---
id: example-first-commit
title: Пример: от /start до первого коммита в свой репозиторий
category: examples
commands: [/start, /git ssh, /tasks, /git, /git commit, /git push]
related: [register, ssh-key, task-create, git-commit, git-push]
---

# Пример: от /start до первого коммита в свой репозиторий

Сценарий целиком: получить доступ, подключить свой приватный репозиторий,
попросить модель внести правку и отправить коммит в GitHub.

## 1. Доступ

Вы: /start
Бот: Заявка принята и ждёт одобрения администратора.

…администратор одобряет заявку…

Бот: Ваша заявка одобрена — можно работать. /help — список команд.

## 2. SSH-ключ

Вы: /git ssh
Бот: Создан новый SSH-ключ.
Публичный ключ:
Бот: ssh-ed25519 AAAA… devbot-123456789
Бот: Добавьте его в GitHub: Settings → SSH and GPG keys → New SSH key, или в
репозитории Settings → Deploy keys (отметьте Allow write access). …

Скопируйте ключ и добавьте его в GitHub: в профиль или как Deploy key
репозитория с отметкой `Allow write access`.

## 3. Задача и рабочая копия

Вы: /tasks notes octo/notes
Бот: Активная задача: notes — https://github.com/octo/notes
Рабочая копия готова.

## 4. Правка руками модели

Вы: Добавь в README.md раздел «Установка» с командой pip install -r
requirements.txt.
Бот: (модель читает README.md и правит его) Добавил раздел «Установка» в
README.md. Проверьте изменения: /git diff, затем /git commit.

## 5. Проверка

Вы: /git
Бот:
```
Ветка: main → origin/main
Впереди: 0, позади: 0
Изменения:
  изменён: README.md
```

Вы: /git diff
Бот: README.md — добавлен раздел «Установка» с командой установки
зависимостей. Рисков не видно.

## 6. Коммит и отправка

Вы: /git commit Add installation section to README
Бот:
```
Коммит 3f2a9c1

Add installation section to README

Файлы:
M	README.md
```

Вы: /git push
Бот: Изменения отправлены.

Коммит появился в GitHub. Сообщение коммита можно было не писать: `/git commit`
без текста попросит модель составить его по изменениям.

## Если что-то пошло не так

- На шаге 3 `Клонировать не удалось:` — ключ не добавлен в GitHub или
  добавлен не туда. Исправьте и повторите `/tasks notes`.
- На шаге 6 `push не удался:` с `Нет доступа: …` — у Deploy key не отмечено
  `Allow write access`.
