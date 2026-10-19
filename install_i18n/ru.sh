#!/usr/bin/env bash
# install_i18n/ru.sh — Russian messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Проверка Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 не найден. Установите python3 и повторите попытку."
I_PYTHON_OLD="Требуется Python 3.10+ (найдено: %s)."

I_VENV_EXISTS="Найдена существующая venv в .venv — создание пропущено."
I_VENV_CREATE="Создание виртуальной среды в .venv ..."
I_PIP_UPGRADE="Обновление pip..."
I_REQS_INSTALL="Установка зависимостей Python..."
I_REQS_DONE="Пакеты Python установлены."

I_OLLAMA_EXISTS="Ollama уже установлен (%s)."
I_OLLAMA_INSTALL="Установка Ollama..."
I_BREW_MISSING="Homebrew не найден. Установите Ollama вручную с https://ollama.com/download/mac"
I_BREW_RERUN="Затем повторно запустите этот скрипт."
I_OLLAMA_DONE="Ollama установлен."
I_OLLAMA_START="Запуск службы Ollama..."
I_OLLAMA_STARTED="Служба Ollama запущена (pid %s)."
I_OLLAMA_RUNNING="Служба Ollama уже запущена."
I_OLLAMA_TIMEOUT="Служба Ollama не запустилась вовремя."

I_MODEL_CHECK="Проверка модели '%s'..."
I_MODEL_EXISTS="Модель '%s' уже присутствует."
I_MODEL_PULL="Загрузка '%s' — это может занять несколько минут..."
I_MODEL_READY="Модель '%s' готова."

I_DIRS_READY="Каталоги данных готовы."
I_SETTINGS_CREATED="Настройки по умолчанию созданы в data/settings.json."
I_DB_ERROR="При настройке базы данных возникла ошибка — продолжите в /databases в игре."
I_DB_MISSING="tools/install_dbs.py не найден — загрузка базы данных пропущена."

I_DONE="Установка завершена!"
I_START_HINT="Чтобы запустить игру:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Виртуальная среда не найдена. Сначала запустите ./install.sh."
R_NO_UVICORN="uvicorn не найден в venv. Сначала запустите ./install.sh."
R_OLLAMA_START="Ollama не запущен — запуск в фоновом режиме..."
R_OLLAMA_TIMEOUT="Ollama не ответил — функции LLM будут отключены."
R_PORT_BUSY="Порт %s уже используется. Попытка использовать порт 8080..."
R_SHUTDOWN="Завершение работы..."
R_NO_CKPT="Контрольные точки модели ИИ не найдены — функции ИИ будут ограничены."
R_DL_CKPT="Загрузите их через ./install.sh или откройте /databases в игре."
R_STARTING="Запуск Nine Men's Morris по адресу %s ..."
R_WAITING="Ожидание готовности сервера..."
R_SERVER_SLOW="Сервер слишком долго не отвечал — браузер открывается в любом случае."
R_OPEN_BROWSER="Открытие браузера по адресу %s"
R_NO_BROWSER="Не удалось обнаружить браузер. Откройте %s вручную."
R_RUNNING="Nine Men's Morris работает по адресу %s"
R_STOP_HINT="Нажмите Ctrl-C для остановки."
