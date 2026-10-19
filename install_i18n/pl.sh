#!/usr/bin/env bash
# install_i18n/pl.sh — Polish messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Sprawdzanie Pythona..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Nie znaleziono Pythona 3. Zainstaluj python3 i spróbuj ponownie."
I_PYTHON_OLD="Wymagany Python 3.10+ (znaleziono: %s)."

I_VENV_EXISTS="Znaleziono istniejące venv w .venv — pomijanie tworzenia."
I_VENV_CREATE="Tworzenie środowiska wirtualnego w .venv ..."
I_PIP_UPGRADE="Aktualizowanie pip..."
I_REQS_INSTALL="Instalowanie zależności Pythona..."
I_REQS_DONE="Pakiety Pythona zainstalowane."

I_OLLAMA_EXISTS="Ollama jest już zainstalowany (%s)."
I_OLLAMA_INSTALL="Instalowanie Ollama..."
I_BREW_MISSING="Nie znaleziono Homebrew. Zainstaluj Ollama ręcznie ze strony https://ollama.com/download/mac"
I_BREW_RERUN="Następnie uruchom ponownie ten skrypt."
I_OLLAMA_DONE="Ollama zainstalowany."
I_OLLAMA_START="Uruchamianie usługi Ollama..."
I_OLLAMA_STARTED="Usługa Ollama uruchomiona (pid %s)."
I_OLLAMA_RUNNING="Usługa Ollama jest już uruchomiona."
I_OLLAMA_TIMEOUT="Usługa Ollama nie uruchomiła się w czasie."

I_MODEL_CHECK="Sprawdzanie modelu '%s'..."
I_MODEL_EXISTS="Model '%s' jest już obecny."
I_MODEL_PULL="Pobieranie '%s' — może to potrwać kilka minut..."
I_MODEL_READY="Model '%s' gotowy."

I_DIRS_READY="Katalogi danych gotowe."
I_SETTINGS_CREATED="Ustawienia domyślne utworzone w data/settings.json."
I_DB_ERROR="Konfiguracja bazy danych napotkała błąd — kontynuuj w /databases w grze."
I_DB_MISSING="Nie znaleziono tools/install_dbs.py — pomijanie pobierania bazy danych."

I_DONE="Instalacja zakończona!"
I_START_HINT="Aby uruchomić grę:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Nie znaleziono środowiska wirtualnego. Najpierw uruchom ./install.sh."
R_NO_UVICORN="Nie znaleziono uvicorn w venv. Najpierw uruchom ./install.sh."
R_OLLAMA_START="Ollama nie działa — uruchamianie w tle..."
R_OLLAMA_TIMEOUT="Ollama nie odpowiedział — funkcje LLM zostaną wyłączone."
R_PORT_BUSY="Port %s jest już zajęty. Próba użycia portu 8080..."
R_SHUTDOWN="Zamykanie..."
R_NO_CKPT="Nie znaleziono punktów kontrolnych modelu AI — funkcje AI będą ograniczone."
R_DL_CKPT="Pobierz je przez ./install.sh lub otwórz /databases w grze."
R_STARTING="Uruchamianie Nine Men's Morris pod adresem %s ..."
R_WAITING="Oczekiwanie na gotowość serwera..."
R_SERVER_SLOW="Serwer zbyt długo nie odpowiadał — przeglądarka zostanie otwarta mimo to."
R_OPEN_BROWSER="Otwieranie przeglądarki pod adresem %s"
R_NO_BROWSER="Nie można wykryć przeglądarki. Otwórz %s ręcznie."
R_RUNNING="Nine Men's Morris działa pod adresem %s"
R_STOP_HINT="Naciśnij Ctrl-C, aby zatrzymać."
