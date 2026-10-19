#!/usr/bin/env bash
# install_i18n/cs.sh — Czech messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Kontrola Pythonu..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 nebyl nalezen. Nainstalujte python3 a zkuste to znovu."
I_PYTHON_OLD="Vyžaduje se Python 3.10+ (nalezeno: %s)."

I_VENV_EXISTS="Nalezeno existující venv v .venv — vytváření se přeskakuje."
I_VENV_CREATE="Vytváření virtuálního prostředí v .venv ..."
I_PIP_UPGRADE="Aktualizace pip..."
I_REQS_INSTALL="Instalace závislostí Pythonu..."
I_REQS_DONE="Balíčky Pythonu nainstalovány."

I_OLLAMA_EXISTS="Ollama je již nainstalován (%s)."
I_OLLAMA_INSTALL="Instalace Ollama..."
I_BREW_MISSING="Homebrew nebyl nalezen. Nainstalujte Ollama ručně z https://ollama.com/download/mac"
I_BREW_RERUN="Poté znovu spusťte tento skript."
I_OLLAMA_DONE="Ollama nainstalován."
I_OLLAMA_START="Spouštění služby Ollama..."
I_OLLAMA_STARTED="Služba Ollama spuštěna (pid %s)."
I_OLLAMA_RUNNING="Služba Ollama již běží."
I_OLLAMA_TIMEOUT="Služba Ollama se nespustila včas."

I_MODEL_CHECK="Kontrola modelu '%s'..."
I_MODEL_EXISTS="Model '%s' je již přítomen."
I_MODEL_PULL="Stahování '%s' — může to trvat několik minut..."
I_MODEL_READY="Model '%s' připraven."

I_DIRS_READY="Datové adresáře připraveny."
I_SETTINGS_CREATED="Výchozí nastavení vytvořeno v data/settings.json."
I_DB_ERROR="Nastavení databáze narazilo na chybu — pokračujte přes /databases ve hře."
I_DB_MISSING="tools/install_dbs.py nebyl nalezen — stahování databáze se přeskakuje."

I_DONE="Instalace dokončena!"
I_START_HINT="Spuštění hry:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtuální prostředí nenalezeno. Nejprve spusťte ./install.sh."
R_NO_UVICORN="uvicorn nebyl nalezen ve venv. Nejprve spusťte ./install.sh."
R_OLLAMA_START="Ollama neběží — spouštění na pozadí..."
R_OLLAMA_TIMEOUT="Ollama neodpověděl — funkce LLM budou vypnuty."
R_PORT_BUSY="Port %s je již obsazen. Zkouší se port 8080..."
R_SHUTDOWN="Vypínání..."
R_NO_CKPT="Kontrolní body modelu AI nebyly nalezeny — funkce AI budou omezeny."
R_DL_CKPT="Stáhněte je přes ./install.sh nebo otevřete /databases ve hře."
R_STARTING="Spouštění Nine Men's Morris na %s ..."
R_WAITING="Čekání na připravenost serveru..."
R_SERVER_SLOW="Server příliš dlouho neodpovídal — prohlížeč se otevře i tak."
R_OPEN_BROWSER="Otevírání prohlížeče na %s"
R_NO_BROWSER="Nepodařilo se zjistit prohlížeč. Otevřete %s ručně."
R_RUNNING="Nine Men's Morris běží na %s"
R_STOP_HINT="Pro zastavení stiskněte Ctrl-C."
