#!/usr/bin/env bash
# install_i18n/nl.sh — Dutch messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Python controleren..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 niet gevonden. Installeer python3 en probeer opnieuw."
I_PYTHON_OLD="Python 3.10+ vereist (gevonden: %s)."

I_VENV_EXISTS="Bestaande venv gevonden in .venv — aanmaken overgeslagen."
I_VENV_CREATE="Virtuele omgeving aanmaken in .venv ..."
I_PIP_UPGRADE="pip bijwerken..."
I_REQS_INSTALL="Python-vereisten installeren..."
I_REQS_DONE="Python-pakketten geïnstalleerd."

I_OLLAMA_EXISTS="Ollama is al geïnstalleerd (%s)."
I_OLLAMA_INSTALL="Ollama installeren..."
I_BREW_MISSING="Homebrew niet gevonden. Installeer Ollama handmatig via https://ollama.com/download/mac"
I_BREW_RERUN="Voer daarna dit script opnieuw uit."
I_OLLAMA_DONE="Ollama geïnstalleerd."
I_OLLAMA_START="Ollama-service starten..."
I_OLLAMA_STARTED="Ollama-service gestart (pid %s)."
I_OLLAMA_RUNNING="Ollama-service is al actief."
I_OLLAMA_TIMEOUT="Ollama-service is niet op tijd gestart."

I_MODEL_CHECK="Model '%s' controleren..."
I_MODEL_EXISTS="Model '%s' is al aanwezig."
I_MODEL_PULL="'%s' downloaden — dit kan enkele minuten duren..."
I_MODEL_READY="Model '%s' gereed."

I_DIRS_READY="Gegevensmappen gereed."
I_SETTINGS_CREATED="Standaardinstellingen aangemaakt in data/settings.json."
I_DB_ERROR="Database-installatie heeft een fout ondervonden — ga verder via /databases in het spel."
I_DB_MISSING="tools/install_dbs.py niet gevonden — database-download overgeslagen."

I_DONE="Installatie voltooid!"
I_START_HINT="Het spel starten:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtuele omgeving niet gevonden. Voer eerst ./install.sh uit."
R_NO_UVICORN="uvicorn niet gevonden in venv. Voer eerst ./install.sh uit."
R_OLLAMA_START="Ollama draait niet — wordt op de achtergrond gestart..."
R_OLLAMA_TIMEOUT="Ollama reageerde niet — LLM-functies worden uitgeschakeld."
R_PORT_BUSY="Poort %s is al in gebruik. Poort 8080 wordt geprobeerd..."
R_SHUTDOWN="Afsluiten..."
R_NO_CKPT="AI-model-checkpoints niet gevonden — AI-functies zullen beperkt zijn."
R_DL_CKPT="Download ze via ./install.sh of open /databases in het spel."
R_STARTING="Nine Men's Morris starten op %s ..."
R_WAITING="Wachten totdat de server gereed is..."
R_SERVER_SLOW="De server reageerde te traag — browser wordt toch geopend."
R_OPEN_BROWSER="Browser openen op %s"
R_NO_BROWSER="Kon geen browser-opener detecteren. Open %s handmatig."
R_RUNNING="Nine Men's Morris is actief op %s"
R_STOP_HINT="Druk op Ctrl-C om te stoppen."
