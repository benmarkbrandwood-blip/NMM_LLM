#!/usr/bin/env bash
# install_i18n/sv.sh — Swedish messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Kontrollerar Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 hittades inte. Installera python3 och försök igen."
I_PYTHON_OLD="Python 3.10+ krävs (hittade: %s)."

I_VENV_EXISTS="Befintlig venv hittad i .venv — hoppar över skapande."
I_VENV_CREATE="Skapar virtuell miljö i .venv ..."
I_PIP_UPGRADE="Uppgraderar pip..."
I_REQS_INSTALL="Installerar Python-beroenden..."
I_REQS_DONE="Python-paket installerade."

I_OLLAMA_EXISTS="Ollama är redan installerat (%s)."
I_OLLAMA_INSTALL="Installerar Ollama..."
I_BREW_MISSING="Homebrew hittades inte. Installera Ollama manuellt från https://ollama.com/download/mac"
I_BREW_RERUN="Kör sedan om det här skriptet."
I_OLLAMA_DONE="Ollama installerat."
I_OLLAMA_START="Startar Ollama-tjänsten..."
I_OLLAMA_STARTED="Ollama-tjänsten startad (pid %s)."
I_OLLAMA_RUNNING="Ollama-tjänsten körs redan."
I_OLLAMA_TIMEOUT="Ollama-tjänsten startade inte i tid."

I_MODEL_CHECK="Kontrollerar modell '%s'..."
I_MODEL_EXISTS="Modell '%s' finns redan."
I_MODEL_PULL="Laddar ner '%s' — detta kan ta flera minuter..."
I_MODEL_READY="Modell '%s' redo."

I_DIRS_READY="Datamappar redo."
I_SETTINGS_CREATED="Standardinställningar skapade i data/settings.json."
I_DB_ERROR="Databasinstallationen stötte på ett fel — fortsätt via /databases i spelet."
I_DB_MISSING="tools/install_dbs.py hittades inte — hoppar över databasnedladdning."

I_DONE="Installationen slutförd!"
I_START_HINT="Så här startar du spelet:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtuell miljö hittades inte. Kör ./install.sh först."
R_NO_UVICORN="uvicorn hittades inte i venv. Kör ./install.sh först."
R_OLLAMA_START="Ollama körs inte — startar i bakgrunden..."
R_OLLAMA_TIMEOUT="Ollama svarade inte — LLM-funktioner kommer att inaktiveras."
R_PORT_BUSY="Port %s används redan. Försöker med port 8080..."
R_SHUTDOWN="Stänger av..."
R_NO_CKPT="AI-modellens kontrollpunkter hittades inte — AI-funktioner kommer att vara begränsade."
R_DL_CKPT="Ladda ner dem via ./install.sh eller öppna /databases i spelet."
R_STARTING="Startar Nine Men's Morris på %s ..."
R_WAITING="Väntar på att servern ska bli redo..."
R_SERVER_SLOW="Servern tog för lång tid att svara — öppnar webbläsaren ändå."
R_OPEN_BROWSER="Öppnar webbläsare på %s"
R_NO_BROWSER="Kunde inte hitta ett webbläsarprogram. Öppna %s manuellt."
R_RUNNING="Nine Men's Morris körs på %s"
R_STOP_HINT="Tryck Ctrl-C för att stoppa."
