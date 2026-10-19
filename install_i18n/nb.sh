#!/usr/bin/env bash
# install_i18n/nb.sh — Norwegian Bokmål messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Sjekker Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 ikke funnet. Installer python3 og prøv igjen."
I_PYTHON_OLD="Python 3.10+ kreves (fant: %s)."

I_VENV_EXISTS="Eksisterende venv funnet i .venv — hopper over oppretting."
I_VENV_CREATE="Oppretter virtuelt miljø i .venv ..."
I_PIP_UPGRADE="Oppgraderer pip..."
I_REQS_INSTALL="Installerer Python-avhengigheter..."
I_REQS_DONE="Python-pakker installert."

I_OLLAMA_EXISTS="Ollama er allerede installert (%s)."
I_OLLAMA_INSTALL="Installerer Ollama..."
I_BREW_MISSING="Homebrew ikke funnet. Installer Ollama manuelt fra https://ollama.com/download/mac"
I_BREW_RERUN="Kjør deretter dette skriptet på nytt."
I_OLLAMA_DONE="Ollama installert."
I_OLLAMA_START="Starter Ollama-tjenesten..."
I_OLLAMA_STARTED="Ollama-tjenesten startet (pid %s)."
I_OLLAMA_RUNNING="Ollama-tjenesten kjører allerede."
I_OLLAMA_TIMEOUT="Ollama-tjenesten startet ikke i tide."

I_MODEL_CHECK="Sjekker modell '%s'..."
I_MODEL_EXISTS="Modell '%s' er allerede til stede."
I_MODEL_PULL="Laster ned '%s' — dette kan ta flere minutter..."
I_MODEL_READY="Modell '%s' klar."

I_DIRS_READY="Datamapper klare."
I_SETTINGS_CREATED="Standardinnstillinger opprettet i data/settings.json."
I_DB_ERROR="Databaseoppsett støtte på en feil — fortsett via /databases i spillet."
I_DB_MISSING="tools/install_dbs.py ikke funnet — hopper over databasenedlasting."

I_DONE="Installasjonen er fullført!"
I_START_HINT="Slik starter du spillet:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtuelt miljø ikke funnet. Kjør ./install.sh først."
R_NO_UVICORN="uvicorn ikke funnet i venv. Kjør ./install.sh først."
R_OLLAMA_START="Ollama kjører ikke — starter i bakgrunnen..."
R_OLLAMA_TIMEOUT="Ollama svarte ikke — LLM-funksjoner vil bli deaktivert."
R_PORT_BUSY="Port %s er allerede i bruk. Prøver port 8080..."
R_SHUTDOWN="Avslutter..."
R_NO_CKPT="AI-modellens kontrollpunkter ikke funnet — AI-funksjoner vil være begrenset."
R_DL_CKPT="Last dem ned via ./install.sh eller åpne /databases i spillet."
R_STARTING="Starter Nine Men's Morris på %s ..."
R_WAITING="Venter på at serveren er klar..."
R_SERVER_SLOW="Serveren brukte for lang tid på å svare — åpner nettleseren likevel."
R_OPEN_BROWSER="Åpner nettleser på %s"
R_NO_BROWSER="Kunne ikke oppdage en nettleser. Åpne %s manuelt."
R_RUNNING="Nine Men's Morris kjører på %s"
R_STOP_HINT="Trykk Ctrl-C for å stoppe."
