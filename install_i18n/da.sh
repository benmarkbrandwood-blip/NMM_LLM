#!/usr/bin/env bash
# install_i18n/da.sh — Danish messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Kontrollerer Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 ikke fundet. Installer python3 og prøv igen."
I_PYTHON_OLD="Python 3.10+ påkrævet (fandt: %s)."

I_VENV_EXISTS="Eksisterende venv fundet i .venv — springer oprettelse over."
I_VENV_CREATE="Opretter virtuelt miljø i .venv ..."
I_PIP_UPGRADE="Opgraderer pip..."
I_REQS_INSTALL="Installerer Python-afhængigheder..."
I_REQS_DONE="Python-pakker installeret."

I_OLLAMA_EXISTS="Ollama er allerede installeret (%s)."
I_OLLAMA_INSTALL="Installerer Ollama..."
I_BREW_MISSING="Homebrew ikke fundet. Installer Ollama manuelt fra https://ollama.com/download/mac"
I_BREW_RERUN="Kør derefter dette script igen."
I_OLLAMA_DONE="Ollama installeret."
I_OLLAMA_START="Starter Ollama-tjenesten..."
I_OLLAMA_STARTED="Ollama-tjenesten startet (pid %s)."
I_OLLAMA_RUNNING="Ollama-tjenesten kører allerede."
I_OLLAMA_TIMEOUT="Ollama-tjenesten startede ikke i tide."

I_MODEL_CHECK="Kontrollerer model '%s'..."
I_MODEL_EXISTS="Model '%s' er allerede til stede."
I_MODEL_PULL="Henter '%s' — dette kan tage flere minutter..."
I_MODEL_READY="Model '%s' klar."

I_DIRS_READY="Datamapper klar."
I_SETTINGS_CREATED="Standardindstillinger oprettet i data/settings.json."
I_DB_ERROR="Databaseopsætningen stødte på en fejl — fortsæt via /databases i spillet."
I_DB_MISSING="tools/install_dbs.py ikke fundet — springer databasedownload over."

I_DONE="Installation fuldført!"
I_START_HINT="Sådan starter du spillet:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtuelt miljø ikke fundet. Kør ./install.sh først."
R_NO_UVICORN="uvicorn ikke fundet i venv. Kør ./install.sh først."
R_OLLAMA_START="Ollama kører ikke — starter i baggrunden..."
R_OLLAMA_TIMEOUT="Ollama reagerede ikke — LLM-funktioner vil blive deaktiveret."
R_PORT_BUSY="Port %s er allerede i brug. Forsøger port 8080..."
R_SHUTDOWN="Lukker ned..."
R_NO_CKPT="AI-model-checkpoints ikke fundet — AI-funktioner vil være begrænsede."
R_DL_CKPT="Download dem via ./install.sh eller åbn /databases i spillet."
R_STARTING="Starter Nine Men's Morris på %s ..."
R_WAITING="Venter på at serveren er klar..."
R_SERVER_SLOW="Serveren tog for lang tid at svare — åbner browser alligevel."
R_OPEN_BROWSER="Åbner browser på %s"
R_NO_BROWSER="Kunne ikke registrere en browser. Åbn %s manuelt."
R_RUNNING="Nine Men's Morris kører på %s"
R_STOP_HINT="Tryk Ctrl-C for at stoppe."
