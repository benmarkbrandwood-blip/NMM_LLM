#!/usr/bin/env bash
# install_i18n/de.sh — German messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Python wird geprüft..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 nicht gefunden. Bitte python3 installieren und erneut versuchen."
I_PYTHON_OLD="Python 3.10+ erforderlich (gefunden: %s)."

I_VENV_EXISTS="Vorhandene venv in .venv gefunden — Erstellung wird übersprungen."
I_VENV_CREATE="Virtuelle Umgebung in .venv wird erstellt ..."
I_PIP_UPGRADE="pip wird aktualisiert..."
I_REQS_INSTALL="Python-Abhängigkeiten werden installiert..."
I_REQS_DONE="Python-Pakete installiert."

I_OLLAMA_EXISTS="Ollama ist bereits installiert (%s)."
I_OLLAMA_INSTALL="Ollama wird installiert..."
I_BREW_MISSING="Homebrew nicht gefunden. Ollama manuell installieren unter https://ollama.com/download/mac"
I_BREW_RERUN="Dann dieses Skript erneut ausführen."
I_OLLAMA_DONE="Ollama installiert."
I_OLLAMA_START="Ollama-Dienst wird gestartet..."
I_OLLAMA_STARTED="Ollama-Dienst gestartet (pid %s)."
I_OLLAMA_RUNNING="Ollama-Dienst läuft bereits."
I_OLLAMA_TIMEOUT="Ollama-Dienst hat nicht rechtzeitig gestartet."

I_MODEL_CHECK="Modell '%s' wird geprüft..."
I_MODEL_EXISTS="Modell '%s' bereits vorhanden."
I_MODEL_PULL="'%s' wird geladen — dies kann einige Minuten dauern..."
I_MODEL_READY="Modell '%s' bereit."

I_DIRS_READY="Datenverzeichnisse bereit."
I_SETTINGS_CREATED="Standardeinstellungen erstellt unter data/settings.json."
I_DB_ERROR="Datenbankeinrichtung ist fehlgeschlagen — weiter unter /databases im Spiel."
I_DB_MISSING="tools/install_dbs.py nicht gefunden — Datenbankdownload wird übersprungen."

I_DONE="Installation abgeschlossen!"
I_START_HINT="So starten Sie das Spiel:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtuelle Umgebung nicht gefunden. Zuerst ./install.sh ausführen."
R_NO_UVICORN="uvicorn nicht in venv gefunden. Zuerst ./install.sh ausführen."
R_OLLAMA_START="Ollama läuft nicht — wird im Hintergrund gestartet..."
R_OLLAMA_TIMEOUT="Ollama hat nicht geantwortet — LLM-Funktionen werden deaktiviert."
R_PORT_BUSY="Port %s wird bereits verwendet. Port 8080 wird versucht..."
R_SHUTDOWN="Wird beendet..."
R_NO_CKPT="KI-Modell-Checkpoints nicht gefunden — KI-Funktionen werden eingeschränkt sein."
R_DL_CKPT="Herunterladen über ./install.sh oder /databases im Spiel öffnen."
R_STARTING="Nine Men's Morris wird gestartet unter %s ..."
R_WAITING="Warten auf Server-Bereitschaft..."
R_SERVER_SLOW="Server hat zu lange nicht geantwortet — Browser wird trotzdem geöffnet."
R_OPEN_BROWSER="Browser wird geöffnet unter %s"
R_NO_BROWSER="Kein Browser-Öffner gefunden. Bitte %s manuell öffnen."
R_RUNNING="Nine Men's Morris läuft unter %s"
R_STOP_HINT="Strg-C zum Beenden drücken."
