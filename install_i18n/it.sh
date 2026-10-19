#!/usr/bin/env bash
# install_i18n/it.sh — Italian messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Verifica di Python in corso..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 non trovato. Installare python3 e riprovare."
I_PYTHON_OLD="È richiesto Python 3.10+ (trovato: %s)."

I_VENV_EXISTS="Ambiente virtuale esistente trovato in .venv — creazione saltata."
I_VENV_CREATE="Creazione dell'ambiente virtuale in .venv ..."
I_PIP_UPGRADE="Aggiornamento di pip..."
I_REQS_INSTALL="Installazione dei requisiti Python..."
I_REQS_DONE="Pacchetti Python installati."

I_OLLAMA_EXISTS="Ollama è già installato (%s)."
I_OLLAMA_INSTALL="Installazione di Ollama..."
I_BREW_MISSING="Homebrew non trovato. Installare Ollama manualmente da https://ollama.com/download/mac"
I_BREW_RERUN="Poi rieseguire questo script."
I_OLLAMA_DONE="Ollama installato."
I_OLLAMA_START="Avvio del servizio Ollama..."
I_OLLAMA_STARTED="Servizio Ollama avviato (pid %s)."
I_OLLAMA_RUNNING="Il servizio Ollama è già in esecuzione."
I_OLLAMA_TIMEOUT="Il servizio Ollama non si è avviato in tempo."

I_MODEL_CHECK="Verifica del modello '%s' in corso..."
I_MODEL_EXISTS="Il modello '%s' è già presente."
I_MODEL_PULL="Download di '%s' — l'operazione potrebbe richiedere diversi minuti..."
I_MODEL_READY="Modello '%s' pronto."

I_DIRS_READY="Directory dei dati pronte."
I_SETTINGS_CREATED="Impostazioni predefinite create in data/settings.json."
I_DB_ERROR="La configurazione del database ha riscontrato un errore — continuare su /databases nel gioco."
I_DB_MISSING="tools/install_dbs.py non trovato — download del database saltato."

I_DONE="Installazione completata!"
I_START_HINT="Per avviare il gioco:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Ambiente virtuale non trovato. Eseguire prima ./install.sh."
R_NO_UVICORN="uvicorn non trovato nel venv. Eseguire prima ./install.sh."
R_OLLAMA_START="Ollama non è in esecuzione — avvio in background..."
R_OLLAMA_TIMEOUT="Ollama non ha risposto — le funzionalità LLM saranno disabilitate."
R_PORT_BUSY="La porta %s è già in uso. Tentativo sulla porta 8080..."
R_SHUTDOWN="Arresto in corso..."
R_NO_CKPT="Checkpoint del modello IA non trovati — le funzionalità IA saranno limitate."
R_DL_CKPT="Scaricali tramite ./install.sh o apri /databases nel gioco."
R_STARTING="Avvio di Nine Men's Morris su %s ..."
R_WAITING="In attesa che il server sia pronto..."
R_SERVER_SLOW="Il server ha impiegato troppo tempo a rispondere — apertura del browser comunque."
R_OPEN_BROWSER="Apertura del browser su %s"
R_NO_BROWSER="Impossibile rilevare un browser. Aprire %s manualmente."
R_RUNNING="Nine Men's Morris è in esecuzione su %s"
R_STOP_HINT="Premere Ctrl-C per interrompere."
