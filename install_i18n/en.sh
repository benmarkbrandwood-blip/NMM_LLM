#!/usr/bin/env bash
# install_i18n/en.sh — English messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Checking Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 not found. Install python3 and retry."
I_PYTHON_OLD="Python 3.10+ required (found %s)."

I_VENV_EXISTS="Existing venv found at .venv — skipping creation."
I_VENV_CREATE="Creating virtual environment in .venv ..."
I_PIP_UPGRADE="Upgrading pip..."
I_REQS_INSTALL="Installing Python requirements..."
I_REQS_DONE="Python packages installed."

I_OLLAMA_EXISTS="Ollama already installed (%s)."
I_OLLAMA_INSTALL="Installing Ollama..."
I_BREW_MISSING="Homebrew not found. Install Ollama manually from https://ollama.com/download/mac"
I_BREW_RERUN="Then re-run this script."
I_OLLAMA_DONE="Ollama installed."
I_OLLAMA_START="Starting Ollama service..."
I_OLLAMA_STARTED="Ollama service started (pid %s)."
I_OLLAMA_RUNNING="Ollama service already running."
I_OLLAMA_TIMEOUT="Ollama service did not start in time."

I_MODEL_CHECK="Checking for model '%s'..."
I_MODEL_EXISTS="Model '%s' already present."
I_MODEL_PULL="Pulling '%s' — this may take several minutes..."
I_MODEL_READY="Model '%s' ready."

I_DIRS_READY="Data directories ready."
I_SETTINGS_CREATED="Default settings created at data/settings.json."
I_DB_ERROR="Database setup encountered an error — continue at /databases in the game."
I_DB_MISSING="tools/install_dbs.py not found — skipping database download."

I_DONE="Installation complete!"
I_START_HINT="To start the game:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Virtual environment not found. Run ./install.sh first."
R_NO_UVICORN="uvicorn not found in venv. Run ./install.sh first."
R_OLLAMA_START="Ollama not running — starting it in the background..."
R_OLLAMA_TIMEOUT="Ollama did not respond — LLM features will be disabled."
R_PORT_BUSY="Port %s is already in use. Trying port 8080..."
R_SHUTDOWN="Shutting down..."
R_NO_CKPT="AI model checkpoints not found — AI features will be limited."
R_DL_CKPT="Download them via ./install.sh or open /databases in the game."
R_STARTING="Starting Nine Men's Morris at %s ..."
R_WAITING="Waiting for server to be ready..."
R_SERVER_SLOW="Server took too long to respond — opening browser anyway."
R_OPEN_BROWSER="Opening browser at %s"
R_NO_BROWSER="Could not detect a browser opener. Open %s manually."
R_RUNNING="Nine Men's Morris is running at %s"
R_STOP_HINT="Press Ctrl-C to stop."
