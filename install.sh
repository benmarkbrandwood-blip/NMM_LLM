#!/usr/bin/env bash
# install.sh — Set up Nine Men's Morris (NMM) with Ollama LLM support.
# Tested on Ubuntu/Pop!_OS 22.04+. Run once from the project root.

set -euo pipefail

NMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$NMM_DIR/.venv"
PYTHON="${PYTHON:-python3}"
MODEL="llama3.1:8b"

# ── Detect model from settings.json if present ────────────────────────────────
SETTINGS="$NMM_DIR/data/settings.json"
if command -v jq &>/dev/null && [ -f "$SETTINGS" ]; then
    DETECTED=$(jq -r '.ollama_model // empty' "$SETTINGS" 2>/dev/null)
    [ -n "$DETECTED" ] && MODEL="$DETECTED"
elif [ -f "$SETTINGS" ] && command -v python3 &>/dev/null; then
    DETECTED=$(python3 -c "import json,sys; d=json.load(open('$SETTINGS')); print(d.get('ollama_model',''))" 2>/dev/null)
    [ -n "$DETECTED" ] && MODEL="$DETECTED"
fi

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[NMM]${NC} $*"; }
warn()  { echo -e "${YELLOW}[NMM]${NC} $*"; }
error() { echo -e "${RED}[NMM]${NC} $*" >&2; exit 1; }

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   Nine Men's Morris — Installer      ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── 1. Python 3.10+ ───────────────────────────────────────────────────────────
info "Checking Python..."
$PYTHON --version &>/dev/null || error "Python 3 not found. Install python3 and retry."
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required (found $PY_VER)."
fi
info "Python $PY_VER — OK"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    info "Existing venv found at .venv — skipping creation."
else
    info "Creating virtual environment in .venv ..."
    $PYTHON -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

info "Upgrading pip..."
"$VENV_PIP" install --quiet --upgrade pip

# ── 3. Python dependencies ────────────────────────────────────────────────────
info "Installing Python requirements..."
"$VENV_PIP" install --quiet -r "$NMM_DIR/requirements.txt"
info "Python packages installed."

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    info "Ollama already installed ($(ollama --version 2>/dev/null | head -1))."
else
    info "Installing Ollama..."
    if command -v curl &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        error "curl not found. Install curl and re-run, or install Ollama manually from https://ollama.com"
    fi
    info "Ollama installed."
fi

# ── 5. Start Ollama service (if not running) ──────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    info "Starting Ollama service..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 15 s for it to be ready
    for i in $(seq 1 15); do
        sleep 1
        curl -s http://localhost:11434/api/tags &>/dev/null && break
        [ "$i" -eq 15 ] && error "Ollama service did not start in time."
    done
    info "Ollama service started (pid $OLLAMA_PID)."
else
    info "Ollama service already running."
fi

# ── 6. Pull the LLM model ─────────────────────────────────────────────────────
info "Checking for model '$MODEL'..."
if ollama show "$MODEL" &>/dev/null; then
    info "Model '$MODEL' already present."
else
    info "Pulling '$MODEL' — this may take several minutes..."
    ollama pull "$MODEL"
    info "Model '$MODEL' ready."
fi

# ── 7. Create data directories ────────────────────────────────────────────────
for d in data/games data/session_memory data/chroma; do
    mkdir -p "$NMM_DIR/$d"
done
info "Data directories ready."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "  To start the game:"
echo "    ./run_nmm.sh"
echo ""
