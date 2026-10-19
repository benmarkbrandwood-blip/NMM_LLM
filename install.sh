#!/usr/bin/env bash
# install.sh — Set up Nine Men's Morris (NMM) with Ollama LLM support.
# Tested on Ubuntu/Pop!_OS 22.04+. Run once from the project root.

set -euo pipefail

NMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$NMM_DIR/.venv"
PYTHON="${PYTHON:-python3}"
MODEL="llama3.1:8b"
OS_TYPE="$(uname -s 2>/dev/null || echo unknown)"

# ── Locale detection + message file ──────────────────────────────────────────
_lang="${LANG:-en}"
case "$_lang" in
    zh*) _L="zh" ;; de*) _L="de" ;; hu*) _L="hu" ;; fa*) _L="fa" ;;
    ar*) _L="ar" ;; ru*) _L="ru" ;; ro*) _L="ro" ;; it*) _L="it" ;;
    fr*) _L="fr" ;; pl*) _L="pl" ;; cs*) _L="cs" ;; es*) _L="es" ;;
    da*) _L="da" ;; nb*|no*) _L="nb" ;; sv*) _L="sv" ;; nl*) _L="nl" ;;
    pt*) _L="pt" ;; *)  _L="en" ;;
esac
_I18N="$NMM_DIR/install_i18n/${_L}.sh"
# shellcheck disable=SC1090
if [ -f "$_I18N" ]; then source "$_I18N"; else source "$NMM_DIR/install_i18n/en.sh"; fi

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
echo "$I_TITLE_1"
echo "$I_TITLE_2"
echo "$I_TITLE_3"
echo ""

# ── 1. Python 3.10+ ───────────────────────────────────────────────────────────
info "$I_PYTHON_CHECK"
$PYTHON --version &>/dev/null || error "$I_PYTHON_MISSING"
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "$(printf "$I_PYTHON_OLD" "$PY_VER")"
fi
info "$(printf "$I_PYTHON_OK" "$PY_VER")"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    info "$I_VENV_EXISTS"
else
    info "$I_VENV_CREATE"
    $PYTHON -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

info "$I_PIP_UPGRADE"
"$VENV_PIP" install --quiet --upgrade pip

# ── 3. Python dependencies ────────────────────────────────────────────────────
info "$I_REQS_INSTALL"
"$VENV_PIP" install --quiet -r "$NMM_DIR/requirements.txt"
info "$I_REQS_DONE"

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    info "$(printf "$I_OLLAMA_EXISTS" "$(ollama --version 2>/dev/null | head -1)")"
else
    info "$I_OLLAMA_INSTALL"
    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS: official installer script doesn't support Mac; use Homebrew
        if command -v brew &>/dev/null; then
            brew install ollama
        else
            warn "$I_BREW_MISSING"
            warn "$I_BREW_RERUN"
            exit 1
        fi
    elif command -v curl &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        error "curl not found. Install curl and re-run, or install Ollama manually from https://ollama.com"
    fi
    info "$I_OLLAMA_DONE"
fi

# ── 5. Start Ollama service (if not running) ──────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    info "$I_OLLAMA_START"
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 15 s for it to be ready
    for i in $(seq 1 15); do
        sleep 1
        curl -s http://localhost:11434/api/tags &>/dev/null && break
        [ "$i" -eq 15 ] && error "$I_OLLAMA_TIMEOUT"
    done
    info "$(printf "$I_OLLAMA_STARTED" "$OLLAMA_PID")"
else
    info "$I_OLLAMA_RUNNING"
fi

# ── 6. Pull the LLM model ─────────────────────────────────────────────────────
info "$(printf "$I_MODEL_CHECK" "$MODEL")"
if ollama show "$MODEL" &>/dev/null; then
    info "$(printf "$I_MODEL_EXISTS" "$MODEL")"
else
    info "$(printf "$I_MODEL_PULL" "$MODEL")"
    ollama pull "$MODEL"
    info "$(printf "$I_MODEL_READY" "$MODEL")"
fi

# ── 7. Create data directories ────────────────────────────────────────────────
for d in data/games data/ai_games data/human_games data/session_memory \
          data/chroma data/players data/weights data/endgame data/backups \
          data/logs learned_ai/sentinel/checkpoints; do
    mkdir -p "$NMM_DIR/$d"
done
info "$I_DIRS_READY"

# ── 8. Default settings ──────────────────────────────────────────────────────
if [ ! -f "$NMM_DIR/data/settings.json" ] && [ -f "$NMM_DIR/data/settings.template.json" ]; then
    cp "$NMM_DIR/data/settings.template.json" "$NMM_DIR/data/settings.json"
    info "$I_SETTINGS_CREATED"
fi

# ── 9. Download databases ─────────────────────────────────────────────────────
if [ -f "$NMM_DIR/tools/install_dbs.py" ]; then
    echo ""
    "$VENV_PY" "$NMM_DIR/tools/install_dbs.py" "$NMM_DIR" || \
        warn "$I_DB_ERROR"
else
    warn "$I_DB_MISSING"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${I_DONE}${NC}"
echo ""
echo "  $I_START_HINT"
echo "$I_START_CMD"
echo ""
