#!/usr/bin/env bash
# install.sh — Set up Nine Men's Morris (NMM) on Linux/macOS.
# Does not require a pre-installed Python — downloads a private CPython 3.12
# via uv (https://github.com/astral-sh/uv).

set -euo pipefail

NMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NMM_DIR

VENV_DIR="$NMM_DIR/.venv"
OS_TYPE="$(uname -s 2>/dev/null || echo unknown)"

# ── Argument parsing ──────────────────────────────────────────────────────────
NO_OLLAMA=0
YES_ALL=0
MODEL=""
REPAIR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-ollama|--noollama) NO_OLLAMA=1 ;;
        --yes|-y)               YES_ALL=1 ;;
        --model)                MODEL="$2"; shift ;;
        --model=*)              MODEL="${1#*=}" ;;
        --repair)               REPAIR=1 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

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

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[NMM]${NC} $*"; }
warn()  { echo -e "${YELLOW}[NMM]${NC} $*"; }
error() { echo -e "${RED}[NMM]${NC} $*" >&2; exit 1; }

echo ""
echo "$I_TITLE_1"
echo "$I_TITLE_2"
echo "$I_TITLE_3"
echo ""
info "Install directory: $NMM_DIR"

# ── Detect model from settings.json if present ────────────────────────────────
SETTINGS="$NMM_DIR/data/settings.json"
if [[ -z "$MODEL" ]]; then
    if command -v jq &>/dev/null && [[ -f "$SETTINGS" ]]; then
        DETECTED=$(jq -r '.ollama_model // empty' "$SETTINGS" 2>/dev/null || true)
        [[ -n "$DETECTED" ]] && MODEL="$DETECTED"
    elif [[ -f "$SETTINGS" ]]; then
        DETECTED=$(python3 -c "import json; d=json.load(open('$SETTINGS')); print(d.get('ollama_model',''))" 2>/dev/null || true)
        [[ -n "$DETECTED" ]] && MODEL="$DETECTED"
    fi
fi
[[ -z "$MODEL" ]] && MODEL="qwen3:4b-instruct"

# ── Repair mode: delete .venv and reinstall ───────────────────────────────────
if [[ "$REPAIR" -eq 1 ]]; then
    info "Repair mode: removing .venv (data/ and runtime/ are preserved)."
    rm -rf "$VENV_DIR"
fi

# ── 1. Bootstrap private Python 3.12 via uv ───────────────────────────────────
# shellcheck disable=SC1091
source "$NMM_DIR/packaging/bootstrap_python.sh"
# After sourcing: UV_CMD, UV_PYTHON_INSTALL_DIR, NMM_PYTHON are exported.

# ── 2. Create virtual environment ────────────────────────────────────────────
VENV_PY="$VENV_DIR/bin/python"

if [[ -x "$VENV_PY" ]]; then
    _VENV_VER="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0)"
    if [[ "$_VENV_VER" == 3.12* ]]; then
        info "Virtual environment already exists (Python $_VENV_VER)."
    else
        info "Virtual environment is Python $_VENV_VER; recreating with 3.12."
        rm -rf "$VENV_DIR"
        "$UV_CMD" venv --python "$NMM_PYTHON" "$VENV_DIR"
    fi
else
    info "Creating virtual environment ..."
    "$UV_CMD" venv --python "$NMM_PYTHON" "$VENV_DIR"
fi

# ── 3. Install Python requirements from locked files ─────────────────────────
info "Installing Python requirements (this may take a few minutes on first run) ..."

"$UV_CMD" pip install \
    --python "$VENV_PY" \
    --require-hashes \
    -r "$NMM_DIR/packaging/requirements-player.lock"

info "Installing CPU torch ..."
"$UV_CMD" pip install \
    --python "$VENV_PY" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --require-hashes \
    -r "$NMM_DIR/packaging/requirements-torch-cpu.lock"

info "Requirements installed."

# ── 4. Ollama (optional) ──────────────────────────────────────────────────────
INSTALL_OLLAMA=0

if [[ "$NO_OLLAMA" -eq 1 ]]; then
    info "Skipping Ollama (--no-ollama). LLM commentary will be disabled."
elif [[ "$YES_ALL" -eq 1 ]]; then
    INSTALL_OLLAMA=1
    info "Installing Ollama automatically (--yes)."
else
    echo ""
    echo "  Nine Men's Morris can use an LLM (${MODEL}) for commentary."
    echo "  Install Ollama now? (Requires ~5 GB download for the model.)"
    echo "  You can also install Ollama later and re-run this script."
    echo ""
    read -rp "  Install Ollama and download ${MODEL}? [Y/n] " _choice
    if [[ -z "$_choice" || "$_choice" =~ ^[Yy] ]]; then
        INSTALL_OLLAMA=1
    else
        info "Skipping Ollama. LLM commentary will be disabled."
    fi
fi

if [[ "$INSTALL_OLLAMA" -eq 1 ]]; then
    if command -v ollama &>/dev/null; then
        info "Ollama already installed: $(ollama --version 2>/dev/null | head -1)"
    else
        info "Downloading and installing Ollama ..."
        if [[ "$OS_TYPE" == "Darwin" ]]; then
            if command -v brew &>/dev/null; then
                brew install ollama
            else
                warn "Homebrew not found. Install Ollama manually from https://ollama.com and re-run."
                INSTALL_OLLAMA=0
            fi
        elif command -v curl &>/dev/null; then
            curl -fsSL https://ollama.com/install.sh | sh
        else
            warn "curl not found. Install Ollama manually from https://ollama.com and re-run."
            INSTALL_OLLAMA=0
        fi
        [[ "$INSTALL_OLLAMA" -eq 1 ]] && info "Ollama installed."
    fi
fi

# ── 5. Start Ollama service and pull model ────────────────────────────────────
if [[ "$INSTALL_OLLAMA" -eq 1 ]]; then
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        info "Starting Ollama service ..."
        ollama serve &>/dev/null &
        _OLLAMA_PID=$!
        for i in $(seq 1 15); do
            sleep 1
            curl -s http://localhost:11434/api/tags &>/dev/null && break
            if [[ "$i" -eq 15 ]]; then
                warn "Ollama service did not start in time. Pull $MODEL manually: ollama pull $MODEL"
                INSTALL_OLLAMA=0
            fi
        done
        [[ "$INSTALL_OLLAMA" -eq 1 ]] && info "Ollama service started (PID $_OLLAMA_PID)."
    else
        info "Ollama service is already running."
    fi

    if [[ "$INSTALL_OLLAMA" -eq 1 ]]; then
        info "Checking model: $MODEL ..."
        if ollama show "$MODEL" &>/dev/null; then
            info "Model $MODEL already present."
        else
            info "Downloading $MODEL (this may take several minutes) ..."
            if ollama pull "$MODEL"; then
                info "Model $MODEL ready."
            else
                warn "Failed to download $MODEL. Run 'ollama pull $MODEL' manually to enable commentary."
            fi
        fi
    fi
fi

# ── 6. Create data directories ────────────────────────────────────────────────
for d in data/games data/ai_games data/human_games data/session_memory \
          data/chroma data/players data/weights data/endgame data/backups \
          data/logs learned_ai/sentinel/checkpoints \
          data/puzzles/endgame data/puzzles/malom data/puzzles/placement data/puzzles/formation; do
    mkdir -p "$NMM_DIR/$d"
done
info "Data directories ready."

# ── 7. Default settings ──────────────────────────────────────────────────────
if [[ ! -f "$NMM_DIR/data/settings.json" && -f "$NMM_DIR/data/settings.template.json" ]]; then
    cp "$NMM_DIR/data/settings.template.json" "$NMM_DIR/data/settings.json"
    info "Created data/settings.json from template."
fi

# ── 8. Download databases ─────────────────────────────────────────────────────
if [[ -f "$NMM_DIR/tools/install_dbs.py" ]]; then
    echo ""
    "$VENV_PY" "$NMM_DIR/tools/install_dbs.py" "$NMM_DIR" || \
        warn "Database installation had errors — game data may be incomplete."
else
    warn "tools/install_dbs.py not found — skipping database download."
fi

# ── Desktop shortcut ─────────────────────────────────────────────────────────
_CREATE_SC=0
if [[ "$YES_ALL" -eq 1 ]]; then
    _CREATE_SC=1
else
    echo ""
    read -rp "  Create a desktop shortcut? [Y/n] " _sc_choice
    if [[ -z "$_sc_choice" || "$_sc_choice" =~ ^[Yy] ]]; then
        _CREATE_SC=1
    fi
fi

if [[ "$_CREATE_SC" -eq 1 ]]; then
    if [[ -f "$NMM_DIR/tools/create_shortcut.sh" ]]; then
        bash "$NMM_DIR/tools/create_shortcut.sh" || warn "Shortcut creation failed — you can retry: bash tools/create_shortcut.sh"
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Installation complete.${NC}"
echo ""
echo "  To start Nine Men's Morris:"
echo "    ./run_nmm.sh"
echo ""
