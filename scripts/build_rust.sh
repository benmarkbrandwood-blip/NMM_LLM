#!/usr/bin/env bash
# scripts/build_rust.sh — Optional: build the Rust acceleration core (nmm_core).
#
# This is NOT required to run the game. The Python engine works standalone and
# transparently falls back when nmm_core is absent (see ai/native_core.py).
# Building this extension accelerates the hot-path engine primitives + search.
#
# Usage:
#   ./scripts/build_rust.sh            # build + install into ./.venv if present
#   PYTHON=python3.11 ./scripts/build_rust.sh
#
# Requirements: a Rust toolchain (rustc/cargo). If absent, this script prints
# install instructions and exits non-fatally (exit 0) so installers can call it
# without aborting the whole setup.

set -uo pipefail

NMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$NMM_DIR/native/nmm_core"
VENV_DIR="$NMM_DIR/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}[rust]${NC} $*"; }
warn() { echo -e "${YELLOW}[rust]${NC} $*"; }

# ── Pick the interpreter / pip (prefer the project venv) ──────────────────────
if [ -x "$VENV_DIR/bin/python" ]; then
    PY="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
    info "Using project venv: $VENV_DIR"
else
    PY="${PYTHON:-python3}"
    PIP="$PY -m pip"
    warn "No .venv found — using system interpreter ($PY)."
fi

# ── Rust toolchain check (non-fatal) ──────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
fi
if ! command -v cargo &>/dev/null; then
    warn "Rust toolchain not found. The game runs fine WITHOUT it (Python fallback)."
    warn "To enable acceleration, install Rust:"
    warn "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    warn "Then re-run: ./scripts/build_rust.sh"
    exit 0
fi
info "cargo $(cargo --version | awk '{print $2}') found."

# ── maturin ───────────────────────────────────────────────────────────────────
if ! $PY -c "import maturin" &>/dev/null; then
    info "Installing maturin build backend..."
    $PIP install "maturin>=1.4,<2.0" || { warn "maturin install failed; skipping Rust build."; exit 0; }
fi

# ── Build + install ───────────────────────────────────────────────────────────
cd "$CRATE_DIR"
if [ -n "${VIRTUAL_ENV:-}" ] || [ -x "$VENV_DIR/bin/python" ]; then
    # maturin develop needs an activated venv; activate the project venv if present.
    if [ -x "$VENV_DIR/bin/activate" ] || [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
    info "Building + installing nmm_core (maturin develop --release)..."
    if maturin develop --release; then
        info "nmm_core installed into the venv."
    else
        warn "maturin develop failed; falling back to wheel build + pip install."
        maturin build --release && $PIP install --force-reinstall target/wheels/nmm_core-*.whl
    fi
else
    info "Building wheel (no venv) and pip-installing..."
    maturin build --release || { warn "Rust build failed; Python fallback remains active."; exit 0; }
    $PIP install --force-reinstall target/wheels/nmm_core-*.whl || warn "wheel install failed; Python fallback remains active."
fi

# ── Verify ────────────────────────────────────────────────────────────────────
if $PY -c "import nmm_core" &>/dev/null; then
    info "Verified: 'import nmm_core' works. Acceleration enabled."
else
    warn "nmm_core not importable after build; the game will use the Python fallback."
fi
