#!/usr/bin/env bash
# run_nmm.sh — Launch Nine Men's Morris web server and open the browser.

set -euo pipefail

NMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$NMM_DIR/.venv/bin/python"
VENV_UV="$NMM_DIR/.venv/bin/uvicorn"
HOST="127.0.0.1"
PORT="8000"
URL="http://$HOST:$PORT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[NMM]${NC} $*"; }
warn()  { echo -e "${YELLOW}[NMM]${NC} $*"; }
error() { echo -e "${RED}[NMM]${NC} $*" >&2; exit 1; }

# ── Locale detection + message file ──────────────────────────────────────────
_lang="${LANG:-en}"
case "$_lang" in
    zh*) _L="zh" ;; de*) _L="de" ;; hu*) _L="hu" ;; fa*) _L="fa" ;;
    ar*) _L="ar" ;; ru*) _L="ru" ;; ro*) _L="ro" ;; it*) _L="it" ;;
    fr*) _L="fr" ;; pl*) _L="pl" ;; cs*) _L="cs" ;; es*) _L="es" ;;
    da*) _L="da" ;; nb*|no*) _L="nb" ;; sv*) _L="sv" ;; nl*) _L="nl" ;;
    pt*) _L="pt" ;; *) _L="en" ;;
esac
_I18N="$NMM_DIR/install_i18n/${_L}.sh"
# shellcheck disable=SC1090
if [ -f "$_I18N" ]; then source "$_I18N"; else source "$NMM_DIR/install_i18n/en.sh"; fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[ -f "$VENV_PY" ]  || error "$R_NO_VENV"
[ -f "$VENV_UV" ]  || error "$R_NO_UVICORN"

# ── Ensure Ollama is running ──────────────────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    warn "$R_OLLAMA_START"
    ollama serve &>/dev/null &
    for i in $(seq 1 15); do
        sleep 1
        curl -s http://localhost:11434/api/tags &>/dev/null && break
        [ "$i" -eq 15 ] && warn "$R_OLLAMA_TIMEOUT"
    done
fi

# ── Check port availability ───────────────────────────────────────────────────
if lsof -i :"$PORT" &>/dev/null 2>&1; then
    warn "$(printf "$R_PORT_BUSY" "$PORT")"
    PORT="8080"
    URL="http://$HOST:$PORT"
fi

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    info "$R_SHUTDOWN"
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Pre-flight: AI model checkpoints ─────────────────────────────────────────
CKPT_GEN="$NMM_DIR/learned_ai/checkpoints/scaffolded/s_gen_v5/from_v4/best.pt"
CKPT_SEN="$NMM_DIR/learned_ai/sentinel/checkpoints/best.pt"
if [ ! -f "$CKPT_GEN" ] || [ ! -f "$CKPT_SEN" ]; then
    warn "$R_NO_CKPT"
    warn "$R_DL_CKPT"
fi

# ── Launch server ─────────────────────────────────────────────────────────────
info "$(printf "$R_STARTING" "$URL")"
cd "$NMM_DIR"
"$VENV_UV" web.app:app --host "$HOST" --port "$PORT" \
    --ws-ping-interval 60 --ws-ping-timeout 120 &
SERVER_PID=$!

# Wait for the server to be ready (poll /api/ping; -f fails on HTTP error codes)
info "$R_WAITING"
for i in $(seq 1 60); do
    sleep 0.5
    curl -sf "$URL/api/ping" &>/dev/null && break
    [ "$i" -eq 60 ] && warn "$R_SERVER_SLOW"
done

# ── Open browser ──────────────────────────────────────────────────────────────
info "$(printf "$R_OPEN_BROWSER" "$URL")"
if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" &>/dev/null &
elif command -v open &>/dev/null; then
    open "$URL" &
elif command -v wslview &>/dev/null; then
    wslview "$URL" &
else
    warn "$(printf "$R_NO_BROWSER" "$URL")"
fi

echo ""
echo -e "${GREEN}$(printf "$R_RUNNING" "$URL")${NC}"
echo "  $R_STOP_HINT"
echo ""

# ── Keep running until interrupted ───────────────────────────────────────────
wait "$SERVER_PID"
