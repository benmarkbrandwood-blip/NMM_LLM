#!/usr/bin/env bash
# scripts/train_sentinel_v2_step0.sh
#
# Runs Step 0 of docs/retrain_v2_plan.md — trains Sentinel v2 through Stages
# 1, 2, and 4 (Stage 3 archived, Stage 5 dropped per the plan).
#
# Idempotent: skips any stage whose best.pt already exists.  Delete the
# corresponding v2_stage*/ directory to force a rerun of that stage.
#
# After all three stages complete, copies the Stage 4 best checkpoint into
# learned_ai/sentinel/checkpoints/v2/best.pt so downstream steps in the plan
# (gap dataset rebuild, evaluation) can pick it up without further config.
#
# Environment overrides:
#   MALOM_DB=/path/to/malom   — Malom DB directory (Stages 2 and 4)
#   DEVICE=cuda|cpu           — training device (default cpu)
#   GAME_DIR=path             — AI self-play game directory (default data/games)
#   HUMAN_GAME_DIR=path       — human game directory (default data/human_games)
#
# Usage:
#   ./scripts/train_sentinel_v2_step0.sh
#   DEVICE=cuda ./scripts/train_sentinel_v2_step0.sh
#   MALOM_DB=/mnt/D/malom DEVICE=cuda ./scripts/train_sentinel_v2_step0.sh

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"
CKPT_ROOT="learned_ai/sentinel/checkpoints"
GAME_DIR="${GAME_DIR:-data/games}"
HUMAN_GAME_DIR="${HUMAN_GAME_DIR:-data/human_games}"
DEVICE="${DEVICE:-cpu}"

# Malom DB — resolve in this order:
#   1) MALOM_DB env var
#   2) malom_db_path field in data/training_paths.local.json (if present)
#   3) plan default (/mnt/windows/…)
_MALOM_DEFAULT="/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
if [[ -n "${MALOM_DB:-}" ]]; then
    DB_PATH="$MALOM_DB"
elif [[ -f data/training_paths.local.json ]]; then
    DB_PATH="$($PYTHON - <<'PY'
import json, os, sys
try:
    with open("data/training_paths.local.json") as f:
        cfg = json.load(f)
    p = cfg.get("malom_db_path") or ""
    print(p)
except Exception:
    print("")
PY
)"
    [[ -z "$DB_PATH" ]] && DB_PATH="$_MALOM_DEFAULT"
else
    DB_PATH="$_MALOM_DEFAULT"
fi

# ── Prerequisite checks ──────────────────────────────────────────────────────

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found.  Activate or bootstrap .venv first." >&2
    exit 1
fi
if [[ ! -d "$GAME_DIR" ]]; then
    echo "ERROR: game dir $GAME_DIR does not exist." >&2
    exit 1
fi
if [[ ! -d "$DB_PATH" ]]; then
    echo "WARNING: Malom DB path $DB_PATH not found.  Stages 2 and 4 will fail." >&2
    echo "         Set MALOM_DB=/path/to/Std_DD_89adjusted or fix training_paths.local.json." >&2
fi

echo "── Step 0 — Sentinel v2 (Stages 1, 2, 4) ──"
echo "  game dir       : $GAME_DIR"
echo "  human dir      : $HUMAN_GAME_DIR"
echo "  malom db       : $DB_PATH"
echo "  device         : $DEVICE"
echo "  checkpoint root: $CKPT_ROOT"
echo

# ── Stage runner ─────────────────────────────────────────────────────────────

_run_stage() {
    local stage="$1"       # 1, 2, 4
    local config="$2"      # yaml config path
    local out_dir="$3"     # checkpoint output directory
    local resume="$4"      # empty for stage 1; upstream best.pt for stages 2/4
    local use_db="$5"      # "yes" for stages 2 and 4

    local best="$out_dir/best.pt"
    if [[ -f "$best" ]]; then
        echo "── Stage $stage — already complete at $best (skipping) ──"
        return 0
    fi

    if [[ -n "$resume" && ! -f "$resume" ]]; then
        echo "ERROR: Stage $stage requires $resume but it does not exist." >&2
        exit 2
    fi

    echo "── Stage $stage — training → $out_dir ──"
    local cmd=(
        "$PYTHON" scripts/train_sentinel.py
        --config "$config"
        --game-dir "$GAME_DIR"
        --human-game-dir "$HUMAN_GAME_DIR"
        --drop-db-features
        --out-dir "$out_dir"
        --device "$DEVICE"
    )
    if [[ "$use_db" == "yes" ]]; then
        cmd+=(--db-path "$DB_PATH")
    fi
    if [[ -n "$resume" ]]; then
        cmd+=(--resume "$resume")
    fi
    echo "  ${cmd[*]}"
    "${cmd[@]}"

    if [[ ! -f "$best" ]]; then
        echo "ERROR: Stage $stage finished but $best was not produced." >&2
        exit 3
    fi
    echo "── Stage $stage — complete ($best) ──"
    echo
}

# ── Run the three stages ─────────────────────────────────────────────────────

_run_stage 1 configs/sentinel_stage1.yaml "$CKPT_ROOT/v2_stage1" "" no
_run_stage 2 configs/sentinel_stage2.yaml "$CKPT_ROOT/v2_stage2" "$CKPT_ROOT/v2_stage1/best.pt" yes
_run_stage 4 configs/sentinel_stage4.yaml "$CKPT_ROOT/v2_stage4" "$CKPT_ROOT/v2_stage2/best.pt" yes

# ── Promote Stage 4 → v2 slot (per plan, NOT into production best.pt) ───────

mkdir -p "$CKPT_ROOT/v2"
cp "$CKPT_ROOT/v2_stage4/best.pt" "$CKPT_ROOT/v2/best.pt"
echo "── Promoted $CKPT_ROOT/v2_stage4/best.pt → $CKPT_ROOT/v2/best.pt ──"
echo
echo "Step 0 complete."
echo "Next: Step 1 — regenerate GapNet training dataset with the new Sentinel v2 checkpoint."
echo "Production best.pt remains untouched at $CKPT_ROOT/best.pt (v1) until Step 8 promotion."
