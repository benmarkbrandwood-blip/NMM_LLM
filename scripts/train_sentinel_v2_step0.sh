#!/usr/bin/env bash
# scripts/train_sentinel_v2_step0.sh
#
# Sentinel v2 — Step 0 of docs/retrain_v2_plan.md, retooled to consume a
# **combined JSONL + Malom-sampled** dataset for all three stages:
#
#   * 60% Malom-sampled MoveExamples (equal parts placement / midgame / fly).
#     Positions must contain at least one W-or-L legal move (not all-draw).
#   * 40% classic JSONL-replay MoveExamples from data/games + data/human_games.
#   * Total examples doubled vs the plain JSONL pipeline (`--total-examples`
#     defaults to 4M).
#
# The dataset is built once by scripts/build_sentinel_dataset_v2.py, then
# reused across Stages 1, 2, 4 via `train_sentinel.py --dataset PATH`.  The
# per-stage YAML still controls model shape, LR, and per-stage flags like
# `--drop-db-features`.
#
# Idempotent: skips the dataset build if the file already exists, and skips
# any stage whose best.pt is present.  Delete the dataset or a v2_stage{N}/
# directory to force a rerun.
#
# Environment overrides:
#   MALOM_DB=/path/to/malom        Malom DB directory (Stages 2, 4, and Malom sampling)
#   DEVICE=cuda|cpu                training device (default cpu)
#   GAME_DIR=path                  AI self-play game directory
#   HUMAN_GAME_DIR=path            human game directory
#   PATIENCE=N                     override --patience for every stage
#   DATASET_TOTAL_EXAMPLES=N       override --total-examples (default 4_000_000)
#   DATASET_MALOM_FRACTION=F       override --malom-fraction (default 0.60)
#   DATASET_PATH=path              override output/consumed dataset path
#   REBUILD_DATASET=1              force dataset rebuild even if the file exists
#   RUN_STAGE1=1                   include Stage 1 (skipped by default because
#                                  the structural warm-start empirically
#                                  anchors the trunk on heuristic labels and
#                                  Stage 2 then plateaus without moving)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"
# Force unbuffered stdout so `| tee` shows epoch lines as they happen rather
# than sitting in an 8 KB block-buffer for ~40 epochs before flushing.
export PYTHONUNBUFFERED=1
CKPT_ROOT="learned_ai/sentinel/checkpoints"
GAME_DIR="${GAME_DIR:-data/games}"
HUMAN_GAME_DIR="${HUMAN_GAME_DIR:-data/human_games}"
DEVICE="${DEVICE:-cpu}"

DATASET_PATH="${DATASET_PATH:-learned_ai/sentinel/datasets/v2_combined.npz}"
DATASET_TOTAL_EXAMPLES="${DATASET_TOTAL_EXAMPLES:-4000000}"
DATASET_MALOM_FRACTION="${DATASET_MALOM_FRACTION:-0.60}"

# Malom DB path (needed for both dataset build and Stages 2 / 4 label queries).
_MALOM_DEFAULT="/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
if [[ -n "${MALOM_DB:-}" ]]; then
    DB_PATH="$MALOM_DB"
elif [[ -f data/training_paths.local.json ]]; then
    DB_PATH="$($PYTHON - <<'PY'
import json
try:
    with open("data/training_paths.local.json") as f:
        cfg = json.load(f)
    print(cfg.get("malom_db_path", ""))
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
    echo "ERROR: Malom DB path $DB_PATH not found." >&2
    echo "       Set MALOM_DB=/path/to/Std_DD_89adjusted or fix training_paths.local.json." >&2
    exit 1
fi

echo "── Step 0 — Sentinel v2 (Stages 1, 2, 4) with combined JSONL + Malom dataset ──"
echo "  game dir           : $GAME_DIR"
echo "  human dir          : $HUMAN_GAME_DIR"
echo "  malom db           : $DB_PATH"
echo "  device             : $DEVICE"
echo "  checkpoint root    : $CKPT_ROOT"
echo "  dataset path       : $DATASET_PATH"
echo "  dataset total ex.  : $DATASET_TOTAL_EXAMPLES"
echo "  dataset malom fr.  : $DATASET_MALOM_FRACTION"
if [[ -n "${PATIENCE:-}" ]]; then
    echo "  patience (env)     : $PATIENCE (overrides config)"
fi
echo

# ── Stage 0.a — build the combined dataset once ──────────────────────────────

if [[ -f "$DATASET_PATH" && "${REBUILD_DATASET:-0}" != "1" ]]; then
    echo "── Dataset — already present at $DATASET_PATH (set REBUILD_DATASET=1 to force) ──"
else
    echo "── Dataset — building $DATASET_PATH ──"
    "$PYTHON" scripts/build_sentinel_dataset_v2.py \
        --out "$DATASET_PATH" \
        --total-examples "$DATASET_TOTAL_EXAMPLES" \
        --malom-fraction "$DATASET_MALOM_FRACTION" \
        --malom-db "$DB_PATH" \
        --game-dir "$GAME_DIR" \
        --human-game-dir "$HUMAN_GAME_DIR"
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "ERROR: dataset build finished but $DATASET_PATH was not produced." >&2
        exit 2
    fi
fi
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

    echo "── Stage $stage — training → $out_dir (dataset=$DATASET_PATH) ──"
    local cmd=(
        "$PYTHON" scripts/train_sentinel.py
        --config "$config"
        --dataset "$DATASET_PATH"
        --drop-db-features
        --out-dir "$out_dir"
        --device "$DEVICE"
    )
    if [[ -n "${PATIENCE:-}" ]]; then
        cmd+=(--patience "$PATIENCE")
    fi
    if [[ "$use_db" == "yes" ]]; then
        cmd+=(--db-path "$DB_PATH")
    fi
    if [[ -n "$resume" ]]; then
        cmd+=(--resume "$resume")
    fi
    echo "  ${cmd[*]}"
    "${cmd[@]}"

    # Fallback: trainer restores best_val from resume; a plateau stage may not
    # write a fresh best.pt.  Carry the resume checkpoint (or latest.pt)
    # forward so the chain continues.
    if [[ ! -f "$best" ]]; then
        if [[ -n "$resume" && -f "$resume" ]]; then
            echo "  Stage $stage did not improve over its resume baseline;"
            echo "  carrying $resume → $best"
            cp "$resume" "$best"
        elif [[ -f "$out_dir/latest.pt" ]]; then
            echo "  Stage $stage best.pt missing but latest.pt exists;"
            echo "  promoting $out_dir/latest.pt → $best"
            cp "$out_dir/latest.pt" "$best"
        else
            echo "ERROR: Stage $stage finished but neither $best nor a resume" >&2
            echo "       checkpoint nor $out_dir/latest.pt is available." >&2
            exit 3
        fi
    fi
    echo "── Stage $stage — complete ($best) ──"
    echo
}

# ── Run the stages ───────────────────────────────────────────────────────────
# Stage 1 is SKIPPED by default.  Its original job was a structural warm-start
# with heuristic labels; empirically that anchors the trunk on the heuristic
# function, and Stage 2 (at low LR) can't move the weights far enough onto the
# Malom labels — a plateau observed in an earlier run (Stage 2 val==Stage 1 val
# to four decimal places for 10 consecutive epochs).  Set RUN_STAGE1=1 to bring
# it back for A/B comparisons.
#
# With Stage 1 skipped, Stage 2 becomes the entry point and starts from a
# random-initialised model on the Malom-labelled combined dataset at its now-
# bumped LR (0.001), giving it enough gradient to actually learn.

if [[ "${RUN_STAGE1:-0}" == "1" ]]; then
    _run_stage 1 configs/sentinel_stage1.yaml "$CKPT_ROOT/v2_stage1" "" no
    _run_stage 2 configs/sentinel_stage2.yaml "$CKPT_ROOT/v2_stage2" "$CKPT_ROOT/v2_stage1/best.pt" yes
else
    echo "── Stage 1 SKIPPED (set RUN_STAGE1=1 to run it) ──"
    echo
    _run_stage 2 configs/sentinel_stage2.yaml "$CKPT_ROOT/v2_stage2" "" yes
fi
_run_stage 4 configs/sentinel_stage4.yaml "$CKPT_ROOT/v2_stage4" "$CKPT_ROOT/v2_stage2/best.pt" yes

# ── Promote Stage 4 → v2 slot (per plan, NOT into production best.pt) ───────

mkdir -p "$CKPT_ROOT/v2"
cp "$CKPT_ROOT/v2_stage4/best.pt" "$CKPT_ROOT/v2/best.pt"
echo "── Promoted $CKPT_ROOT/v2_stage4/best.pt → $CKPT_ROOT/v2/best.pt ──"
echo
echo "Step 0 complete."
echo "Next: Step 1 — regenerate GapNet training dataset with the new Sentinel v2 checkpoint."
echo "Production best.pt remains untouched at $CKPT_ROOT/best.pt (v1) until Step 8 promotion."
