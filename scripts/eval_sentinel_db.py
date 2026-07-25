#!/usr/bin/env python3
"""scripts/eval_sentinel_db.py — Sentinel eval on a fresh JSONL+Malom sample (Step 6a).

Samples an independent test set with the SAME 60% Malom / 40% JSONL composition
the training builder uses, but with a different default RNG seed (99999 vs the
builder's 42) so overlap with the training set is minimised.  Reports:

    win_acc         fraction of DB-win moves the sentinel scores > 0.5
    loss_acc        fraction of DB-loss moves the sentinel scores < 0.5
    top1_win_rate   positions with a DB-win available where sentinel #1 is a win
    spearman_r      Spearman rank correlation between sentinel and DB quality
    dtm_pearson_r   Pearson correlation between sentinel and DB DTM
    phase_breakdown same four metrics split by place / move / fly
    source_breakdown same four metrics split by Malom source vs JSONL source

The per-source split is the *contamination fence*: the Malom-source metrics
still overlap the training pool at the state-key level (some fraction of
sampled state_keys will coincide with training positions), whereas the JSONL-
source metrics are pulled from a random subset of game FILES which are unlikely
to have been fully consumed at training time.  A large gap between the two
suggests memorisation; near-identical numbers suggest genuine generalisation.

specialist_db is NOT sampled — its `pos_hash` primary key is a non-reversible
SHA-1, so board reconstruction from the DB alone is not possible.

Usage:
    .venv/bin/python scripts/eval_sentinel_db.py \\
        --checkpoint learned_ai/sentinel/checkpoints/best.pt \\
        --output eval_sentinel_db_v1.json --n-samples 1000

    .venv/bin/python scripts/eval_sentinel_db.py \\
        --checkpoint learned_ai/sentinel/checkpoints/v2/best.pt \\
        --output eval_sentinel_db_v2.json --n-samples 1000

Run each seed 2-3 times with different --seed to gauge sampling variance.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase


# ── Board reconstruction (mirrors ai.trajectory_db.make_board_state_key) ──────

def _board_from_state_key(state_key: str) -> BoardState | None:
    parts = state_key.split("|")
    if len(parts) != 7:
        return None
    canon24, turn, _phase, pw, pb, ow, ob = parts
    if len(canon24) != 24:
        return None
    try:
        from game.board import POSITIONS
        positions = {p: (canon24[i] if canon24[i] != "." else "") for i, p in enumerate(POSITIONS)}
        return BoardState(
            positions=positions,
            turn=turn,
            pieces_on_board={"W": int(ow), "B": int(ob)},
            pieces_placed={"W": int(pw), "B": int(pb)},
            pieces_captured={"W": max(0, int(pb) - int(ob)),
                             "B": max(0, int(pw) - int(ow))},
        )
    except Exception:
        return None


# ── Statistical helpers ──────────────────────────────────────────────────────

def _spearman_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rank_x = _rank(xs)
    rank_y = _rank(ys)
    return _pearson(rank_x, rank_y)


def _rank(vs: list[float]) -> list[float]:
    idx    = sorted(range(len(vs)), key=lambda i: vs[i])
    ranks  = [0.0] * len(vs)
    for r, i in enumerate(idx):
        ranks[i] = float(r)
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


# ── Per-move DB quality signal ───────────────────────────────────────────────

_WDL_QUALITY = {"L": 1.0, "D": 0.5, "W": 0.0}   # child perspective: L = we caused opponent loss = good


def _malom_move_signals(board: BoardState, legal: list[dict], malom_db) -> list[tuple[str, int]]:
    """Return [(wdl, dtw)] for each legal move by applying → querying Malom."""
    out: list[tuple[str, int]] = []
    for m in legal:
        try:
            succ = board.apply_move(m)
            v    = malom_db.query(succ)
            if v is None:
                out.append(("?", 0))
            else:
                out.append((v["outcome"], v["dtw"]))
        except Exception:
            out.append(("?", 0))
    return out


# ── Main eval ────────────────────────────────────────────────────────────────

def _sample_boards_from_jsonl(
    game_dir: Path, human_game_dir: "Path | None",
    n_files: int, rng,
) -> "list[BoardState]":
    """Pick N random JSONL game files and yield boards from every ply."""
    import glob
    from learned_ai.sentinel.dataset import _iter_game_records, _board_from_fen_before

    files = sorted(glob.glob(str(game_dir / "**" / "*.jsonl"), recursive=True))
    if human_game_dir is not None:
        files += sorted(glob.glob(str(human_game_dir / "**" / "*.jsonl"), recursive=True))
    if not files:
        return []
    if len(files) > n_files:
        idx = rng.choice(len(files), size=n_files, replace=False)
        files = [files[i] for i in idx]
    boards: list = []
    for path in files:
        for record in _iter_game_records(path):
            moves = record.get("moves") or []
            for log_move in moves:
                fen = log_move.get("board_fen_before")
                if not fen:
                    continue
                board = _board_from_fen_before(fen)
                if board is not None:
                    boards.append(board)
    return boards


def evaluate(
    checkpoint: Path,
    human_db: Path,
    malom_db_path: Path,
    n_samples: int,
    seed: int,
    jsonl_fraction: float = 0.40,
    game_dir: "Path | None" = None,
    human_game_dir: "Path | None" = None,
) -> dict:
    """Run the eval on a fresh composite sample: (1-jsonl_fraction) from
    human_db state_keys (Malom-labelled, phase-stratified) + jsonl_fraction
    from replayed JSONL games.  Matches the training builder's 60/40 default
    composition but uses a different `seed` (default 99999) so overlap with
    the training sample is minimised."""
    import numpy as np
    from ai.malom_db import MalomDB
    from learned_ai.sentinel.infer import SentinelAdvisor

    if not (0.0 <= jsonl_fraction <= 1.0):
        raise ValueError("jsonl_fraction must be in [0, 1]")

    rng = np.random.default_rng(seed)

    n_malom = int(round(n_samples * (1.0 - jsonl_fraction)))
    n_jsonl = n_samples - n_malom

    # ── Malom-sampled portion (phase-stratified from human_db) ──────────────
    conn = sqlite3.connect(str(human_db))

    def _sample(phase: str, n: int) -> list[str]:
        rows = conn.execute(
            "SELECT state_key FROM positions WHERE malom_wdl IS NOT NULL "
            f"AND state_key LIKE '%|{phase}|%' LIMIT ?",
            (n * 8,),
        ).fetchall()
        if not rows:
            return []
        keys = [r[0] for r in rows]
        if len(keys) <= n:
            return keys
        idx = rng.choice(len(keys), size=n, replace=False)
        return [keys[i] for i in idx]

    per_phase = max(1, n_malom // 3)
    malom_boards: list = []
    for phase in ("place", "move", "fly"):
        for sk in _sample(phase, per_phase):
            b = _board_from_state_key(sk)
            if b is not None:
                malom_boards.append(b)
    conn.close()
    print(f"Malom-source: {len(malom_boards)} boards "
          f"(target {n_malom} across place / move / fly, seed={seed}).")

    # ── JSONL-sampled portion (random game files replayed) ────────────────
    jsonl_boards: list = []
    if n_jsonl > 0 and game_dir is not None:
        # ~10-40 boards per file — sample enough files to hit n_jsonl.
        est_boards_per_file = 20
        n_files = max(10, n_jsonl // est_boards_per_file)
        jsonl_boards = _sample_boards_from_jsonl(game_dir, human_game_dir, n_files, rng)
        if len(jsonl_boards) > n_jsonl:
            idx = rng.choice(len(jsonl_boards), size=n_jsonl, replace=False)
            jsonl_boards = [jsonl_boards[i] for i in idx]
        print(f"JSONL-source: {len(jsonl_boards)} boards from {n_files} game files.")
    elif n_jsonl > 0:
        print(f"JSONL-source: 0 boards (no --game-dir supplied; --jsonl-fraction 0 to silence).")

    # Tag boards with source so metrics can be reported per source.
    tagged = [(b, "malom") for b in malom_boards] + [(b, "jsonl") for b in jsonl_boards]
    # Shuffle so progress bar reflects both sources evenly.
    rng.shuffle(tagged)

    # Load sentinel + malom.
    advisor = SentinelAdvisor(checkpoint_path=str(checkpoint))
    malom   = MalomDB(str(malom_db_path))
    # Warm sentinel lazy state.
    board_tmp = BoardState.new_game()
    advisor.advise(board_tmp, [{"from": None, "to": "a1", "capture": None}],
                   board_tmp.turn, played_move_idx=0)

    # Per-phase AND per-source counters.
    counters = {
        "overall": defaultdict(int),
        "place":   defaultdict(int),
        "move":    defaultdict(int),
        "fly":     defaultdict(int),
        "src_malom": defaultdict(int),
        "src_jsonl": defaultdict(int),
    }
    # Split top1/spearman/dtm collectors per source so training-contamination
    # (which mostly affects the malom source) is visible in the output.
    quality_pairs: dict[str, list[tuple[float, float]]] = {"malom": [], "jsonl": []}
    dtm_pairs:     dict[str, list[tuple[float, int]]]   = {"malom": [], "jsonl": []}
    top1_positions = {"malom": 0, "jsonl": 0}
    top1_win_positions = {"malom": 0, "jsonl": 0}
    top1_win_correct   = {"malom": 0, "jsonl": 0}
    skipped              = 0

    for i, (board, source) in enumerate(tagged):
        try:
            phase = get_game_phase(board, board.turn)
        except Exception:
            skipped += 1
            continue
        legal = get_all_legal_moves(board)
        if not legal:
            skipped += 1
            continue
        try:
            advice = advisor.advise(board, legal, board.turn, played_move_idx=0)
        except Exception:
            skipped += 1
            continue
        if advice is None or len(advice.move_scores) != len(legal):
            skipped += 1
            continue
        sent_scores = list(advice.move_scores)
        malom_move  = _malom_move_signals(board, legal, malom)
        if not any(w != "?" for w, _ in malom_move):
            skipped += 1
            continue

        for (wdl, dtw), score in zip(malom_move, sent_scores):
            if wdl not in _WDL_QUALITY:
                continue
            q = _WDL_QUALITY[wdl]
            quality_pairs[source].append((score, q))
            dtm_pairs[source].append((score, int(dtw)))
            for cell in ("overall", phase, f"src_{source}"):
                c = counters[cell]
                if wdl == "L":
                    c["win_total"] += 1
                    if score > 0.5:
                        c["win_correct"] += 1
                elif wdl == "W":
                    c["loss_total"] += 1
                    if score < 0.5:
                        c["loss_correct"] += 1

        # top1_win_rate: positions where a DB-win exists, does sentinel top-1 pick one?
        db_win_indices = [j for j, (w, _) in enumerate(malom_move) if w == "L"]
        if db_win_indices:
            top1_win_positions[source] += 1
            top1_idx = max(range(len(sent_scores)), key=lambda k: sent_scores[k])
            if malom_move[top1_idx][0] == "L":
                top1_win_correct[source] += 1
        top1_positions[source] += 1

        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(tagged)}  skipped={skipped}")

    def _safe_div(a: int, b: int) -> float:
        return round(a / b, 4) if b > 0 else 0.0

    def _phase_summary(bucket) -> dict:
        return {
            "win_acc":  _safe_div(bucket["win_correct"],  bucket["win_total"]),
            "loss_acc": _safe_div(bucket["loss_correct"], bucket["loss_total"]),
            "n_win":    bucket["win_total"],
            "n_loss":   bucket["loss_total"],
        }

    def _all_quality_pairs():
        return quality_pairs["malom"] + quality_pairs["jsonl"]

    def _all_dtm_pairs():
        return dtm_pairs["malom"] + dtm_pairs["jsonl"]

    def _source_summary(src: str) -> dict:
        return {
            "n_positions":    top1_positions[src],
            "win_acc":        _phase_summary(counters[f"src_{src}"])["win_acc"],
            "loss_acc":       _phase_summary(counters[f"src_{src}"])["loss_acc"],
            "top1_win_rate":  _safe_div(top1_win_correct[src], top1_win_positions[src]),
            "spearman_r":     round(_spearman_r([p[0] for p in quality_pairs[src]],
                                                [p[1] for p in quality_pairs[src]]), 4),
            "dtm_pearson_r":  round(_pearson([p[0] for p in dtm_pairs[src]],
                                             [float(p[1]) for p in dtm_pairs[src]]), 4),
        }

    result = {
        "checkpoint":     str(checkpoint),
        "seed":           seed,
        "jsonl_fraction": jsonl_fraction,
        "n_positions":    sum(top1_positions.values()),
        "n_skipped":      skipped,
        "win_acc":        _phase_summary(counters["overall"])["win_acc"],
        "loss_acc":       _phase_summary(counters["overall"])["loss_acc"],
        "top1_win_rate":  _safe_div(sum(top1_win_correct.values()),
                                    sum(top1_win_positions.values())),
        "spearman_r":     round(_spearman_r([p[0] for p in _all_quality_pairs()],
                                            [p[1] for p in _all_quality_pairs()]), 4),
        "dtm_pearson_r":  round(_pearson([p[0] for p in _all_dtm_pairs()],
                                         [float(p[1]) for p in _all_dtm_pairs()]), 4),
        "phase_breakdown": {
            phase: _phase_summary(counters[phase]) for phase in ("place", "move", "fly")
        },
        "source_breakdown": {
            "malom": _source_summary("malom"),
            "jsonl": _source_summary("jsonl"),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Sentinel checkpoint to evaluate.")
    parser.add_argument("--human-db",  type=Path, default=Path("data/human_db.sqlite"))
    parser.add_argument("--malom-db",  type=Path,
                        default=Path("/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"))
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Total positions to sample; split between Malom-source "
                             "and JSONL-source per --jsonl-fraction.")
    parser.add_argument("--jsonl-fraction", type=float, default=0.40,
                        help="Fraction of the sample drawn from JSONL replay "
                             "(matches training builder's 60/40 default).")
    parser.add_argument("--game-dir",       type=Path,
                        default=Path("data/games"),
                        help="AI self-play JSONL directory for the JSONL-source sample.")
    parser.add_argument("--human-game-dir", type=Path,
                        default=Path("data/human_games"),
                        help="Human game JSONL directory for the JSONL-source sample.")
    parser.add_argument("--output",    type=Path, default=None,
                        help="Optional JSON path for the result summary.")
    parser.add_argument("--seed",      type=int, default=99999,
                        help="RNG seed for the eval sample.  Default 99999 is "
                             "intentionally different from the training builder's "
                             "default (42) so overlap with the training set is "
                             "minimised.  Pick a distinct seed per eval run to "
                             "explore sampling variance.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.human_db.exists():
        raise SystemExit(f"human_db not found: {args.human_db}")
    if not args.malom_db.exists():
        raise SystemExit(f"malom_db not found: {args.malom_db}")

    result = evaluate(
        checkpoint=args.checkpoint,
        human_db=args.human_db,
        malom_db_path=args.malom_db,
        n_samples=args.n_samples,
        seed=args.seed,
        jsonl_fraction=args.jsonl_fraction,
        game_dir=args.game_dir if args.game_dir.exists() else None,
        human_game_dir=args.human_game_dir if args.human_game_dir and args.human_game_dir.exists() else None,
    )
    print()
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
