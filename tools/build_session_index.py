#!/usr/bin/env python3
"""tools/build_session_index.py — game-level and player-level split diagnostic index.

Scans every JSONL file in `data/human_games/`, maps each game to a
split tier via `game_level_split(session_id)`, and maps each observed
board position to a `game_split_mask` bitmask indicating which split
tiers reached that state_key in the training dataset.

Bitmask encoding (uint8):
    bit 0 (0x01) — state_key appeared in at least one game-train game
    bit 1 (0x02) — state_key appeared in at least one game-val game
    bit 2 (0x04) — state_key appeared in at least one game-test game

Positions that appear only in game-val games (mask & 0x03 == 0x02) are
"game-val-only" — they can be used to diagnose how much of the val
signal survives a stricter split.

Also builds a `player_split_mask` array on the same basis, using the
white_player / black_player fields as the player identity.

Output: `data/human_move_policy_session_index.npz`

Usage
-----
    .venv/bin/python tools/build_session_index.py \\
        --dataset-dir data/human_move_policy_dataset \\
        --games-dir   data/human_games \\
        --output      data/human_move_policy_session_index.npz

Smoke run:
    .venv/bin/python tools/build_session_index.py --limit-files 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.trajectory_db import make_board_state_key           # noqa: E402
from game.board import BoardState                            # noqa: E402
from learned_ai.data.human_db_split import game_level_split  # noqa: E402


# Bitmask constants for game_split_mask / player_split_mask.
MASK_TRAIN = np.uint8(0x01)
MASK_VAL   = np.uint8(0x02)
MASK_TEST  = np.uint8(0x04)

_SPLIT_TO_MASK = {"train": MASK_TRAIN, "val": MASK_VAL, "test": MASK_TEST}


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build(
    dataset_dir: Path,
    games_dir: Path,
    output: Path,
    limit_files: Optional[int] = None,
) -> dict:
    t0 = time.time()

    # Load the dataset state_key index — we need to know which row index
    # corresponds to each state_key.
    meta_path = dataset_dir / "metadata.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.npz not found in {dataset_dir}")
    d = np.load(meta_path, allow_pickle=True)
    state_keys = d["state_keys"]   # object array of strings
    n_states   = int(state_keys.shape[0])
    print(f"[session_index] Dataset: {n_states:,} state_keys")

    # Build a str → int lookup (state_key → row index in state_keys).
    sk_to_idx: dict[str, int] = {str(sk): i for i, sk in enumerate(state_keys)}

    # Initialise output masks.
    game_split_mask:   np.ndarray = np.zeros(n_states, dtype=np.uint8)
    player_split_mask: np.ndarray = np.zeros(n_states, dtype=np.uint8)

    # Enumerate JSONL files.
    jsonl_files = sorted(games_dir.glob("*.jsonl"))
    if limit_files is not None:
        jsonl_files = jsonl_files[:limit_files]
    print(f"[session_index] Scanning {len(jsonl_files):,} JSONL files …")

    n_games_seen    = 0
    n_games_skipped = 0
    n_moves_seen    = 0
    n_hits          = 0

    for file_idx, fpath in enumerate(jsonl_files):
        with fpath.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                except Exception:
                    n_games_skipped += 1
                    continue

                session_id   = str(rec.get("session_id") or fpath.stem)
                game_split   = game_level_split(session_id)
                game_mask    = _SPLIT_TO_MASK[game_split]

                # Player identities for the player-level diagnostic.
                white_player = str(rec.get("white_player") or "")
                black_player = str(rec.get("black_player") or "")

                moves = rec.get("moves") or []
                for mv in moves:
                    fen = mv.get("board_fen_before")
                    if not fen:
                        continue
                    try:
                        board = BoardState.from_fen_string(fen)
                        sk, _ = make_board_state_key(board)
                    except Exception:
                        continue
                    n_moves_seen += 1

                    idx = sk_to_idx.get(str(sk))
                    if idx is None:
                        continue
                    n_hits += 1
                    game_split_mask[idx] |= game_mask

                    # Player-level: use colour to assign the player identity
                    # for this move, then hash it.
                    color = mv.get("color") or ""
                    mover_player = white_player if color == "white" else black_player
                    if mover_player:
                        player_split = game_level_split(mover_player)
                        player_split_mask[idx] |= _SPLIT_TO_MASK[player_split]

                n_games_seen += 1

        if (file_idx + 1) % 5000 == 0:
            print(f"[session_index]  … {file_idx + 1:,}/{len(jsonl_files):,} files "
                  f"(games={n_games_seen:,}  hits={n_hits:,}  t={time.time()-t0:.0f}s)")

    # Summary counters.
    game_val_only = int(np.sum(
        ((game_split_mask & MASK_TRAIN) == 0) & ((game_split_mask & MASK_VAL) != 0)
    ))
    n_covered = int((game_split_mask != 0).sum())

    provenance = {
        "builder_version":       "1",
        "builder_git_commit":    _git_head() or "",
        "dataset_dir":           str(dataset_dir),
        "dataset_meta_sha256":   _sha256_file(meta_path),
        "games_dir":             str(games_dir),
        "n_jsonl_files_scanned": len(jsonl_files),
        "n_games_seen":          n_games_seen,
        "n_games_skipped":       n_games_skipped,
        "n_moves_seen":          n_moves_seen,
        "n_hits":                n_hits,
        "n_state_keys_covered":  n_covered,
        "n_game_val_only":       game_val_only,
        "mask_encoding":         "uint8: bit0=train bit1=val bit2=test",
        "elapsed_seconds":       round(time.time() - t0, 1),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        game_split_mask=game_split_mask,
        player_split_mask=player_split_mask,
        provenance=np.array(json.dumps(provenance), dtype=object),
    )
    print(f"[session_index] Saved → {output}")
    print(f"[session_index] Covered {n_covered:,}/{n_states:,} state_keys  "
          f"game_val_only={game_val_only:,}  elapsed={provenance['elapsed_seconds']}s")
    return provenance


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/human_move_policy_dataset"))
    p.add_argument("--games-dir",   type=Path, default=Path("data/human_games"))
    p.add_argument("--output",      type=Path,
                   default=Path("data/human_move_policy_session_index.npz"))
    p.add_argument("--limit-files", type=int, default=None,
                   help="Cap number of JSONL files scanned (smoke test).")
    args = p.parse_args()

    prov = build(args.dataset_dir, args.games_dir, args.output,
                 limit_files=args.limit_files)
    prov_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
