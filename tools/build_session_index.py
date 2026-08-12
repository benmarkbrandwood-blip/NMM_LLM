#!/usr/bin/env python3
"""tools/build_session_index.py — game-level and player-level split diagnostic index.

Scans every JSONL file listed in a frozen session ledger, uses the ledger's
session→split assignments (NOT a re-computed game_level_split call), and
maps each observed board position to a `game_split_mask` bitmask indicating
which split tiers reached that state_key in the training dataset.

Bitmask encoding (uint8):
    bit 0 (0x01) — state_key appeared in at least one game-train game
    bit 1 (0x02) — state_key appeared in at least one game-val game
    bit 2 (0x04) — state_key appeared in at least one game-test game

Also builds a `player_split_mask` array on the same basis, using the
white_player / black_player fields as the player identity (via
game_level_split(player_id) — player-level split is independent of the
session ledger).

Codex P1-A hardening (2026-08-12): the session_index is now cryptographically
bound to the source ledger.
- --session-ledger PATH is REQUIRED.
- Ledger loaded via _verify_ledger_complete() from build_gap_v3_session_ledger
  (fail-closed on partial/non-strict ledgers).
- Each JSONL file being scanned is verified to appear in the ledger's file
  manifest with matching SHA-256 — drift/rewritten files fail closed.
- session→split lookup comes from the ledger; sessions absent from the
  ledger are counted (n_sessions_not_in_ledger) but never contribute to
  a mask.
- Output provenance records ledger_path, ledger_sha256,
  ledger_files_manifest_sha256, ledger_version, and split_manifest_version
  so downstream (HMPN extractor) can verify the (ledger, index) pair.
- Atomic publish (write to .tmp, fsync, rename) and no-clobber.

Output: `data/human_move_policy_session_index.npz`

Usage
-----
    .venv/bin/python tools/build_session_index.py \\
        --dataset-dir     data/human_move_policy_dataset \\
        --games-dir       data/human_games \\
        --session-ledger  data/gap_v3_session_ledger.json \\
        --output          data/human_move_policy_session_index.npz

Smoke run (partial ledger permitted via --allow-partial-ledger):
    .venv/bin/python tools/build_session_index.py \\
        --session-ledger  data/gap_v3_session_ledger.smoke.json \\
        --allow-partial-ledger \\
        --limit-files 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from tools.build_gap_v3_session_ledger import (              # noqa: E402
    LedgerBuildError, _verify_ledger_complete,
)


# Bitmask constants for game_split_mask / player_split_mask.
MASK_TRAIN = np.uint8(0x01)
MASK_VAL   = np.uint8(0x02)
MASK_TEST  = np.uint8(0x04)

_SPLIT_TO_MASK = {"train": MASK_TRAIN, "val": MASK_VAL, "test": MASK_TEST}


class SessionIndexBuildError(Exception):
    """Raised when the session_index builder refuses under P1-A fail-closed rules."""


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


def _read_file_bytes_and_sha(path: Path) -> tuple[bytes, str]:
    """Single-pass read: hash + content from the same bytes."""
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def build(
    dataset_dir: Path,
    games_dir: Path,
    session_ledger_path: Path,
    output: Path,
    limit_files: Optional[int] = None,
    force: bool = False,
    allow_partial_ledger: bool = False,
) -> dict:
    """Build a state_key → game_split_mask index bound to the given ledger.

    Codex P1-A (2026-08-12): the index is cryptographically bound to
    `session_ledger_path` via ledger SHA and files_manifest_sha256 recorded
    in the output provenance.  Downstream HMPN extractor verifies these
    match its own `--session-ledger` argument.
    """
    t0 = time.time()

    # ── Verify + load ledger ─────────────────────────────────────────────────
    try:
        ledger = _verify_ledger_complete(session_ledger_path,
                                         allow_partial=allow_partial_ledger)
    except LedgerBuildError as e:
        raise SessionIndexBuildError(
            f"[session_index] Ledger verification failed: {e}"
        ) from e
    ledger_sha = _sha256_file(session_ledger_path)

    session_split_map: dict[str, str] = {
        s["session_id"]: s["split"] for s in ledger["sessions"]
    }
    ledger_file_shas: dict[str, str] = {
        f["rel_path"]: f["sha256"] for f in ledger["files"]
    }
    print(f"[session_index] Ledger: {len(session_split_map):,} sessions, "
          f"{len(ledger_file_shas):,} files  "
          f"(files_manifest_sha256={ledger.get('files_manifest_sha256', '?')[:16]}…)")

    # ── No-clobber ───────────────────────────────────────────────────────────
    if output.exists() and not force:
        raise SessionIndexBuildError(
            f"[session_index] Refusing to overwrite existing output: {output}.  "
            f"Delete or pass --force to overwrite."
        )

    # ── Load dataset state_keys ──────────────────────────────────────────────
    meta_path = dataset_dir / "metadata.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.npz not found in {dataset_dir}")
    d = np.load(meta_path, allow_pickle=True)
    state_keys = d["state_keys"]
    n_states   = int(state_keys.shape[0])
    print(f"[session_index] Dataset: {n_states:,} state_keys")

    sk_to_idx: dict[str, int] = {str(sk): i for i, sk in enumerate(state_keys)}

    # ── Output masks ─────────────────────────────────────────────────────────
    game_split_mask   = np.zeros(n_states, dtype=np.uint8)
    player_split_mask = np.zeros(n_states, dtype=np.uint8)

    # ── Enumerate JSONL files; verify against ledger file manifest ──────────
    jsonl_files = sorted(games_dir.glob("*.jsonl"))
    if limit_files is not None:
        jsonl_files = jsonl_files[:limit_files]
    print(f"[session_index] Scanning {len(jsonl_files):,} JSONL files "
          f"(limit_files={limit_files}) …")

    n_games_seen             = 0
    n_games_skipped_malformed = 0
    n_moves_seen             = 0
    n_hits                   = 0
    n_sessions_not_in_ledger = 0
    n_files_missing_from_ledger = 0

    for file_idx, fpath in enumerate(jsonl_files):
        rel_path = fpath.relative_to(games_dir).as_posix()
        expected_sha = ledger_file_shas.get(rel_path)
        raw_bytes, actual_sha = _read_file_bytes_and_sha(fpath)
        if expected_sha is None:
            n_files_missing_from_ledger += 1
            if limit_files is None:
                # Full run: refuse if scanned file is not in ledger inventory
                raise SessionIndexBuildError(
                    f"[session_index] File {rel_path} not in ledger's file "
                    f"manifest.  Ledger and games_dir have drifted; rebuild "
                    f"the ledger against the current games_dir."
                )
            # Under --limit-files smoke run, skip the file
            continue
        if actual_sha != expected_sha:
            raise SessionIndexBuildError(
                f"[session_index] File SHA-256 mismatch for {rel_path}: "
                f"ledger recorded {expected_sha}, current file is {actual_sha}.  "
                f"The JSONL file has been modified since the ledger was built."
            )

        for raw_line_b in raw_bytes.splitlines():
            try:
                raw_line = raw_line_b.decode("utf-8").strip()
            except UnicodeDecodeError:
                n_games_skipped_malformed += 1
                continue
            if not raw_line:
                continue
            try:
                rec = json.loads(raw_line)
            except Exception:
                n_games_skipped_malformed += 1
                continue

            session_id = str(rec.get("session_id") or fpath.stem)
            game_split = session_split_map.get(session_id)
            if game_split is None:
                n_sessions_not_in_ledger += 1
                continue
            game_mask = _SPLIT_TO_MASK[game_split]

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

                color = mv.get("color") or ""
                mover_player = white_player if color == "white" else black_player
                if mover_player:
                    player_split = game_level_split(mover_player)
                    player_split_mask[idx] |= _SPLIT_TO_MASK[player_split]

            n_games_seen += 1

        if (file_idx + 1) % 5000 == 0:
            print(f"[session_index]  … {file_idx + 1:,}/{len(jsonl_files):,} files "
                  f"(games={n_games_seen:,}  hits={n_hits:,}  "
                  f"t={time.time()-t0:.0f}s)")

    # ── Summary counters ─────────────────────────────────────────────────────
    game_val_only = int(np.sum(
        ((game_split_mask & MASK_TRAIN) == 0) & ((game_split_mask & MASK_VAL) != 0)
    ))
    n_covered = int((game_split_mask != 0).sum())

    provenance = {
        "builder_version":       "2",   # P1-A: bumped for ledger binding
        "builder_git_commit":    _git_head() or "",
        "dataset_dir":           str(dataset_dir),
        "dataset_meta_sha256":   _sha256_file(meta_path),
        "games_dir":             str(games_dir),
        # ── Ledger binding (Codex P1-A) ────────────────────────────────────
        "session_ledger_path":                str(session_ledger_path),
        "ledger_sha256":                      ledger_sha,
        "ledger_files_manifest_sha256":       ledger.get("files_manifest_sha256"),
        "ledger_version":                     ledger.get("ledger_version"),
        "ledger_split_manifest_version":      ledger.get("split_manifest_version"),
        "ledger_n_sessions":                  ledger.get("n_sessions"),
        "ledger_is_partial":                  ledger.get("is_partial", False),
        # ── Scan stats ─────────────────────────────────────────────────────
        "n_jsonl_files_scanned":              len(jsonl_files),
        "n_games_seen":                       n_games_seen,
        "n_games_skipped_malformed":          n_games_skipped_malformed,
        "n_moves_seen":                       n_moves_seen,
        "n_hits":                             n_hits,
        "n_state_keys_covered":               n_covered,
        "n_game_val_only":                    game_val_only,
        "n_sessions_not_in_ledger":           n_sessions_not_in_ledger,
        "n_files_missing_from_ledger":        n_files_missing_from_ledger,
        "mask_encoding":                      "uint8: bit0=train bit1=val bit2=test",
        "limit_files":                        limit_files,
        "elapsed_seconds":                    round(time.time() - t0, 1),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    np.savez(
        tmp_path,
        game_split_mask=game_split_mask,
        player_split_mask=player_split_mask,
        provenance=np.array(json.dumps(provenance), dtype=object),
    )
    # np.savez may append ".npz" — normalise before rename
    if not tmp_path.exists() and tmp_path.with_suffix(tmp_path.suffix + ".npz").exists():
        tmp_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    # Ensure fsync on the temporary file
    with tmp_path.open("rb") as f:
        os.fsync(f.fileno())
    tmp_path.replace(output)

    print(f"[session_index] Saved → {output}")
    print(f"[session_index] Covered {n_covered:,}/{n_states:,} state_keys  "
          f"game_val_only={game_val_only:,}  "
          f"elapsed={provenance['elapsed_seconds']}s")
    if n_sessions_not_in_ledger:
        print(f"[session_index] ⚠️  n_sessions_not_in_ledger={n_sessions_not_in_ledger} "
              f"(session_ids present in JSONL but absent from ledger)")
    return provenance


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-dir",    type=Path, default=Path("data/human_move_policy_dataset"))
    p.add_argument("--games-dir",      type=Path, default=Path("data/human_games"))
    p.add_argument("--session-ledger", type=Path, required=True,
                   help="Path to data/gap_v3_session_ledger.json.  REQUIRED per Codex P1-A: "
                        "session_id → split assignments come from the ledger, "
                        "not a recomputed game_level_split() call.")
    p.add_argument("--output",         type=Path,
                   default=Path("data/human_move_policy_session_index.npz"))
    p.add_argument("--limit-files",    type=int, default=None,
                   help="Cap number of JSONL files scanned (smoke test).  Under a "
                        "smoke run, files missing from the ledger inventory are "
                        "skipped rather than fatal.")
    p.add_argument("--force",          action="store_true",
                   help="Overwrite existing --output.")
    p.add_argument("--allow-partial-ledger", action="store_true",
                   help="Accept a ledger built with --limit-files (is_partial=True).  "
                        "For smoke tests only; production runs must use a full ledger.")
    args = p.parse_args()

    prov = build(
        args.dataset_dir, args.games_dir, args.session_ledger, args.output,
        limit_files=args.limit_files,
        force=args.force,
        allow_partial_ledger=args.allow_partial_ledger,
    )
    prov_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
