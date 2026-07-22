"""Build and verify the reviewed Sanmill Oracle start-position corpus.

The source book uses Sanmill's staged place/remove action model.  NMM_LLM
stores a complete turn atomically and its compact FEN has no pending-removal
field.  Only stable ``action=p`` source positions are therefore eligible as
direct evaluation starts.  Pending removals are projected only for provenance
and must resolve into an already selected ring16 orbit.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from game.board import ADJACENCY, POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.training.run_contract import canonical_sha256


CORPUS_SCHEMA = "nmm.oracle-start-corpus.v2"
REVIEW_MANIFEST_SCHEMA = "nmm.oracle-review-assets.v1"
EVALUATION_ID = "dev-v4-formal-paired-eval-v1"
EXPECTED_SANMILL_COMMIT = "6a64010aed7ea4193502ea17c242f68e09fe576a"
EXPECTED_SOURCE_SHA256 = "d304434a46a812a6ecbd5fe1779baf853025516082dee7e50de09969bd270a6b"
EXPECTED_INVALID_ORACLE_MOVE = {
    "raw_key_sha256": (
        "904777ade504367c4e62446f105f1b125aaea7d6bec217984518025d8df3b0d1"
    ),
    "action": "p",
    "moves": ["c3"],
}
OWNER_REVIEW_DECISION_DATE = "2026-07-22"
OWNER_REVIEW_EXCLUSION = {
    "original_review_index": 101,
    "fen": ".BB..B.WBWBW.BWWW....WW.|B|8|7",
    "ring16_canonical_fen": "....W.WW.WBWBWWB..BB.W.B|B|8|7",
    "raw_key_sha256": EXPECTED_INVALID_ORACLE_MOVE["raw_key_sha256"],
}
SOURCE_ASSET = PurePosixPath(
    "src/ui/flutter_app/assets/opening_books/nmm/opening_book.json"
)

_NODE_FOR_NMM_POSITION = (
    23, 16, 17, 18, 19, 20, 21, 22,
    15, 8, 9, 10, 11, 12, 13, 14,
    7, 0, 1, 2, 3, 4, 5, 6,
)
_D4 = (
    (1, 0, 0, 1),
    (0, -1, 1, 0),
    (-1, 0, 0, -1),
    (0, 1, -1, 0),
    (-1, 0, 0, 1),
    (1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, -1, -1, 0),
)
_FILE_X = {"a": -3, "b": -2, "c": -1, "d": 0, "e": 1, "f": 2, "g": 3}
_POSITION_COORDS = {
    position: (_FILE_X[position[0]], int(position[1]) - 4)
    for position in POSITIONS
}
_COORD_POSITION = {coords: position for position, coords in _POSITION_COORDS.items()}
_POSITION_INDEX = {position: index for index, position in enumerate(POSITIONS)}


class OracleCorpusError(RuntimeError):
    """Raised when source or generated corpus evidence is invalid."""


@dataclass(frozen=True)
class OracleProjection:
    raw_key: str
    raw_key_sha256: str
    oracle_moves: tuple[str, ...]
    legal_oracle_moves: tuple[str, ...]
    illegal_oracle_moves: tuple[str, ...]
    action: str
    direct_fen: str | None
    stable_fen: str

    def source_record(self) -> dict[str, Any]:
        return {
            "raw_key": self.raw_key,
            "raw_key_sha256": self.raw_key_sha256,
            "action": self.action,
            "oracle_moves": list(self.oracle_moves),
            "nmm_legal_oracle_moves": list(self.legal_oracle_moves),
            "nmm_illegal_oracle_moves": list(self.illegal_oracle_moves),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_document(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _run_git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={root.resolve().as_posix()}",
        "-C",
        str(root),
        *args,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OracleCorpusError(f"Sanmill Git inspection failed: {detail}")
    return completed.stdout.strip()


def _run_git_bytes(root: Path, *args: str) -> bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={root.resolve().as_posix()}",
        "-C",
        str(root),
        *args,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail_bytes = completed.stderr.strip() or completed.stdout.strip()
        detail = detail_bytes.decode("utf-8", errors="replace")
        raise OracleCorpusError(f"Sanmill Git inspection failed: {detail}")
    return completed.stdout


def load_pinned_sanmill_book(
    paths_config: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the pinned Sanmill opening-book blob via the local registry."""
    config_path = Path(paths_config)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        root_value = config["sanmill_checkout"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise OracleCorpusError(
            "paths config must provide a readable sanmill_checkout"
        ) from exc
    if not isinstance(root_value, str) or not root_value:
        raise OracleCorpusError("sanmill_checkout must be a non-empty path")
    root = Path(root_value)
    commit = _run_git(
        root,
        "rev-parse",
        "--verify",
        f"{EXPECTED_SANMILL_COMMIT}^{{commit}}",
    )
    if commit != EXPECTED_SANMILL_COMMIT:
        raise OracleCorpusError(
            f"Sanmill commit resolution changed: expected {EXPECTED_SANMILL_COMMIT}, got {commit}"
        )
    asset_bytes = _run_git_bytes(
        root,
        "show",
        f"{EXPECTED_SANMILL_COMMIT}:{SOURCE_ASSET.as_posix()}",
    )
    source_sha256 = hashlib.sha256(asset_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise OracleCorpusError(
            "Sanmill opening-book SHA-256 differs from the reviewed source"
        )
    try:
        book = json.loads(asset_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleCorpusError("cannot parse the Sanmill opening-book asset") from exc
    source = {
        "path_lookup_key": "sanmill_checkout",
        "asset_path": SOURCE_ASSET.as_posix(),
        "sanmill_commit": commit,
        "asset_sha256": source_sha256,
        "asset_schema_version": book.get("schemaVersion"),
        "asset_variant": book.get("variant"),
        "asset_symmetry": book.get("symmetry"),
    }
    return book, source


def _parse_board_nodes(board_field: str) -> list[str]:
    groups = board_field.split("/")
    if len(groups) != 3 or any(len(group) != 8 for group in groups):
        raise OracleCorpusError("Oracle board field must contain three 8-node rings")
    nodes = list("".join(groups))
    if any(piece not in "*@O" for piece in nodes):
        raise OracleCorpusError("Oracle board field contains an unknown piece")
    return nodes


def _nmm_board(nodes: Sequence[str]) -> str:
    return "".join(
        "W" if nodes[node] == "O" else "B" if nodes[node] == "@" else "."
        for node in _NODE_FOR_NMM_POSITION
    )


def _move_base(move: Mapping[str, Any]) -> str:
    source = move.get("from")
    target = move.get("to")
    return str(target) if source is None else f"{source}-{target}"


def _oracle_move_matches(move: Mapping[str, Any], notation: str) -> bool:
    notation = notation.strip()
    if not notation or notation.startswith("x"):
        return False
    base, separator, capture = notation.partition("x")
    if _move_base(move) != base:
        return False
    if separator:
        return move.get("capture") == capture
    return True


def project_oracle_source(
    raw_key: str,
    oracle_moves: Sequence[str],
) -> OracleProjection:
    """Project one staged Sanmill Oracle key without losing removal semantics."""
    if not isinstance(raw_key, str) or not raw_key:
        raise OracleCorpusError("Oracle key must be non-empty text")
    if (
        not isinstance(oracle_moves, Sequence)
        or isinstance(oracle_moves, (str, bytes))
        or not oracle_moves
        or any(not isinstance(move, str) or not move for move in oracle_moves)
    ):
        raise OracleCorpusError("Oracle moves must be a non-empty text array")
    fields = raw_key.split()
    if len(fields) < 18 or fields[-1] != "ids:nodes":
        raise OracleCorpusError("unsupported Sanmill Oracle key format")
    if fields[1] not in ("w", "b") or fields[2] != "p":
        raise OracleCorpusError("this corpus supports placement-phase Oracle keys only")
    action = fields[3]
    if action not in ("p", "r"):
        raise OracleCorpusError(f"unsupported staged Oracle action: {action}")

    nodes = _parse_board_nodes(fields[0])
    board_text = _nmm_board(nodes)
    try:
        on_white, hand_white, on_black, hand_black = map(int, fields[4:8])
    except ValueError as exc:
        raise OracleCorpusError("Oracle piece counts must be integers") from exc
    if not all(0 <= value <= 9 for value in (on_white, hand_white, on_black, hand_black)):
        raise OracleCorpusError("Oracle piece counts must be in the standard 0..9 range")
    if board_text.count("W") != on_white or board_text.count("B") != on_black:
        raise OracleCorpusError("Oracle board and on-board counts disagree")
    placed_white, placed_black = 9 - hand_white, 9 - hand_black
    if placed_white < on_white or placed_black < on_black:
        raise OracleCorpusError("Oracle placed counts are smaller than board counts")
    turn = fields[1].upper()
    direct_fen = f"{board_text}|{turn}|{placed_white}|{placed_black}"
    board = BoardState.from_fen_string(direct_fen)
    if board.to_fen_string() != direct_fen:
        raise OracleCorpusError("NMM projection does not round-trip")

    if action == "p":
        expected_turn = "W" if placed_white == placed_black else "B"
        if placed_white - placed_black not in (0, 1) or turn != expected_turn:
            raise OracleCorpusError("stable placement key has inconsistent turn parity")
        legal = get_all_legal_moves(board)
        legal_oracle_moves = tuple(
            notation
            for notation in oracle_moves
            if any(_oracle_move_matches(move, notation) for move in legal)
        )
        illegal_oracle_moves = tuple(
            notation for notation in oracle_moves if notation not in legal_oracle_moves
        )
        return OracleProjection(
            raw_key=raw_key,
            raw_key_sha256=hashlib.sha256(raw_key.encode("utf-8")).hexdigest(),
            oracle_moves=tuple(oracle_moves),
            legal_oracle_moves=legal_oracle_moves,
            illegal_oracle_moves=illegal_oracle_moves,
            action=action,
            direct_fen=direct_fen,
            stable_fen=direct_fen,
        )

    expected_actor = "B" if placed_white == placed_black else "W"
    if placed_white - placed_black not in (0, 1) or turn != expected_actor:
        raise OracleCorpusError("pending-removal key has inconsistent actor parity")
    if len(oracle_moves) != 1 or not oracle_moves[0].startswith("x"):
        raise OracleCorpusError("pending removal must have one explicit Oracle capture")
    capture = oracle_moves[0][1:]
    if capture not in POSITIONS:
        raise OracleCorpusError(f"unknown Oracle capture coordinate: {capture}")
    opponent = "B" if turn == "W" else "W"
    if board.positions[capture] != opponent:
        raise OracleCorpusError("Oracle removal does not target an opponent piece")
    if capture not in board.legal_captures(turn):
        raise OracleCorpusError("Oracle removal is not legal under NMM capture rules")
    stable_positions = dict(board.positions)
    stable_positions[capture] = ""
    stable_board = "".join(stable_positions[position] or "." for position in POSITIONS)
    next_turn = "B" if turn == "W" else "W"
    stable_fen = f"{stable_board}|{next_turn}|{placed_white}|{placed_black}"
    stable = BoardState.from_fen_string(stable_fen)
    if stable.to_fen_string() != stable_fen:
        raise OracleCorpusError("post-removal projection does not round-trip")
    return OracleProjection(
        raw_key=raw_key,
        raw_key_sha256=hashlib.sha256(raw_key.encode("utf-8")).hexdigest(),
        oracle_moves=tuple(oracle_moves),
        legal_oracle_moves=tuple(oracle_moves),
        illegal_oracle_moves=(),
        action=action,
        direct_fen=None,
        stable_fen=stable_fen,
    )


def _transform_board(board_text: str, matrix: tuple[int, int, int, int]) -> str:
    result = ["?"] * len(POSITIONS)
    a, b, c, d = matrix
    for old_index, position in enumerate(POSITIONS):
        x, y = _POSITION_COORDS[position]
        transformed = _COORD_POSITION[(a * x + b * y, c * x + d * y)]
        result[_POSITION_INDEX[transformed]] = board_text[old_index]
    return "".join(result)


def ring16_canonical_fen(fen: str) -> str:
    """Canonicalize an NMM FEN under D4 plus abstract inner/outer swap."""
    board_text, turn, placed_white, placed_black = fen.split("|")
    if len(board_text) != len(POSITIONS):
        raise OracleCorpusError("NMM FEN board must contain 24 positions")
    swapped = board_text[16:24] + board_text[8:16] + board_text[0:8]
    canonical_board = min(
        _transform_board(candidate, matrix)
        for candidate in (board_text, swapped)
        for matrix in _D4
    )
    return f"{canonical_board}|{turn}|{placed_white}|{placed_black}"


def _book_token_matches(move: Mapping[str, Any], token: str) -> bool:
    token = token.strip()
    if not token or token.startswith("x"):
        return False
    base, separator, capture = token.partition("x")
    if _move_base(move) != base:
        return False
    return not separator or move.get("capture") == capture


def audit_named_line_trajectories(openings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exhaustively replay omitted-capture named lines and index their orbits."""
    all_orbits: set[str] = set()
    early_orbits: set[str] = set()
    failures: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    terminal_endpoint_ids: list[str] = []

    for opening in openings:
        opening_id = str(opening.get("id", ""))
        raw_moves = opening.get("lineMoves", "")
        if isinstance(raw_moves, str):
            tokens = raw_moves.split()
        elif isinstance(raw_moves, Sequence):
            tokens = [str(token) for token in raw_moves]
        else:
            tokens = []
        states = {BoardState.new_game().to_fen_string(): BoardState.new_game()}
        root_orbit = ring16_canonical_fen(next(iter(states)))
        all_orbits.add(root_orbit)
        early_orbits.add(root_orbit)
        failed_at: int | None = None
        for ply, token in enumerate(tokens, 1):
            next_states: dict[str, BoardState] = {}
            for board in states.values():
                if is_terminal(board)[0]:
                    continue
                for move in get_all_legal_moves(board):
                    if not _book_token_matches(move, token):
                        continue
                    next_board = board.apply_move(move)
                    next_states[next_board.to_fen_string()] = next_board
            if not next_states:
                failed_at = ply
                states = {}
                break
            states = next_states
            for fen in states:
                orbit = ring16_canonical_fen(fen)
                all_orbits.add(orbit)
                if ply <= 8:
                    early_orbits.add(orbit)
        if failed_at is not None:
            failures.append({"opening_id": opening_id, "failed_at_ply": failed_at})
            continue
        endpoint_count = len(states)
        if endpoint_count > 1:
            ambiguous.append(
                {"opening_id": opening_id, "endpoint_count": endpoint_count}
            )
        if any(is_terminal(board)[0] for board in states.values()):
            terminal_endpoint_ids.append(opening_id)

    if len(openings) != 107:
        raise OracleCorpusError(f"expected 107 named lines, found {len(openings)}")
    endpoint_counts = [item["endpoint_count"] for item in ambiguous]
    summary = {
        "named_line_count": len(openings),
        "replay_success_count": len(openings) - len(failures),
        "replay_failures": failures,
        "ambiguous_line_count": len(ambiguous),
        "ambiguous_endpoint_count_min": min(endpoint_counts) if endpoint_counts else None,
        "ambiguous_endpoint_count_max": max(endpoint_counts) if endpoint_counts else None,
        "terminal_endpoint_opening_ids": sorted(terminal_endpoint_ids),
        "trajectory_ring16_orbit_count": len(all_orbits),
        "first_eight_plies_ring16_orbit_count": len(early_orbits),
        "_all_orbits": all_orbits,
        "_early_orbits": early_orbits,
    }
    expected = (1, 49, 2, 42)
    actual = (
        len(failures),
        len(ambiguous),
        summary["ambiguous_endpoint_count_min"],
        summary["ambiguous_endpoint_count_max"],
    )
    if actual != expected:
        raise OracleCorpusError(
            f"named-line replay audit changed: expected {expected}, got {actual}"
        )
    return summary


def _entry_for_group(
    index: int,
    fen: str,
    sources: Sequence[OracleProjection],
    asset_directory: PurePosixPath,
    named_audit: Mapping[str, Any],
) -> dict[str, Any]:
    board = BoardState.from_fen_string(fen)
    terminal, _winner = is_terminal(board)
    legal_moves = get_all_legal_moves(board)
    if terminal or not legal_moves or board.phase != "place":
        raise OracleCorpusError("selected Oracle start is not a playable placement state")
    orbit = ring16_canonical_fen(fen)
    return {
        "index": index,
        "fen": fen,
        "ring16_canonical_fen": orbit,
        "side_to_move": board.turn,
        "phase": board.phase,
        "pieces": {
            "white_placed": board.pieces_placed["W"],
            "black_placed": board.pieces_placed["B"],
            "white_on_board": board.pieces_on_board["W"],
            "black_on_board": board.pieces_on_board["B"],
            "total_placed": board.pieces_placed["W"] + board.pieces_placed["B"],
        },
        "legal_move_count": len(legal_moves),
        "terminal": False,
        "named_line_overlap": {
            "any_trajectory": orbit in named_audit["_all_orbits"],
            "first_eight_plies": orbit in named_audit["_early_orbits"],
        },
        "sources": [source.source_record() for source in sources],
        "review_png": (
            asset_directory / "positions" / f"oracle-{index:03d}.png"
        ).as_posix(),
    }


def build_corpus_payload(
    book: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    review_asset_directory: str | PurePosixPath,
) -> dict[str, Any]:
    """Build the 106-start owner-reviewed payload from the pinned asset."""
    if (
        book.get("schemaVersion") != 1
        or book.get("variant") != "nmm"
        or book.get("symmetry") != "ring16"
    ):
        raise OracleCorpusError("unexpected Sanmill opening-book contract")
    oracle = book.get("oracle")
    openings = book.get("openings")
    if not isinstance(oracle, Mapping) or not isinstance(openings, Sequence):
        raise OracleCorpusError("Sanmill asset lacks Oracle or opening data")
    projections = [
        project_oracle_source(raw_key, moves)
        for raw_key, moves in sorted(oracle.items())
    ]
    if len(projections) != 110:
        raise OracleCorpusError(f"expected 110 raw Oracle keys, found {len(projections)}")
    direct = [projection for projection in projections if projection.direct_fen]
    removals = [projection for projection in projections if not projection.direct_fen]
    if len(direct) != 108 or len(removals) != 2:
        raise OracleCorpusError("expected 108 stable placement keys and 2 pending removals")
    invalid_source_moves = [
        {
            "raw_key_sha256": projection.raw_key_sha256,
            "action": projection.action,
            "moves": list(projection.illegal_oracle_moves),
        }
        for projection in projections
        if projection.illegal_oracle_moves
    ]
    if invalid_source_moves != [EXPECTED_INVALID_ORACLE_MOVE]:
        raise OracleCorpusError(
            "Oracle move-legality audit changed: expected exactly the reviewed "
            f"c3 defect, got {invalid_source_moves}"
        )

    groups: dict[str, list[OracleProjection]] = {}
    for projection in direct:
        assert projection.direct_fen is not None
        groups.setdefault(projection.direct_fen, []).append(projection)
    if len(groups) != 107:
        raise OracleCorpusError(
            f"expected 107 exact stable placement FENs, found {len(groups)}"
        )
    named_audit = audit_named_line_trajectories(openings)
    asset_directory = PurePosixPath(review_asset_directory)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            sum(map(int, item[0].split("|")[2:4])),
            item[0],
        ),
    )
    exclusion_index = int(OWNER_REVIEW_EXCLUSION["original_review_index"])
    exclusion_fen = str(OWNER_REVIEW_EXCLUSION["fen"])
    if ordered_groups[exclusion_index - 1][0] != exclusion_fen:
        raise OracleCorpusError("owner-review exclusion no longer has index 101")
    excluded_sources = ordered_groups[exclusion_index - 1][1]
    excluded_source_hashes = [source.raw_key_sha256 for source in excluded_sources]
    if excluded_source_hashes != [OWNER_REVIEW_EXCLUSION["raw_key_sha256"]]:
        raise OracleCorpusError("owner-review exclusion source identity changed")
    selected_groups = [
        (fen, sources) for fen, sources in ordered_groups if fen != exclusion_fen
    ]
    entries = [
        _entry_for_group(index, fen, sources, asset_directory, named_audit)
        for index, (fen, sources) in enumerate(selected_groups, 1)
    ]
    orbits = [entry["ring16_canonical_fen"] for entry in entries]
    if len(set(orbits)) != len(entries):
        raise OracleCorpusError("selected stable positions are not ring16 unique")

    entry_by_fen = {entry["fen"]: entry for entry in entries}
    entry_by_orbit = {entry["ring16_canonical_fen"]: entry for entry in entries}
    removal_evidence: list[dict[str, Any]] = []
    for removal in removals:
        exact = entry_by_fen.get(removal.stable_fen)
        if exact is not None:
            relation = "exact_duplicate"
            target = exact
        else:
            target = entry_by_orbit.get(ring16_canonical_fen(removal.stable_fen))
            relation = "ring16_duplicate"
        if target is None:
            raise OracleCorpusError(
                "pending-removal successor introduces an unreviewed start orbit"
            )
        removal_evidence.append(
            {
                **removal.source_record(),
                "stable_successor_fen": removal.stable_fen,
                "relation_to_selected_corpus": relation,
                "selected_entry_index": target["index"],
            }
        )

    start_positions = [entry["fen"] for entry in entries]
    start_positions_sha256 = canonical_sha256(start_positions)
    direct_duplicate_groups = [
        {
            "fen": fen,
            "source_key_sha256": [source.raw_key_sha256 for source in sources],
            "differing_raw_fields": [
                index
                for index, values in enumerate(
                    zip(*(source.raw_key.split() for source in sources))
                )
                if len(set(values)) > 1
            ],
        }
        for fen, sources in groups.items()
        if len(sources) > 1
    ]
    if len(direct_duplicate_groups) != 1:
        raise OracleCorpusError("expected one counter-only direct projection duplicate")

    named_summary = {
        key: value
        for key, value in named_audit.items()
        if not key.startswith("_")
    }
    overlap_any = sum(
        entry["named_line_overlap"]["any_trajectory"] for entry in entries
    )
    overlap_early = sum(
        entry["named_line_overlap"]["first_eight_plies"] for entry in entries
    )
    named_summary.update(
        {
            "selected_overlap_any_trajectory": overlap_any,
            "selected_overlap_first_eight_plies": overlap_early,
        }
    )
    owner_exclusion = {
        "original_review_index": exclusion_index,
        "disposition": "remove",
        "fen": exclusion_fen,
        "ring16_canonical_fen": ring16_canonical_fen(exclusion_fen),
        "sources": [source.source_record() for source in excluded_sources],
    }
    return {
        "schema_version": CORPUS_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "status": "owner_review_complete_not_frozen",
        "owner_review": {
            "status": "complete",
            "decision_date": OWNER_REVIEW_DECISION_DATE,
            "reviewed_start_count": len(groups),
            "accepted_start_count": len(entries),
            "excluded_starts": [owner_exclusion],
        },
        "source": dict(source),
        "projection_contract": {
            "board_mapping": "Sanmill node-id Mill FEN via NMM_POSITION_ORDER_NODES",
            "stable_start_action": "p",
            "pending_removal_disposition": (
                "apply the sole Oracle removal for provenance, then require the "
                "successor to match an already selected ring16 orbit"
            ),
            "selection_order": (
                "total placed pieces ascending, then NMM FEN; apply the "
                "recorded owner-review exclusion; reindex contiguously"
            ),
            "symmetry": "D4 x inner/outer-ring swap (ring16)",
        },
        "automated_audit": {
            "raw_oracle_keys": len(projections),
            "stable_action_p_keys": len(direct),
            "pending_action_r_keys": len(removals),
            "exact_action_p_fens": len(groups),
            "owner_excluded_starts": 1,
            "selected_ring16_orbits": len(set(orbits)),
            "playable_starts": len(entries),
            "phase_counts": {"placement": len(entries), "movement": 0, "flying": 0},
            "total_placed_min": min(entry["pieces"]["total_placed"] for entry in entries),
            "total_placed_max": max(entry["pieces"]["total_placed"] for entry in entries),
            "terminal_starts": 0,
            "starts_without_legal_moves": 0,
            "oracle_move_legality": {
                "valid_source_moves": sum(
                    len(projection.legal_oracle_moves) for projection in projections
                ),
                "invalid_source_moves": sum(
                    len(projection.illegal_oracle_moves) for projection in projections
                ),
                "invalid_records": invalid_source_moves,
                "disposition": (
                    "the sole invalid recommendation belongs to the start excluded "
                    "by owner review; source moves remain provenance only"
                ),
            },
            "named_line_trajectory": named_summary,
        },
        "projection_evidence": {
            "direct_duplicate_groups": direct_duplicate_groups,
            "pending_removal_successors": removal_evidence,
        },
        "start_positions": start_positions,
        "start_positions_sha256": start_positions_sha256,
        "entries": entries,
    }


@lru_cache(maxsize=1)
def _load_fonts() -> dict[str, Any]:
    from PIL import ImageFont

    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    regular_path = windows / "segoeui.ttf"
    bold_path = windows / "segoeuib.ttf"
    if not regular_path.is_file() or not bold_path.is_file():
        raise OracleCorpusError("Segoe UI fonts are required for review rendering")
    return {
        "title": ImageFont.truetype(str(bold_path), 29),
        "heading": ImageFont.truetype(str(bold_path), 18),
        "body": ImageFont.truetype(str(regular_path), 16),
        "small": ImageFont.truetype(str(regular_path), 13),
        "node": ImageFont.truetype(str(bold_path), 13),
        "sheet": ImageFont.truetype(str(bold_path), 25),
    }


def render_position_image(
    entry: Mapping[str, Any],
    *,
    total: int,
    start_positions_sha256: str,
):
    """Render one deterministic human-review board panel."""
    from PIL import Image, ImageDraw

    fonts = _load_fonts()
    image = Image.new("RGB", (720, 840), "#f5f0e7")
    draw = ImageDraw.Draw(image)
    index = int(entry["index"])
    board = BoardState.from_fen_string(str(entry["fen"]))
    pieces = entry["pieces"]
    title_prefix = str(entry.get("review_title_prefix", "Oracle start"))
    draw.text(
        (28, 20),
        f"{title_prefix} {index:03d}/{total:03d}",
        fill="#2b2722",
        font=fonts["title"],
    )
    meta = (
        f"turn {board.turn}   placed W/B {pieces['white_placed']}/{pieces['black_placed']}"
        f"   on-board {pieces['white_on_board']}/{pieces['black_on_board']}"
        f"   legal {entry['legal_move_count']}"
    )
    draw.text((29, 61), meta, fill="#3f3931", font=fonts["body"])
    draw.text((29, 90), str(entry["fen"]), fill="#51493f", font=fonts["small"])
    review_details = entry.get("review_detail_lines")
    if review_details is None:
        source_moves = sorted(
            {
                move
                for source in entry["sources"]
                for move in source["oracle_moves"]
            }
        )
        detail_lines = ["Oracle moves: " + " ".join(source_moves)]
    else:
        detail_lines = [str(line) for line in review_details]
    wrapped_details = [
        wrapped
        for detail in detail_lines
        for wrapped in textwrap.wrap(detail, width=82)
    ]
    for line_number, line in enumerate(wrapped_details[:2]):
        draw.text((29, 116 + line_number * 20), line, fill="#51493f", font=fonts["small"])
    illegal_moves = sorted(
        {
            move
            for source in entry["sources"]
            for move in source["nmm_illegal_oracle_moves"]
        }
    )
    if illegal_moves:
        draw.text(
            (29, 152),
            "SOURCE WARNING: illegal NMM recommendation " + " ".join(illegal_moves),
            fill="#b3261e",
            font=fonts["small"],
        )

    center_x, center_y, scale = 360, 472, 79
    pixels = {
        position: (
            center_x + _POSITION_COORDS[position][0] * scale,
            center_y - _POSITION_COORDS[position][1] * scale,
        )
        for position in POSITIONS
    }
    seen_edges: set[tuple[str, str]] = set()
    for source, targets in ADJACENCY.items():
        for target in targets:
            edge = tuple(sorted((source, target)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            draw.line((pixels[source], pixels[target]), fill="#62594d", width=5)

    for position in POSITIONS:
        x, y = pixels[position]
        piece = board.positions[position]
        if piece == "W":
            draw.ellipse(
                (x - 20, y - 20, x + 20, y + 20),
                fill="#fffdf7",
                outline="#25231f",
                width=4,
            )
        elif piece == "B":
            draw.ellipse(
                (x - 20, y - 20, x + 20, y + 20),
                fill="#20242a",
                outline="#050607",
                width=4,
            )
        else:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#b9aa95", outline="#62594d", width=2)
        label_fill = "#a33a2c" if piece else "#4c443a"
        draw.text((x + 10, y - 19), position, fill=label_fill, font=fonts["node"])

    review_footer = entry.get("review_footer")
    if review_footer is None:
        overlap = entry["named_line_overlap"]
        review_footer = (
            "named-line overlap: "
            f"any={'yes' if overlap['any_trajectory'] else 'no'}, "
            f"first-8={'yes' if overlap['first_eight_plies'] else 'no'}"
        )
    draw.text((29, 778), str(review_footer), fill="#3f3931", font=fonts["small"])
    draw.text(
        (29, 804),
        f"start_positions_sha256 {start_positions_sha256[:20]}…",
        fill="#6b6257",
        font=fonts["small"],
    )
    return image


def _save_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, compress_level=9)


def render_review_assets(
    payload: Mapping[str, Any],
    asset_root: str | Path,
) -> dict[str, Any]:
    """Render individual panels and 4x3 contact sheets, returning a manifest."""
    from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

    root = Path(asset_root)
    if root.exists():
        raise FileExistsError(f"review asset directory exists: {root}")
    root.mkdir(parents=True)
    entries = list(payload["entries"])
    start_hash = str(payload["start_positions_sha256"])
    rendered: list[tuple[Mapping[str, Any], Any, Path]] = []
    individual_manifest: list[dict[str, Any]] = []
    image_prefix = str(payload.get("image_prefix", "oracle"))
    evaluation_id = str(payload.get("evaluation_id", EVALUATION_ID))
    sheet_title = str(payload.get("review_sheet_title", "Oracle starts"))
    for entry in entries:
        image = render_position_image(
            entry,
            total=len(entries),
            start_positions_sha256=start_hash,
        )
        relative = Path("positions") / (
            f"{image_prefix}-{int(entry['index']):03d}.png"
        )
        target = root / relative
        _save_png(image, target)
        rendered.append((entry, image, relative))
        individual_manifest.append(
            {
                "index": entry["index"],
                "path": relative.as_posix(),
                "sha256": _sha256_file(target),
                "width": image.width,
                "height": image.height,
            }
        )

    contact_manifest: list[dict[str, Any]] = []
    per_sheet = 12
    thumb_size = (360, 420)
    for offset in range(0, len(rendered), per_sheet):
        page = rendered[offset : offset + per_sheet]
        sheet_number = offset // per_sheet + 1
        sheet = Image.new("RGB", (1440, 1310), "#ddd5c8")
        sheet_draw = ImageDraw.Draw(sheet)
        try:
            sheet_font = _load_fonts()["sheet"]
        except OracleCorpusError:
            sheet_font = ImageFont.load_default()
        first_index = int(page[0][0]["index"])
        last_index = int(page[-1][0]["index"])
        sheet_draw.text(
            (24, 9),
            f"{evaluation_id} — {sheet_title} "
            f"{first_index:03d}–{last_index:03d}",
            fill="#2b2722",
            font=sheet_font,
        )
        for local_index, (_entry, image, _relative) in enumerate(page):
            thumbnail = image.resize(thumb_size, Image.Resampling.LANCZOS)
            x = (local_index % 4) * thumb_size[0]
            y = 50 + (local_index // 4) * thumb_size[1]
            sheet.paste(thumbnail, (x, y))
        relative = Path("contact-sheets") / f"sheet-{sheet_number:02d}.png"
        target = root / relative
        _save_png(sheet, target)
        contact_manifest.append(
            {
                "sheet": sheet_number,
                "first_index": first_index,
                "last_index": last_index,
                "path": relative.as_posix(),
                "sha256": _sha256_file(target),
                "width": sheet.width,
                "height": sheet.height,
            }
        )

    manifest_body = {
        "schema_version": REVIEW_MANIFEST_SCHEMA,
        "evaluation_id": evaluation_id,
        "start_positions_sha256": start_hash,
        "renderer": {
            "name": "learned_ai.evaluation.oracle_corpus",
            "pillow_version": pillow_version,
            "font_family": "Segoe UI",
            "individual_size": [720, 840],
            "contact_sheet_layout": "4 columns x 3 rows",
        },
        "individual_images": individual_manifest,
        "contact_sheets": contact_manifest,
    }
    manifest = {
        **manifest_body,
        "manifest_identity": canonical_sha256(manifest_body),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(_json_document(manifest).encode("utf-8"))
    logical_directory = PurePosixPath(str(entries[0]["review_png"])).parent.parent
    return {
        "directory": logical_directory.as_posix(),
        "manifest": "manifest.json",
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_identity": manifest["manifest_identity"],
        "individual_image_count": len(individual_manifest),
        "contact_sheet_count": len(contact_manifest),
    }


def validate_corpus_artifact(payload: Mapping[str, Any]) -> dict[str, int]:
    """Independently validate the persisted corpus without the Sanmill checkout."""
    if payload.get("schema_version") != CORPUS_SCHEMA:
        raise OracleCorpusError("unsupported corpus schema")
    entries = payload.get("entries")
    starts = payload.get("start_positions")
    if not isinstance(entries, Sequence) or not isinstance(starts, Sequence):
        raise OracleCorpusError("corpus entries and starts must be arrays")
    if len(entries) != 106 or len(starts) != 106:
        raise OracleCorpusError("the owner-reviewed corpus must contain 106 starts")
    if payload.get("status") != "owner_review_complete_not_frozen":
        raise OracleCorpusError("owner-reviewed corpus has the wrong lifecycle status")
    if list(starts) != [entry.get("fen") for entry in entries]:
        raise OracleCorpusError("entry order differs from start_positions")
    if canonical_sha256(list(starts)) != payload.get("start_positions_sha256"):
        raise OracleCorpusError("start_positions_sha256 mismatch")
    starts_artifact = payload.get("start_positions_artifact")
    if not isinstance(starts_artifact, Mapping):
        raise OracleCorpusError("start_positions_artifact metadata is missing")
    relative_starts_path = PurePosixPath(str(starts_artifact.get("path", "")))
    if (
        relative_starts_path.is_absolute()
        or ".." in relative_starts_path.parts
        or relative_starts_path.as_posix() in ("", ".")
    ):
        raise OracleCorpusError("start_positions_artifact path must be relative")
    starts_file_sha256 = hashlib.sha256(
        _json_document(list(starts)).encode("utf-8")
    ).hexdigest()
    if starts_artifact.get("file_sha256") != starts_file_sha256:
        raise OracleCorpusError("start-position list file SHA-256 mismatch")
    body = dict(payload)
    corpus_identity = body.pop("corpus_identity", None)
    if corpus_identity is not None and canonical_sha256(body) != corpus_identity:
        raise OracleCorpusError("corpus identity mismatch")

    exact: set[str] = set()
    orbits: set[str] = set()
    direct_source_keys: set[str] = set()
    for expected_index, entry in enumerate(entries, 1):
        if entry.get("index") != expected_index:
            raise OracleCorpusError("corpus indices are not contiguous")
        fen = str(entry["fen"])
        board = BoardState.from_fen_string(fen)
        if board.to_fen_string() != fen:
            raise OracleCorpusError(f"entry {expected_index} does not round-trip")
        if board.phase != "place" or is_terminal(board)[0] or not get_all_legal_moves(board):
            raise OracleCorpusError(f"entry {expected_index} is not playable placement")
        orbit = ring16_canonical_fen(fen)
        if entry.get("ring16_canonical_fen") != orbit:
            raise OracleCorpusError(f"entry {expected_index} has wrong ring16 identity")
        exact.add(fen)
        orbits.add(orbit)
        for source in entry.get("sources", []):
            projection = project_oracle_source(source["raw_key"], source["oracle_moves"])
            if projection.action != "p" or projection.direct_fen != fen:
                raise OracleCorpusError("direct source does not reproduce its entry")
            if source != projection.source_record():
                raise OracleCorpusError("direct source audit fields do not reproduce")
            direct_source_keys.add(projection.raw_key)
    if len(exact) != 106 or len(orbits) != 106 or len(direct_source_keys) != 107:
        raise OracleCorpusError("exact, ring16, or direct-source uniqueness failed")

    owner_review = payload.get("owner_review")
    if not isinstance(owner_review, Mapping):
        raise OracleCorpusError("owner-review decision is missing")
    if (
        owner_review.get("status") != "complete"
        or owner_review.get("decision_date") != OWNER_REVIEW_DECISION_DATE
        or owner_review.get("reviewed_start_count") != 107
        or owner_review.get("accepted_start_count") != 106
    ):
        raise OracleCorpusError("owner-review decision metadata is inconsistent")
    excluded_items = owner_review.get("excluded_starts")
    if not isinstance(excluded_items, Sequence) or len(excluded_items) != 1:
        raise OracleCorpusError("owner review must contain exactly one exclusion")
    excluded_item = excluded_items[0]
    if not isinstance(excluded_item, Mapping):
        raise OracleCorpusError("owner-review exclusion is malformed")
    for key in ("original_review_index", "fen", "ring16_canonical_fen"):
        if excluded_item.get(key) != OWNER_REVIEW_EXCLUSION[key]:
            raise OracleCorpusError(f"owner-review exclusion changed field {key}")
    if excluded_item.get("disposition") != "remove":
        raise OracleCorpusError("owner-review exclusion disposition changed")
    excluded_fen = str(excluded_item["fen"])
    if ring16_canonical_fen(excluded_fen) != excluded_item[
        "ring16_canonical_fen"
    ]:
        raise OracleCorpusError("owner-review exclusion has wrong ring16 identity")
    if excluded_fen in exact:
        raise OracleCorpusError("owner-excluded start remains in the selected corpus")
    excluded_source_keys: set[str] = set()
    excluded_sources = excluded_item.get("sources")
    if not isinstance(excluded_sources, Sequence) or len(excluded_sources) != 1:
        raise OracleCorpusError("owner-review exclusion source evidence is incomplete")
    for source in excluded_sources:
        projection = project_oracle_source(source["raw_key"], source["oracle_moves"])
        if projection.action != "p" or projection.direct_fen != excluded_fen:
            raise OracleCorpusError("owner-review exclusion does not reproduce")
        if source != projection.source_record():
            raise OracleCorpusError("owner-review exclusion source fields changed")
        if projection.raw_key_sha256 != OWNER_REVIEW_EXCLUSION["raw_key_sha256"]:
            raise OracleCorpusError("owner-review exclusion source identity changed")
        excluded_source_keys.add(projection.raw_key)
    if direct_source_keys & excluded_source_keys:
        raise OracleCorpusError("owner-excluded source remains selected")

    removal_keys: set[str] = set()
    removal_items = payload.get("projection_evidence", {}).get(
        "pending_removal_successors", []
    )
    for item in removal_items:
        projection = project_oracle_source(item["raw_key"], item["oracle_moves"])
        if projection.action != "r" or projection.stable_fen != item.get(
            "stable_successor_fen"
        ):
            raise OracleCorpusError("pending-removal evidence does not reproduce")
        for key, value in projection.source_record().items():
            if item.get(key) != value:
                raise OracleCorpusError(
                    "pending-removal source audit fields do not reproduce"
                )
        target_index = item.get("selected_entry_index")
        if not isinstance(target_index, int) or not 1 <= target_index <= len(entries):
            raise OracleCorpusError("pending-removal target index is invalid")
        target = entries[target_index - 1]
        relation = item.get("relation_to_selected_corpus")
        if relation == "exact_duplicate":
            matches = projection.stable_fen == target["fen"]
        elif relation == "ring16_duplicate":
            matches = ring16_canonical_fen(projection.stable_fen) == target[
                "ring16_canonical_fen"
            ]
        else:
            matches = False
        if not matches:
            raise OracleCorpusError("pending-removal relation is incorrect")
        removal_keys.add(projection.raw_key)
    selected_or_excluded = direct_source_keys | excluded_source_keys
    if len(removal_keys) != 2 or selected_or_excluded & removal_keys:
        raise OracleCorpusError("pending-removal source coverage is wrong")
    if len(direct_source_keys | excluded_source_keys | removal_keys) != 110:
        raise OracleCorpusError("raw Oracle source coverage is incomplete")
    return {
        "starts": len(starts),
        "ring16_orbits": len(orbits),
        "direct_sources": len(direct_source_keys),
        "excluded_starts": len(excluded_items),
        "pending_removals": len(removal_keys),
    }


def validate_review_manifest(
    asset_root: str | Path,
    *,
    expected_individuals: int = 106,
    expected_sheets: int = 9,
) -> dict[str, int]:
    """Verify every PNG hash and dimension recorded in the review manifest."""
    from PIL import Image

    root = Path(asset_root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleCorpusError("review manifest is unreadable") from exc
    if manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
        raise OracleCorpusError("unsupported review manifest schema")
    body = dict(manifest)
    identity = body.pop("manifest_identity", None)
    if canonical_sha256(body) != identity:
        raise OracleCorpusError("review manifest identity mismatch")
    individuals = manifest.get("individual_images", [])
    sheets = manifest.get("contact_sheets", [])
    if (
        len(individuals) != expected_individuals
        or len(sheets) != expected_sheets
    ):
        raise OracleCorpusError("review image counts are incomplete")
    if [item.get("index") for item in individuals] != list(
        range(1, expected_individuals + 1)
    ):
        raise OracleCorpusError("review image indices are not contiguous")
    for record in [*individuals, *sheets]:
        path = root / record["path"]
        if _sha256_file(path) != record["sha256"]:
            raise OracleCorpusError(f"review image hash mismatch: {record['path']}")
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                raise OracleCorpusError(
                    f"review image dimensions mismatch: {record['path']}"
                )
            image.verify()
    return {"individual_images": len(individuals), "contact_sheets": len(sheets)}


def write_review_package(
    book: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    output_path: str | Path,
    start_positions_path: str | Path,
    asset_root: str | Path,
    review_asset_directory: str | PurePosixPath,
) -> dict[str, Any]:
    """Create an immutable owner-reviewed package; refuse existing targets."""
    output = Path(output_path)
    starts_output = Path(start_positions_path)
    assets = Path(asset_root)
    if output.exists():
        raise FileExistsError(f"corpus output exists: {output}")
    if starts_output.exists():
        raise FileExistsError(f"start-position output exists: {starts_output}")
    if assets.exists():
        raise FileExistsError(f"review asset directory exists: {assets}")
    try:
        starts_relative = starts_output.resolve().relative_to(
            output.parent.resolve()
        )
    except ValueError as exc:
        raise OracleCorpusError(
            "start-position list must be inside the corpus JSON directory"
        ) from exc
    payload = build_corpus_payload(
        book,
        source,
        review_asset_directory=review_asset_directory,
    )
    review_assets = render_review_assets(payload, assets)
    payload["review_assets"] = review_assets
    starts_text = _json_document(payload["start_positions"])
    payload["start_positions_artifact"] = {
        "path": PurePosixPath(*starts_relative.parts).as_posix(),
        "file_sha256": hashlib.sha256(starts_text.encode("utf-8")).hexdigest(),
    }
    payload["corpus_identity"] = canonical_sha256(payload)
    validate_corpus_artifact(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    starts_output.parent.mkdir(parents=True, exist_ok=True)
    starts_output.write_bytes(starts_text.encode("utf-8"))
    output.write_bytes(_json_document(payload).encode("utf-8"))
    return payload
