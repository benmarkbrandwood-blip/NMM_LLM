from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.retained_late_import_heldout_pool import (
    RetainedLateImportPoolError,
    _notation,
    discover_late_imports,
    extract_source_candidates,
    select_independent_records,
    validate_retained_late_import_pool,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN_POOL = ROOT / (
    "docs/experiments/"
    "sanmill-retained-v3-v4-late-import-heldout-pool-v1.json"
)


def _source_game_bytes() -> bytes:
    board = BoardState.new_game()
    moves = []
    for index in range(13):
        legal = sorted(get_all_legal_moves(board), key=_notation)
        move = legal[0]
        moves.append(
            {
                "turn": index + 1,
                "color": board.turn,
                "type": "place",
                "from": move["from"],
                "to": move["to"],
                "capture": move["capture"],
                "notation": _notation(move),
                "board_fen_before": board.to_fen_string(),
            }
        )
        board = board.apply_move(move)
    return json.dumps(
        {
            "session_id": "ml-test-source",
            "source": "playok",
            "source_type": "human_vs_human",
            "winner": "W",
            "moves": moves,
        }
    ).encode("utf-8")


def test_discover_late_imports_uses_normalized_humandb_membership() -> None:
    manifest = {
        "ml1": "2026-07-18T00:00:00",
        "ml2": "2026-07-20T00:00:00",
        "ml3": "2026-07-20T00:00:01",
    }

    assert discover_late_imports(manifest, {"ml1"}) == ["ml2", "ml3"]


def test_extract_source_candidates_replays_without_using_outcome() -> None:
    source, candidates = extract_source_candidates(
        _source_game_bytes(),
        relative_path="data/human_games/human_ml-test-source.jsonl",
        imported_at="2026-07-20T00:00:00",
    )

    assert source["source"] == "playok"
    assert source["source_type"] == "human_vs_human"
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["logical_ply_count"] == 12
    assert candidate["phase"] == "placement"
    assert len(candidate["logical_turns"]) == 12
    assert "winner" not in candidate


def test_extract_source_candidates_rejects_changed_history() -> None:
    payload = json.loads(_source_game_bytes())
    payload["moves"][8]["board_fen_before"] = BoardState.new_game().to_fen_string()

    with pytest.raises(RetainedLateImportPoolError, match="board history"):
        extract_source_candidates(
            json.dumps(payload).encode("utf-8"),
            relative_path="data/human_games/changed.jsonl",
            imported_at="2026-07-20T00:00:00",
        )


def _candidate(game: str, phase: str, suffix: str) -> dict:
    return {
        "source": {"source_identity": game},
        "phase": phase,
        "fen": f"fen-{suffix}",
        "ring16_canonical_fen": f"ring-{suffix}",
        "candidate_identity": f"candidate-{suffix}",
        "selection_rank": f"rank-{suffix}",
    }


def test_selection_reserves_flying_and_keeps_one_unique_source() -> None:
    eligible = [
        _candidate("f", "placement", "f-p"),
        _candidate("f", "movement", "f-m"),
        _candidate("f", "flying", "f-f"),
        _candidate("p", "placement", "p"),
        _candidate("m", "movement", "m"),
        _candidate("d1", "placement", "d1-p"),
        _candidate("d1", "movement", "d1-m"),
        _candidate("d2", "placement", "d2-p"),
        _candidate("d2", "movement", "d2-m"),
    ]

    records, summary = select_independent_records(eligible)

    assert len(records) == 5
    assert len({row["source"]["source_identity"] for row in records}) == 5
    assert len({row["ring16_canonical_fen"] for row in records}) == 5
    flying = [row for row in records if row["source"]["source_identity"] == "f"]
    assert [row["phase"] for row in flying] == ["flying"]
    assert summary["flying_sources_reserved_for_flying"] == 1


def test_frozen_pool_validates_and_exposes_nested_precision_limits() -> None:
    payload = json.loads(FROZEN_POOL.read_bytes())

    records = validate_retained_late_import_pool(payload)

    assert payload["pool_identity"] == (
        "2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7"
    )
    assert len(records) == 361
    profiles = {
        item["target_starts"]: item for item in payload["nested_precision_prefixes"]
    }
    assert profiles[142]["available"] is True
    assert profiles[253]["available"] is True
    assert profiles[568]["available"] is False


def test_frozen_pool_rejects_relabelled_candidate_selection() -> None:
    payload = json.loads(FROZEN_POOL.read_bytes())
    changed = copy.deepcopy(payload)
    changed["selection_contract"]["candidate_policy_loaded"] = True
    body = {key: value for key, value in changed.items() if key != "pool_identity"}
    changed["pool_identity"] = canonical_sha256(body)

    with pytest.raises(RetainedLateImportPoolError, match="loaded a candidate"):
        validate_retained_late_import_pool(changed)
