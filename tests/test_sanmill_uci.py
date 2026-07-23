from __future__ import annotations

from pathlib import Path

import pytest

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_OPENING_BOOK_ORACLE_ENTRIES,
    EXPECTED_OPENING_BOOK_RECOMMENDATIONS,
    EXPECTED_SANMILL_BINARY_SHA256,
    EXPECTED_SANMILL_BINARY_SIZE,
    EXPECTED_SANMILL_LICENSE_SHA256,
    PINNED_SANMILL_COMMIT,
    PINNED_SANMILL_TREE,
    REMOVED_INVALID_ORACLE_KEY_SHA256,
    SanmillBridgeError,
    SanmillUciSession,
    assert_pending_removal_parity,
    assert_stable_legal_parity,
    atomic_move_for_actions,
    inspect_sanmill_installation,
    inspect_sanmill_opening_book,
    parse_debug_outcome,
    parse_search_line,
    strict_contract_record,
    strict_option_values,
    validate_uci_action_token,
)


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"


def test_strict_contract_preserves_the_normal_opening_depth_policy() -> None:
    options = dict(strict_option_values())
    contract = strict_contract_record()

    assert contract["search_command"] == "go nodes <positive-N>"
    assert options["Threads"] == "1"
    assert options["Shuffling"] == "false"
    assert options["SearchShuffleSeed"] == "42"
    assert options["DeveloperMode"] == "false"
    assert options["DrawOnHumanExperience"] == "true"
    assert options["FocusOnBlockingPaths"] == "false"
    assert options["UsePerfectDatabase"] == "false"
    assert options["PatchAvoidTraps"] == "false"
    assert options["PatchMakeTraps"] == "false"
    assert contract["draw_on_human_experience_semantics"]["effective_in_smoke"]
    assert not contract["knowledge_sources"]["opening_book"]["active_in_bridge_smoke"]


@pytest.mark.parametrize(
    "token",
    ["a7", "d6-d5", "xc3"],
)
def test_validate_uci_action_token_accepts_standard_actions(token: str) -> None:
    assert validate_uci_action_token(token) == token


@pytest.mark.parametrize(
    "token",
    ["", "a7 d7", "x", "xh8", "a7-d7-g7", "h8"],
)
def test_validate_uci_action_token_rejects_malformed_actions(token: str) -> None:
    with pytest.raises(SanmillBridgeError):
        validate_uci_action_token(token)


def test_parse_search_line_preserves_semantic_fields() -> None:
    result = parse_search_line(
        "info depth 3 score cp -7 nodes 1234 bestmove d6-d5",
        0.25,
    )

    assert result.semantic_record() == {
        "bestmove": "d6-d5",
        "depth": 3,
        "nodes": 1234,
        "score_kind": "cp",
        "score": -7,
    }
    assert result.elapsed_seconds == 0.25


@pytest.mark.parametrize(
    ("lines", "winner", "reason"),
    [
        (("winner: -1", "outcome_reason: 0"), "none", "ongoing"),
        (("winner: 0", "outcome_reason: 6"), "white", "loseNoLegalMoves"),
        (("winner: 2", "outcome_reason: 5"), "draw", "drawThreefoldRepetition"),
    ],
)
def test_parse_debug_outcome_preserves_sanmill_authority(
    lines: tuple[str, ...],
    winner: str,
    reason: str,
) -> None:
    outcome = parse_debug_outcome(lines)

    assert outcome.winner == winner
    assert outcome.reason == reason
    assert outcome.terminal == (winner != "none")


@pytest.mark.parametrize(
    "lines",
    [
        ("winner: -1",),
        ("winner: -1", "outcome_reason: 5"),
        ("winner: 2", "outcome_reason: 1"),
        ("winner: 0", "outcome_reason: 8"),
        ("winner: 7", "outcome_reason: 0"),
    ],
)
def test_parse_debug_outcome_rejects_missing_or_inconsistent_fields(
    lines: tuple[str, ...],
) -> None:
    with pytest.raises(SanmillBridgeError):
        parse_debug_outcome(lines)


def test_start_position_has_the_same_primary_legal_actions() -> None:
    board = BoardState.new_game()

    moves = assert_stable_legal_parity(board, POSITIONS)

    assert len(moves) == 24


def test_pending_removal_parity_selects_one_atomic_nmm_move() -> None:
    board = BoardState.from_setup(
        {"b6": "W", "d6": "W", "a7": "B", "a4": "B"},
        turn="W",
        phase="place",
    )
    nmm_moves = get_all_legal_moves(board)

    removals = assert_pending_removal_parity(
        nmm_moves,
        "f6",
        ("xa7", "xa4"),
    )
    selected = atomic_move_for_actions(nmm_moves, "f6", "xa7")

    assert removals == ("xa4", "xa7")
    assert selected == {"from": None, "to": "f6", "capture": "a7"}


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_pinned_sanmill_contract_and_book_gate() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    book_gate = inspect_sanmill_opening_book(installation)

    assert installation.commit == PINNED_SANMILL_COMMIT
    assert installation.tree == PINNED_SANMILL_TREE
    assert installation.binary_sha256 == EXPECTED_SANMILL_BINARY_SHA256
    assert installation.binary_size == EXPECTED_SANMILL_BINARY_SIZE
    assert installation.license_sha256 == EXPECTED_SANMILL_LICENSE_SHA256
    assert book_gate.oracle_entries == EXPECTED_OPENING_BOOK_ORACLE_ENTRIES
    assert (
        book_gate.oracle_recommendations
        == EXPECTED_OPENING_BOOK_RECOMMENDATIONS
    )
    assert (
        book_gate.removed_invalid_key_sha256
        == REMOVED_INVALID_ORACLE_KEY_SHA256
    )

    with SanmillUciSession(installation) as session:
        session.new_game()
        session.position_startpos()
        result = session.search_fixed_nodes(100_000)

    # SkillLevel=30 would request depth 30 if the ordinary opening policy were
    # bypassed.  Depth 1 is the pinned non-developer opening-table result.
    assert result.depth == 1
    assert 0 < result.nodes < 100_000
