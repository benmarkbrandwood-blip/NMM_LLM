from __future__ import annotations

import json
from collections.abc import Callable
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
    SanmillProtocolError,
    SanmillUciSession,
    assert_pending_removal_parity,
    assert_stable_legal_parity,
    atomic_move_for_actions,
    inspect_sanmill_installation,
    inspect_sanmill_opening_book,
    parse_debug_outcome,
    parse_logical_turn_line,
    parse_protocol_error,
    parse_search_line,
    parse_state_json_line,
    strict_contract_record,
    strict_option_values,
    validate_uci_action_token,
)


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"


def test_strict_contract_preserves_the_normal_opening_depth_policy() -> None:
    options = dict(strict_option_values())
    contract = strict_contract_record()

    assert contract["search_command"].startswith("go logical nodes")
    assert contract["state_command"] == "statejson"
    assert options["Threads"] == "1"
    assert options["StrictFailurePolicy"] == "true"
    assert options["IDSEnabled"] == "true"
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


def _machine_line(prefix: str, payload: dict[str, object]) -> str:
    return prefix + json.dumps(payload, separators=(",", ":"))


def _valid_state_payload() -> dict[str, object]:
    return {
        "protocol_version": 1,
        "status": "ok",
        "ruleset_id": "nmm",
        "rules_identity": {
            "format_version": 1,
            "sha256": (
                "3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f"
            ),
        },
        "rules_options": {"pieceCount": 9},
        "history_origin": "game_start",
        "fen": (
            "********/********/******** w p p 0 9 0 9 0 0 "
            "-1 -1 -1 -1 0 0 1 ids:nodes"
        ),
        "side_to_move": "white",
        "phase": "placing",
        "action": "place",
        "pending_removal": False,
        "pending_removal_count": 0,
        "pending_removals": [0, 0],
        "legal_actions": list(POSITIONS),
        "action_token_count": 0,
        "logical_ply_count": 0,
        "logical_plies_by_side": [0, 0],
        "no_capture_count": 0,
        "repetition_current_count": 0,
        "repetition_history_length": 0,
        "snapshot_history_length": 0,
        "history_sha256": "a" * 64,
        "terminal": False,
        "winner": None,
        "winner_code": None,
        "outcome_reason": "ongoing",
        "outcome_reason_code": "ongoing",
    }


def _valid_logical_payload() -> dict[str, object]:
    return {
        "protocol_version": 1,
        "status": "ok",
        "full_turn_actions": ["d6-d5", "xc3"],
        "logical_move_id": "d6-d5xc3",
        "model_action": {"from": "d6", "to": "d5", "capture": "c3"},
        "logical_ply_delta": 1,
        "resulting_fen": (
            "***O****/OO@*@*@@/@OOO@*O* b m s 7 0 5 0 0 0 "
            "-1 -1 -1 -1 0 0 16 ids:nodes"
        ),
        "resulting_side_to_move": "black",
        "terminal": False,
        "winner": None,
        "winner_code": None,
        "outcome_reason": "ongoing",
        "effective_depth": 8,
        "completed_depth": 8,
        "score_kind": "cp",
        "score": 11,
        "score_perspective": "white",
        "node_budget": 500_000,
        "primary_nodes": 11_776,
        "removal_nodes": 0,
        "total_nodes": 11_776,
        "search_calls": 8,
    }


def test_machine_error_parser_preserves_strict_position_context() -> None:
    line = _machine_line(
        "info string sanmill_error ",
        {
            "protocol_version": 1,
            "status": "error",
            "code": "position_history_illegal_action",
            "command": "position",
            "message": "history action is not legal",
            "action_index": 1,
            "token": "a7",
        },
    )

    error = parse_protocol_error(line)

    assert error.code == "position_history_illegal_action"
    assert error.command == "position"
    assert error.action_index == 1
    assert error.token == "a7"


@pytest.mark.parametrize(
    ("parser", "prefix"),
    [
        (parse_protocol_error, "info string sanmill_error "),
        (parse_state_json_line, "info string sanmill_state "),
        (parse_logical_turn_line, "info string sanmill_logical_turn "),
    ],
)
def test_machine_json_parsers_reject_malformed_json(
    parser: Callable[[str], object],
    prefix: str,
) -> None:
    with pytest.raises(SanmillBridgeError):
        parser(prefix + "{")


def test_state_parser_rejects_unavailable_position_snapshot() -> None:
    line = _machine_line(
        "info string sanmill_state ",
        {
            "protocol_version": 1,
            "status": "position_unavailable",
            "code": "position_unavailable",
            "message": "the most recent position command was rejected",
            "position_error_code": "position_history_truncated",
        },
    )

    with pytest.raises(SanmillProtocolError) as raised:
        parse_state_json_line(line)

    assert raised.value.code == "position_unavailable"
    assert "position_history_truncated" in raised.value.message


def test_state_parser_accepts_authoritative_protocol_v1_snapshot() -> None:
    state = parse_state_json_line(
        _machine_line("info string sanmill_state ", _valid_state_payload())
    )

    assert state.status == "ok"
    assert state.ruleset_id == "nmm"
    assert state.logical_ply_count == 0
    assert state.logical_plies_by_side == (0, 0)
    assert len(state.legal_actions) == 24
    assert "strict_referee_identity" not in state.portable_record()


def test_state_parser_preserves_strict_referee_identity() -> None:
    payload = _valid_state_payload()
    payload["strict_referee_identity"] = {
        "format": "SANMILL-STRICT-REFEREE-RULES/1",
        "profile": "mif-stable-moving-v1",
        "repetitionObservation": "stable-moving-v1",
        "originCounted": True,
        "semanticDigest": (
            "sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94"
        ),
    }

    state = parse_state_json_line(
        _machine_line("info string sanmill_state ", payload)
    )

    assert state.strict_referee_identity is not None
    assert state.strict_referee_identity.portable_record() == payload[
        "strict_referee_identity"
    ]


def test_state_parser_rejects_extended_strict_referee_identity() -> None:
    payload = _valid_state_payload()
    payload["strict_referee_identity"] = {
        "format": "SANMILL-STRICT-REFEREE-RULES/1",
        "profile": "mif-stable-moving-v1",
        "repetitionObservation": "stable-moving-v1",
        "originCounted": True,
        "semanticDigest": "sha256:" + "1" * 64,
        "unreviewedMeaning": True,
    }

    with pytest.raises(SanmillBridgeError, match="wrong shape"):
        parse_state_json_line(
            _machine_line("info string sanmill_state ", payload)
        )


def test_state_parser_allows_terminal_snapshot_to_retain_remove_action() -> None:
    payload = _valid_state_payload()
    payload.update(
        {
            "status": "terminal",
            "fen": (
                "**O**O**/@**OOOOO/**O*@*** w o r 8 0 2 0 0 0 "
                "-1 -1 -1 -1 0 0 29 ids:nodes"
            ),
            "side_to_move": None,
            "phase": "game_over",
            "action": "remove",
            "legal_actions": [],
            "action_token_count": 2,
            "logical_ply_count": 1,
            "logical_plies_by_side": [1, 0],
            "terminal": True,
            "winner": "white",
            "winner_code": 0,
            "outcome_reason": "loseFewerThanThree",
            "outcome_reason_code": "lose_fewer_than_three",
        }
    )

    state = parse_state_json_line(
        _machine_line("info string sanmill_state ", payload)
    )

    assert state.terminal
    assert state.action == "remove"
    assert not state.removal_pending


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", 2),
        ("history_sha256", "not-a-digest"),
        ("logical_plies_by_side", [1, 0]),
        ("pending_removal", True),
    ],
)
def test_state_parser_rejects_identity_and_count_drift(
    field: str,
    value: object,
) -> None:
    payload = _valid_state_payload()
    payload[field] = value

    with pytest.raises(SanmillBridgeError):
        parse_state_json_line(
            _machine_line("info string sanmill_state ", payload)
        )


def test_logical_turn_parser_validates_complete_turn_and_budget() -> None:
    result = parse_logical_turn_line(
        _machine_line(
            "info string sanmill_logical_turn ",
            _valid_logical_payload(),
        ),
        elapsed_seconds=0.25,
    )

    assert result.full_turn_actions == ("d6-d5", "xc3")
    assert result.model_action == {
        "from": "d6",
        "to": "d5",
        "capture": "c3",
    }
    assert result.total_nodes == result.primary_nodes
    assert result.elapsed_seconds == 0.25


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_nodes", 500_001),
        ("logical_move_id", "d6-d5"),
        ("model_action", {"from": "d6", "to": "d5", "capture": None}),
        ("completed_depth", 9),
    ],
)
def test_logical_turn_parser_rejects_inconsistent_results(
    field: str,
    value: object,
) -> None:
    payload = _valid_logical_payload()
    payload[field] = value

    with pytest.raises(SanmillBridgeError):
        parse_logical_turn_line(
            _machine_line("info string sanmill_logical_turn ", payload)
        )


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
        before = session.state_json()
        result = session.search_logical_turn(100_000)
        session.position_startpos(result.full_turn_actions)
        after = session.state_json()

    # SkillLevel=30 would request depth 30 if the ordinary opening policy were
    # bypassed.  Depth 1 is the pinned non-developer opening-table result.
    assert result.effective_depth == 1
    assert result.completed_depth == 1
    assert 0 < result.total_nodes < 100_000
    assert result.resulting_fen == after.fen
    assert before.logical_ply_count == 0
    assert after.logical_ply_count == 1
    assert before.history_sha256 != after.history_sha256


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_strict_position_error_is_immediate_and_stream_stays_aligned() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)

    with SanmillUciSession(installation) as session:
        session.position_startpos()
        with pytest.raises(SanmillProtocolError) as raised:
            session.position_startpos(("a7", "a7"))
        assert raised.value.code == "position_history_illegal_action"

        # This second command proves that the failed position call consumed its
        # own readyok instead of leaving a stale synchronization response.
        session.position_startpos(("a7",))
        recovered = session.state_json()

    assert recovered.action_token_count == 1
    assert recovered.logical_ply_count == 1


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_logical_turn_is_reproducible_and_includes_mill_removal() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    prefix = ("d2", "d6", "f4", "b4", "f2", "g4")

    semantic_runs = []
    for _ in range(2):
        with SanmillUciSession(installation, seed=7) as session:
            session.position_startpos(prefix)
            before = session.state_json()
            result = session.search_logical_turn(100_000, depth=5)
            session.position_startpos((*prefix, *result.full_turn_actions))
            after = session.state_json()
        assert len(result.full_turn_actions) == 2
        assert result.full_turn_actions[1].startswith("x")
        assert result.resulting_fen == after.fen
        assert after.logical_ply_count == before.logical_ply_count + 1
        assert (
            after.action_token_count
            == before.action_token_count + len(result.full_turn_actions)
        )
        semantic_runs.append(result.semantic_record())

    assert semantic_runs[0] == semantic_runs[1]


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored sanmill_checkout path registry entry",
)
def test_local_terminal_state_returns_zero_node_logical_result() -> None:
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    terminal_fen = (
        "**O**O**/**@**@**/******** w m s 2 0 2 0 0 0 "
        "-1 -1 -1 -1 0 0 1 ids:nodes"
    )

    with SanmillUciSession(installation) as session:
        session.position_fen(terminal_fen)
        state = session.state_json()
        result = session.search_logical_turn(100)

    assert state.terminal
    assert state.outcome_reason == "loseFewerThanThree"
    assert result.status == "terminal"
    assert result.full_turn_actions == ()
    assert result.total_nodes == 0
