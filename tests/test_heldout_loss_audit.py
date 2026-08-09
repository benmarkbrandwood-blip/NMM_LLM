from __future__ import annotations

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.heldout_loss_audit import (
    audit_game_wdl_transitions,
    select_losses_and_matched_draws,
)


def _selection_record(
    ordinal: int,
    *,
    score: float,
    stratum: str,
    color: str,
    strict: bool,
) -> dict:
    return {
        "candidate_color": color,
        "candidate_score": score,
        "ordinal": ordinal,
        "stratum": stratum,
        "strict_independence_sensitivity": strict,
    }


def test_loss_controls_match_colour_stratum_and_strict_without_reuse() -> None:
    records = [
        _selection_record(0, score=0.0, stratum="book", color="B", strict=True),
        _selection_record(1, score=0.0, stratum="book", color="B", strict=True),
        _selection_record(2, score=0.5, stratum="book", color="B", strict=True),
        _selection_record(3, score=0.5, stratum="book", color="B", strict=True),
        _selection_record(4, score=0.5, stratum="book", color="W", strict=True),
    ]

    losses, controls, matches = select_losses_and_matched_draws(records)

    assert [row["ordinal"] for row in losses] == [0, 1]
    assert [row["ordinal"] for row in controls] == [2, 3]
    assert [row["control_ordinal"] for row in matches] == [2, 3]


def test_game_audit_finds_first_candidate_wdl_downgrade_without_policy() -> None:
    board = BoardState.new_game()
    moves = []
    states = [board]
    for _ in range(4):
        move = get_all_legal_moves(board)[0]
        board = board.apply_move(move)
        moves.append(move)
        states.append(board)
    state_wdl = {
        states[0].to_fen_string(): "W",
        states[1].to_fen_string(): "L",
        states[2].to_fen_string(): "D",
        states[3].to_fen_string(): "W",
        states[4].to_fen_string(): "L",
    }
    turns = []
    for index, (move, after) in enumerate(zip(moves, states[1:]), 1):
        turns.append(
            {
                "actor": "candidate" if index % 2 else "sanmill",
                "local_fen_after": after.to_fen_string(),
                "move": move,
                "mover_color": "W" if index % 2 else "B",
                "post_prefix_logical_ply": index,
            }
        )
    record = {
        "candidate_color": "W",
        "candidate_score": 0.0,
        "game_id": "synthetic-loss",
        "ordinal": 0,
        "outcome_reason": "loseFewerThanThree",
        "pair_index": 0,
        "prefix": {"final_nmm_fen": states[0].to_fen_string()},
        "source_core_id": "synthetic-source",
        "stratum": "book",
        "strict_independence_sensitivity": True,
        "turns": turns,
    }

    result = audit_game_wdl_transitions(
        record,
        lambda state: state_wdl[state.to_fen_string()],
        cohort="candidate_loss",
    )

    assert result["classification"] == "candidate_wdl_downgrade_found"
    assert result["candidate_turns_probed"] == 2
    assert result["known_candidate_turns"] == 2
    downgrade = result["first_candidate_wdl_downgrade"]
    assert downgrade["after_candidate_wdl"] == "L"
    assert downgrade["after_fen"] == states[3].to_fen_string()
    assert downgrade["before_candidate_wdl"] == "D"
    assert downgrade["before_fen"] == states[2].to_fen_string()
    assert downgrade["delta"] == -1
    assert downgrade["move"] == moves[2]
    assert downgrade["phase"] == "place"
    assert downgrade["post_prefix_logical_ply"] == 3
    assert downgrade["before_position_features"]["candidate"]["pieces_on_board"] == 1
    assert downgrade["after_position_features"]["candidate"]["pieces_on_board"] == 2
    assert result["position_summary_before_stop"]["candidate_phase_counts"] == {
        "place": 2
    }


def test_game_audit_fails_closed_on_unknown_transition() -> None:
    board = BoardState.new_game()
    move = get_all_legal_moves(board)[0]
    after = board.apply_move(move)
    record = {
        "candidate_color": "W",
        "candidate_score": 0.5,
        "game_id": "synthetic-control",
        "ordinal": 1,
        "outcome_reason": "drawThreefoldRepetition",
        "pair_index": 0,
        "prefix": {"final_nmm_fen": board.to_fen_string()},
        "source_core_id": "synthetic-source",
        "stratum": "book",
        "strict_independence_sensitivity": True,
        "turns": [
            {
                "actor": "candidate",
                "local_fen_after": after.to_fen_string(),
                "move": move,
                "mover_color": "W",
                "post_prefix_logical_ply": 1,
            }
        ],
    }

    result = audit_game_wdl_transitions(
        record,
        lambda state: "D" if state.to_fen_string() == board.to_fen_string() else None,
        cohort="matched_draw_control",
    )

    assert result["classification"] == "insufficient_malom_coverage"
    assert result["first_candidate_wdl_downgrade"] is None
    assert result["unknown_transition"]["post_prefix_logical_ply"] == 1
