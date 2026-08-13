"""Tests for zero-game phase-process mechanism reanalysis."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import learned_ai.evaluation.retained_phase_process_mechanism_audit as audit
from game.board import BoardState
from game.rules import get_all_legal_moves


class _DrawValue:
    def __init__(self, perspective: str) -> None:
        self.sector = (9, 9, 0, 0)
        self.sector_value = 0
        self.perspective = perspective
        self.outcome = "D"

    def ordering_key(self) -> tuple[int, int]:
        return 0, 0


class _AllDrawMalom:
    @staticmethod
    def query_state(_board):
        return "D"

    @staticmethod
    def query_all_moves(board, _turn):
        return [
            {
                "move": move,
                "wdl": "draw",
                "oracle_value": _DrawValue(board.turn),
            }
            for move in get_all_legal_moves(board)
        ]


def _one_turn_record() -> dict:
    board = BoardState.new_game()
    move = get_all_legal_moves(board)[0]
    after = board.apply_move(move)
    return {
        "start": {"final_nmm_fen": board.to_fen_string()},
        "turns": [
            {
                "move": move,
                "mover_color": "W",
                "actor": "candidate",
                "candidate_malom_delta": 0.0,
                "post_start_logical_ply": 1,
                "local_fen_after": after.to_fen_string(),
            }
        ],
        "match_key": "start-1:W",
        "start_id": "start-1",
        "candidate_id": "retained-v3-refresh50",
        "candidate_color": "W",
        "phase": "placement",
        "ordinal": 0,
        "game_id": "game-1",
        "outcome_reason": "test",
    }


def _safe_summary(*, candidate_turns: int, missed: int) -> dict:
    counts = audit._empty_safe_counts()
    counts["candidate_turns"] = candidate_turns
    counts["queryable_parent_turns"] = candidate_turns
    counts["all_legal_actions_queryable_turns"] = candidate_turns
    counts["safe_capture_opportunity_turns"] = missed
    counts["missed_safe_capture_turns"] = missed
    return audit._summarise_safe_counts(counts)


def _order_summary(*, candidate_turns: int, regret: float) -> dict:
    counts = audit._empty_order_counts()
    counts["candidate_turns"] = candidate_turns
    counts["parent_queryable_turns"] = candidate_turns
    counts["all_legal_actions_queryable_turns"] = candidate_turns
    counts["chosen_coarse_preserving_turns"] = candidate_turns
    counts["within_wdl_orderable_turns"] = candidate_turns
    counts["normalised_ordinal_regret_sum"] = regret
    return audit._summarise_order_counts(counts)


def _audited_game(record: dict) -> dict:
    candidate = record["candidate_id"]
    is_v4 = candidate == "retained-v4-no-refresh"
    safe = _safe_summary(candidate_turns=10, missed=1 if is_v4 else 0)
    order = _order_summary(candidate_turns=10, regret=1.0 if is_v4 else 0.0)
    empty_safe = _safe_summary(candidate_turns=0, missed=0)
    empty_order = _order_summary(candidate_turns=0, regret=0.0)
    return {
        "match_key": record["match_key"],
        "start_id": record["start_id"],
        "candidate_id": candidate,
        "candidate_color": record["candidate_color"],
        "phase": record["phase"],
        "source_ordinal": record["ordinal"],
        "source_game_id": record["game_id"],
        "outcome_reason": "drawFiftyMove",
        "safe_progress": {
            "all_candidate_turns": safe,
            "after_relative_horizon_candidate_turns": empty_safe,
        },
        "complete_order": {
            "all_candidate_turns": order,
            "after_relative_horizon_candidate_turns": empty_order,
        },
    }


def test_game_audit_replays_from_variable_start_and_labels_denominators() -> None:
    result = audit.audit_game_record(_one_turn_record(), _AllDrawMalom())
    safe = result["safe_progress"]["all_candidate_turns"]
    order = result["complete_order"]["all_candidate_turns"]
    assert safe["candidate_turns"] == 1
    assert safe["parent_query_coverage"] == 1.0
    assert safe["safe_capture_opportunity_turns"] == 0
    assert order["within_wdl_orderable_turns"] == 1
    assert order["full_order_choice_opportunity_turns"] == 0
    assert order["mean_normalised_ordinal_regret_given_orderable"] == 0.0


def test_game_audit_fails_if_the_recorded_local_fen_does_not_replay() -> None:
    record = _one_turn_record()
    record["turns"][0]["local_fen_after"] = "tampered"
    with pytest.raises(audit.RetainedPhaseProcessMechanismError, match="does not replay"):
        audit.audit_game_record(record, _AllDrawMalom())


def test_complete_audit_clusters_both_colours_by_start(monkeypatch) -> None:
    records = []
    candidates = ("retained-v3-refresh50", "retained-v4-no-refresh")
    phases = ("placement", "movement", "flying")
    ordinal = 0
    for start_index in range(39):
        for color in ("W", "B"):
            for candidate in candidates:
                records.append(
                    {
                        "match_key": f"start-{start_index}:{color}",
                        "start_id": f"start-{start_index}",
                        "candidate_id": candidate,
                        "candidate_color": color,
                        "phase": phases[start_index % len(phases)],
                        "ordinal": ordinal,
                        "game_id": f"game-{ordinal}",
                    }
                )
                ordinal += 1
    monkeypatch.setattr(audit, "audit_game_record", lambda record, malom, progress: _audited_game(record))
    result = audit.recompute_mechanism_audit(
        source_spec={"diagnostic_id": "d", "spec_identity": "s" * 64},
        source_records=records,
        source_ledger_sha256="l" * 64,
        source_result_identity="r" * 64,
        implementation_commit="c" * 40,
        malom=SimpleNamespace(),
    )
    safe = result["paired"][
        "start_clustered_missed_safe_capture_share_v4_minus_v3"
    ]
    order = result["paired"][
        "start_clustered_mean_order_regret_v4_minus_v3"
    ]
    assert safe["support"] == 39
    assert safe["mean"] == pytest.approx(0.1)
    assert safe["decision"] == "exploratory_no_directional_gate"
    assert order["support"] == 39
    assert order["mean"] == pytest.approx(0.1)
    assert result["source"]["new_games"] == 0
    assert result["claim_boundary"]["playing_strength_claim"] is False
