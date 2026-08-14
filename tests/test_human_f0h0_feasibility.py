from __future__ import annotations

import pytest

from ai.malom_db import OracleMoveValue, OracleValue
from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.evaluation.human_f0h0_feasibility import (
    F0H0Error,
    CorpusRecord,
    F0D0Boundary,
    ReplayedDecision,
    build_split,
    canonical_sha256,
    component_robust_interval,
    concentration,
    gini,
    label_positional_decision,
    replay_game,
    verify_split,
    wilson_interval,
)


def _plan(*, games: int, plies: int, players: int) -> dict:
    plan = {
        "screen_id": "fixture-f0-h0",
        "input_boundary": {
            "behavior": {
                "games": games,
                "logical_plies": plies,
                "player_keys": players,
            }
        },
        "split": {
            "algorithm": (
                "connected player-game components, descending game count, "
                "minimum normalized target load"
            ),
            "ratios": {
                "train": 0.70,
                "selection": 0.15,
                "one-time-confirmation": 0.075,
                "final-test": 0.075,
            },
            "seed": "fixture-seed",
        },
    }
    return {**plan, "plan_identity": canonical_sha256(plan)}


def _record(
    session: str,
    white: str,
    black: str,
    *,
    plies: int = 1,
) -> CorpusRecord:
    return CorpusRecord(
        session_id=session,
        canonical_file=f"data/human_games/human_{session}.jsonl",
        move_count=plies,
        recorded_outcome="D",
        player_keys=(white, black),
        behavior_eligible=True,
        outcome_eligible=True,
    )


def _boundary(records: list[CorpusRecord]) -> F0D0Boundary:
    return F0D0Boundary(
        manifest={},
        file_sha256="f" * 64,
        records=tuple(records),
        raw_sha256_by_path={},
        raw_size_by_path={},
    )


def test_split_keeps_connected_players_and_games_together() -> None:
    records = [
        _record("g1", "a", "b"),
        _record("g2", "b", "c"),
        _record("g3", "d", "e"),
        _record("g4", "f", "g"),
    ]
    plan = _plan(games=4, plies=4, players=7)
    payload = build_split(boundary=_boundary(records), plan=plan)
    split = {
        **payload,
        "split_identity": canonical_sha256(payload),
    }
    verify_split(split, boundary=_boundary(records), plan=plan)

    game_partition = {row[0]: row[1] for row in split["game_membership"]}
    player_partition = {row[0]: row[1] for row in split["player_membership"]}
    assert game_partition["g1"] == game_partition["g2"]
    assert player_partition["a"] == player_partition["b"]
    assert player_partition["b"] == player_partition["c"]
    assert split["component_count"] == 3
    assert split["access_state"]["final-test_raw_record_reads"] == 0


def test_split_verifier_rejects_player_leakage() -> None:
    records = [_record("g1", "a", "b")]
    plan = _plan(games=1, plies=1, players=2)
    payload = build_split(boundary=_boundary(records), plan=plan)
    rows = [list(row) for row in payload["player_membership"]]
    rows[0][1] = "final-test"
    tampered = {
        **payload,
        "player_membership": rows,
        "player_membership_identity": canonical_sha256(rows),
    }
    tampered["split_identity"] = canonical_sha256(tampered)
    with pytest.raises(F0H0Error, match="crosses split partitions"):
        verify_split(tampered, boundary=_boundary(records), plan=plan)


def test_concentration_and_wilson_are_deterministic() -> None:
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    metrics = concentration([4, 3, 2, 1])
    assert metrics["observations"] == 10
    assert metrics["maximum_share"] == pytest.approx(0.4)
    assert metrics["top_10_percent_share"] == pytest.approx(0.4)
    interval = wilson_interval(5, 10)
    assert interval["point"] == pytest.approx(0.5)
    assert interval["lower_95"] == pytest.approx(0.236593, abs=1e-6)
    assert interval["upper_95"] == pytest.approx(0.763407, abs=1e-6)


def test_component_interval_envelopes_wilson_and_fails_closed() -> None:
    interval = component_robust_interval(
        [("component-a", True), ("component-a", False), ("component-b", False)]
    )
    assert interval["successes"] == 1
    assert interval["observations"] == 3
    assert interval["independent_components"] == 2
    assert interval["lower_95"] <= interval["fixed_membership_wilson"]["lower_95"]
    assert interval["upper_95"] >= interval["fixed_membership_wilson"]["upper_95"]

    unavailable = component_robust_interval(
        [("only-component", True), ("only-component", False)]
    )
    assert unavailable["lower_95"] == 0.0
    assert unavailable["upper_95"] == 1.0
    assert unavailable["method"].startswith("fail-closed")


def test_replay_game_reconstructs_actor_identity() -> None:
    board = BoardState.new_game()
    first = get_all_legal_moves(board)[0]
    raw_game = {
        "session_id": "g1",
        "moves": [
            {
                "board_fen_before": board.to_fen_string(),
                "color": board.turn,
                "turn": 1,
                "type": get_game_phase(board, board.turn),
                "from": first.get("from"),
                "to": first.get("to"),
                "capture": first.get("capture"),
                "notation": first["to"],
            }
        ],
    }
    record = _record("g1", "white-key", "black-key")
    decisions = replay_game(raw_game, record)
    assert len(decisions) == 1
    assert decisions[0].actor_player_key == "white-key"
    assert decisions[0].move == first


class _FakeMalom:
    def __init__(self, chosen_fen: str) -> None:
        self.chosen_fen = chosen_fen

    def query_value(self, board: BoardState) -> OracleValue:
        key2 = 9 if board.to_fen_string() == self.chosen_fen else 1
        return OracleValue(
            raw_key1=0,
            sector_value=0,
            absolute_key1=0,
            key2=key2,
            entry_kind="value",
            perspective=board.turn,
            sector=(0, 0, 9, 9),
            outcome="D",
        )

    def move_value(
        self,
        parent: OracleValue,
        child: OracleValue,
    ) -> OracleMoveValue:
        return OracleMoveValue(
            key1=0,
            key2=child.key2,
            sector_value=0,
            absolute_key1=0,
            perspective=parent.perspective,
            sector=parent.sector,
            outcome="D",
            source="fixture",
            child_value=child,
        )

    def terminal_move_value(
        self,
        _parent: OracleValue,
        _child_outcome: str,
    ) -> OracleMoveValue:
        raise AssertionError("initial placement cannot be terminal")


def test_positional_label_keeps_a_pos_separate_from_full_order_regret() -> None:
    board = BoardState.new_game()
    chosen = get_all_legal_moves(board)[0]
    chosen_fen = board.apply_move(chosen).to_fen_string()
    decision = ReplayedDecision(
        logical_ply=0,
        board=board,
        move=chosen,
        actor_player_key="player",
        game_id="game",
    )
    label = label_positional_decision(decision, _FakeMalom(chosen_fen))
    assert label.parent_tier == "D"
    assert label.chosen_tier == "D"
    assert label.a_pos_cardinality == 24
    assert label.chosen_preserves_tier is True
    assert label.within_tier_full_regret is False


def test_split_rejects_wrong_preregistered_base() -> None:
    records = [_record("g1", "a", "b")]
    plan = _plan(games=2, plies=1, players=2)
    with pytest.raises(F0H0Error, match="eligibility base differs"):
        build_split(boundary=_boundary(records), plan=plan)
