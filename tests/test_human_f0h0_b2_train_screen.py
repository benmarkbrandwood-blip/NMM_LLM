from __future__ import annotations

from pathlib import Path

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.human_f0h0_b2_freeze import load_membership
from learned_ai.evaluation.human_f0h0_b2_train_screen import (
    EstimabilityCell,
    ProtectedPartitionAccessError,
    SupportCell,
    TrainOnlyAccess,
    clustered_proportion,
    derive_train_sample,
    estimate_action_effects,
    estimability_summary,
    ring16_canonical_transition,
    support_summary,
)
from learned_ai.evaluation.human_f0h0_feasibility import (
    CorpusRecord,
    F0D0Boundary,
    canonical_sha256,
)
from learned_ai.evaluation.oracle_corpus import _D4, _transform_board


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "docs/experiments/f0-h0-design-b2-frozen-membership-v1.json"


def _membership() -> dict:
    return {
        "partitions": {
            "train": {"session_ids": ["train-1"]},
            "selection": {"session_ids": ["selection-1"]},
            "confirmation": {"session_ids": ["confirmation-1"]},
            "final-test": {"session_ids": ["final-1"]},
        }
    }


def _record(session: str) -> CorpusRecord:
    return CorpusRecord(
        session_id=session,
        canonical_file=f"data/human_games/{session}.jsonl",
        move_count=1,
        recorded_outcome=None,
        player_keys=("white", "black"),
        behavior_eligible=True,
        outcome_eligible=False,
    )


def _boundary(record: CorpusRecord) -> F0D0Boundary:
    return F0D0Boundary(
        manifest={},
        file_sha256="0" * 64,
        records=(record,),
        raw_sha256_by_path={record.canonical_file: "0" * 64},
        raw_size_by_path={record.canonical_file: 0},
    )


@pytest.mark.parametrize("partition", ["selection", "confirmation", "final-test"])
def test_train_only_guard_raises_before_protected_raw_reader(
    partition: str,
) -> None:
    session = f"{partition}-1"
    record = _record(session)
    access = TrainOnlyAccess.from_membership(_membership(), ["train-1"])
    called = False

    def reader(*_args):
        nonlocal called
        called = True
        raise AssertionError("protected reader must not run")

    with pytest.raises(ProtectedPartitionAccessError, match="train-only"):
        access.read_raw_game(ROOT, record, _boundary(record), reader=reader)
    assert called is False


@pytest.mark.parametrize("partition", ["selection", "confirmation", "final-test"])
def test_train_only_guard_raises_before_feature_producer(partition: str) -> None:
    access = TrainOnlyAccess.from_membership(_membership(), ["train-1"])
    called = False

    def producer() -> int:
        nonlocal called
        called = True
        return 1

    with pytest.raises(ProtectedPartitionAccessError, match="derived_features"):
        access.derive_features(f"{partition}-1", producer)
    assert called is False


def test_train_feature_producer_is_allowed() -> None:
    access = TrainOnlyAccess.from_membership(_membership(), ["train-1"])
    assert access.derive_features("train-1", lambda: 7) == 7
    assert access.successful_accesses[("train", "derived_features")] == 1


def test_official_fallback_intersection_is_train_only_without_resampling() -> None:
    membership, _file_sha = load_membership(MEMBERSHIP)
    train, composition = derive_train_sample(membership)
    assert composition == {
        "train": 9_113,
        "selection": 887,
        "confirmation": 0,
        "final-test": 0,
    }
    assert len(train) == 9_113
    assert canonical_sha256(train) == canonical_sha256(sorted(train))


def _transform_fen(fen: str, matrix: tuple[int, int, int, int]) -> str:
    board, turn, white, black = fen.split("|")
    return f"{_transform_board(board, matrix)}|{turn}|{white}|{black}"


def test_ring16_transition_uses_one_shared_transform() -> None:
    board = BoardState.new_game()
    before = board.to_fen_string()
    move = get_all_legal_moves(board)[3]
    after = board.apply_move(move).to_fen_string()
    matrix = _D4[3]
    assert ring16_canonical_transition(before, after) == (
        ring16_canonical_transition(
            _transform_fen(before, matrix),
            _transform_fen(after, matrix),
        )
    )


def test_ring16_transition_does_not_collapse_distinct_empty_board_actions() -> None:
    board = BoardState.new_game()
    before = board.to_fen_string()
    legal = get_all_legal_moves(board)
    by_target = {move["to"]: move for move in legal}
    corner = board.apply_move(by_target["a1"]).to_fen_string()
    midpoint = board.apply_move(by_target["d1"]).to_fen_string()
    assert ring16_canonical_transition(before, corner) != (
        ring16_canonical_transition(before, midpoint)
    )


def test_support_summary_applies_player_and_game_floors() -> None:
    supported = SupportCell()
    for index in range(10):
        supported.observe(f"p{index % 5}", f"g{index}")
    unsupported = SupportCell()
    unsupported.observe("p0", "g0")
    summary, keys = support_summary(
        {"supported": supported, "unsupported": unsupported},
        minimum_players=5,
        minimum_games=10,
        tail_thresholds=[1, 10],
    )
    assert keys == {"supported"}
    assert summary["supported_states_or_classes"] == 1
    assert summary["supported_observations"] == 10
    assert summary["observation_count_tails"] == {"1": 2, "10": 1}


def _estimability_cell() -> EstimabilityCell:
    cell = EstimabilityCell()
    for index in range(20):
        action = "a" if index < 10 else "b"
        cell.observe(
            game=f"g{index}",
            players=(f"p{index}", f"q{index}"),
            action=action,
            fold=index % 2,
            event="W->D" if action == "a" and index % 2 == 0 else None,
        )
    return cell


def test_estimability_requires_two_safe_actions_with_m_support() -> None:
    cell = _estimability_cell()
    summary, keys = estimability_summary(
        {"state": cell},
        supported_keys={"state"},
        total_analysis_decisions=100,
        minimum_observations_k=20,
        minimum_per_action_m=5,
    )
    assert keys == {"state"}
    assert summary["classes_with_two_observed_safe_actions_each_at_least_m"] == 1
    assert summary["covered_decision_fraction"] == pytest.approx(0.2)
    assert summary["covered_games"] == 20


def test_crossfit_effect_reports_uncorrected_and_shrunk_estimates() -> None:
    cells: dict[str, EstimabilityCell] = {}
    for class_index in range(20):
        cell = EstimabilityCell()
        for fold in (0, 1):
            for action in ("a", "b"):
                for observation in range(6):
                    event = "W->D" if action == "a" and observation < 4 else None
                    cell.observe(
                        game=f"g-{class_index}-{fold}-{action}-{observation}",
                        players=(f"p-{observation}", f"q-{observation}"),
                        action=action,
                        fold=fold,
                        event=event,
                    )
        cells[f"class-{class_index}"] = cell
    result = estimate_action_effects(
        cells,
        eligible_keys=set(cells),
        minimum_per_action_m=5,
        selection_minimum=2,
        evaluation_minimum=2,
        minimum_crossfit_classes=20,
        bootstrap_replicates=100,
        bootstrap_seed="fixture",
    )
    wd = result["W->D"]
    assert wd["status"] == "estimated"
    assert wd["uncorrected_weighted_within_class_max_minus_min"] > 0.6
    assert wd["corrected_point"] > 0.5
    assert wd["conservative_lower_95"] > 0.5
    assert result["W->L"]["corrected_point"] == pytest.approx(0.0)


def test_clustered_proportion_uses_whole_games_as_clusters() -> None:
    result = clustered_proportion(
        {"g1": 1, "g2": 0, "g3": 2},
        {"g1": 2, "g2": 2, "g3": 2},
    )
    assert result["point"] == pytest.approx(0.5)
    assert result["independent_game_clusters"] == 3
    assert result["lower_95"] <= result["point"] <= result["upper_95"]
