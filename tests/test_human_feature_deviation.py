from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.human_f0h0_feasibility import (
    CorpusRecord,
    F0D0Boundary,
    canonical_sha256,
)
from learned_ai.evaluation import human_feature_deviation as feature_design


def _sealed(path: Path, identity_field: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value[identity_field]
    body = dict(value)
    body.pop(identity_field)
    assert canonical_sha256(body) == identity
    return value


def _players_for_arms(seed: str) -> tuple[list[str], list[str]]:
    exploration: list[str] = []
    confirmation: list[str] = []
    index = 0
    while len(exploration) < 3 or len(confirmation) < 2:
        player = f"p{index}"
        arm = feature_design._player_arm(player, seed=seed)
        (confirmation if arm == "research-confirmation" else exploration).append(player)
        index += 1
    return exploration, confirmation


def test_train_internal_split_is_player_isolated_and_discards_cross_games(
    monkeypatch,
) -> None:
    seed = "unit-player-split"
    exploration, confirmation = _players_for_arms(seed)
    records = (
        CorpusRecord(
            "g-ee", "ee.jsonl", 10, None, (exploration[0], exploration[1]), True, False
        ),
        CorpusRecord(
            "g-cc",
            "cc.jsonl",
            11,
            None,
            (confirmation[0], confirmation[1]),
            True,
            False,
        ),
        CorpusRecord(
            "g-ec", "ec.jsonl", 12, None, (exploration[2], confirmation[0]), True, False
        ),
    )
    boundary = F0D0Boundary(
        manifest={},
        file_sha256=feature_design.EXPECTED_F0D0_FILE_SHA256,
        records=records,
        raw_sha256_by_path={},
        raw_size_by_path={},
    )
    monkeypatch.setitem(feature_design.EXPECTED_B2_COUNTS, "train", 3)
    official = {
        "membership_identity": feature_design.EXPECTED_B2_MEMBERSHIP_IDENTITY,
        "partitions": {"train": {"session_ids": ["g-cc", "g-ec", "g-ee"]}},
    }
    plan = {
        "plan_identity": "a" * 64,
        "train_internal_split": {
            "player_hash_seed": seed,
            "exploration_pilot_hash_seed": "unit-pilot",
            "exploration_pilot_games": 1,
        },
    }

    split = feature_design.build_train_internal_split(boundary, official, plan)

    assert split["partitions"]["research-exploration"]["session_ids"] == ["g-ee"]
    assert split["partitions"]["research-confirmation"]["session_ids"] == ["g-cc"]
    assert split["partitions"]["cross-player-discard"]["session_ids"] == ["g-ec"]
    assert split["player_membership"]["pairwise_player_overlap"] == 0
    assert split["access_state"]["raw_game_files_opened"] == 0


def test_exploration_guard_rejects_internal_and_official_confirmation() -> None:
    official = {
        "partitions": {
            "train": {"session_ids": ["pilot", "internal-confirm"]},
            "selection": {"session_ids": ["selection"]},
            "confirmation": {"session_ids": ["official-confirm"]},
            "final-test": {"session_ids": ["final"]},
        }
    }
    split = {
        "partitions": {
            "research-exploration": {"session_ids": ["pilot"]},
            "research-confirmation": {"session_ids": ["internal-confirm"]},
            "cross-player-discard": {"session_ids": []},
        },
        "exploration_pilot": {"session_ids": ["pilot"]},
    }
    access = feature_design.ExplorationOnlyAccess.from_memberships(official, split)
    access.assert_pilot("pilot", access_kind="raw_game")

    for session in ("internal-confirm", "selection", "official-confirm", "final"):
        with pytest.raises(feature_design.ProtectedResearchAccessError):
            access.assert_pilot(session, access_kind="derived_features")


def test_action_feature_dictionary_is_exact_finite_and_visible() -> None:
    board = BoardState.new_game()
    rows = [
        feature_design.action_feature_scores(board, move)
        for move in get_all_legal_moves(board)
    ]

    assert rows
    assert all(tuple(row) == feature_design.FEATURE_NAMES for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row.values())
    assert all(row["closes_mill"] == 0.0 for row in rows)
    assert len({row["destination_degree"] for row in rows}) > 1


def test_tied_heuristic_maxima_keep_mixed_and_strict_conflicts_separate() -> None:
    mixed = feature_design.FeatureOpportunity()
    mixed.observe(
        scores=[1.0, 1.0, 0.0],
        safe_indices={0},
        chosen_index=1,
        player="p",
        game="g",
        phase="placement",
        tier="D",
        color="W",
    )
    assert mixed.possible_conflict == 1
    assert mixed.mixed_maximum == 1
    assert mixed.strict_conflict == 0
    assert mixed.unsafe_follow == 1

    strict = feature_design.FeatureOpportunity()
    strict.observe(
        scores=[1.0, 1.0, 0.0],
        safe_indices={2},
        chosen_index=1,
        player="p",
        game="g",
        phase="placement",
        tier="D",
        color="W",
    )
    assert strict.strict_conflict == 1
    assert strict.strict_conflict_follow == 1


def test_zero_event_rate_is_not_created_by_a_prior() -> None:
    row = feature_design._rate(0, 100)

    assert row["point"] == 0.0
    assert row["zero_events_not_smoothed"] is True
    assert row["upper_95"] > 0.0


def test_tracked_plan_split_and_exploration_are_identity_consistent() -> None:
    root = Path(__file__).resolve().parent.parent
    plan_path = root / "docs/experiments/human-feature-deviation-screen-v1.json"
    split_path = root / "docs/experiments/human-feature-deviation-train-split-v1.json"
    result_path = (
        root
        / "docs/evidence/human-feature-deviation-exploration-manifest-2026-08-15.json"
    )

    plan, _ = feature_design.load_plan(plan_path)
    split = _sealed(split_path, "split_identity")
    result = _sealed(result_path, "result_identity")

    assert split["plan_identity"] == plan["plan_identity"]
    assert result["plan_identity"] == plan["plan_identity"]
    assert result["split_identity"] == split["split_identity"]
    assert result["status"] == "exploration_only_no_confirmatory_decision"
    assert result["sample"]["games"] == 128
    assert result["access_audit"]["research_confirmation_content_reads"] == 0
    assert result["access_audit"]["selection_content_reads"] == 0
    assert result["access_audit"]["confirmation_content_reads"] == 0
    assert result["access_audit"]["final_test_content_reads"] == 0
    assert result["access_audit"]["source_pool_2eb04f54_records_read_or_consumed"] == 0
