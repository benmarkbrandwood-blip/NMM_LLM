from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import learned_ai.evaluation.human_f0h0_b2_freeze as subject
from learned_ai.evaluation.human_f0h0_b2_freeze import (
    B2FreezeError,
    FinalTestAccessError,
    FrozenSplitAccess,
    characterize_metadata,
    partition_b2_games,
    project_cost,
)
from learned_ai.evaluation.human_f0h0_feasibility import (
    CorpusRecord,
    F0D0Boundary,
)
from learned_ai.evaluation.human_f0h0_split_retest import GameRecord


def _game(
    session: str,
    white: str,
    black: str,
    played_on: str,
    *,
    moves: int = 20,
) -> GameRecord:
    return GameRecord(
        session_id=session,
        canonical_file=f"human_{session}.jsonl",
        file_sha256="0" * 64,
        file_size=1,
        played_on=date.fromisoformat(played_on),
        move_count=moves,
        white_player=white,
        black_player=black,
    )


def test_b2_partition_reapplies_prior_player_rule_at_each_cut() -> None:
    games = [
        _game("train", "a", "b", "2026-02-20"),
        _game("selection", "c", "d", "2026-03-05"),
        _game("selection-old", "a", "e", "2026-03-06"),
        _game("confirmation", "f", "g", "2026-04-05"),
        _game("confirmation-old", "c", "h", "2026-04-06"),
        _game("final", "i", "j", "2026-05-05"),
        _game("final-old", "f", "k", "2026-05-06"),
    ]

    split = partition_b2_games(
        games,
        train_cut=date(2026, 3, 1),
        confirmation_cut=date(2026, 4, 1),
        final_cut=date(2026, 5, 1),
    )

    assert {game.session_id for game in split["train"]} == {"train"}
    assert {game.session_id for game in split["selection"]} == {"selection"}
    assert {game.session_id for game in split["confirmation"]} == {
        "confirmation"
    }
    assert {game.session_id for game in split["final-test"]} == {"final"}
    assert set.union(
        *(
            {player for game in split[name] for player in game.players}
            for name in ("selection", "confirmation", "final-test")
        )
    ) == {"c", "d", "f", "g", "i", "j"}


def test_b2_partition_rejects_nonmonotonic_cuts() -> None:
    with pytest.raises(B2FreezeError, match="cut order"):
        partition_b2_games(
            [],
            train_cut=date(2026, 3, 1),
            confirmation_cut=date(2026, 3, 1),
            final_cut=date(2026, 5, 1),
        )


def _record(session: str) -> CorpusRecord:
    return CorpusRecord(
        session_id=session,
        canonical_file=f"human_{session}.jsonl",
        move_count=1,
        recorded_outcome=None,
        player_keys=("white", "black"),
        behavior_eligible=True,
        outcome_eligible=False,
    )


def _raw_boundary(record: CorpusRecord) -> F0D0Boundary:
    return F0D0Boundary(
        manifest={},
        file_sha256="0" * 64,
        records=(record,),
        raw_sha256_by_path={record.canonical_file: "0" * 64},
        raw_size_by_path={record.canonical_file: 1},
    )


def test_final_guard_raises_before_raw_reader_is_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("sealed-final")
    (tmp_path / record.canonical_file).write_text("x", encoding="utf-8")
    called = False

    def forbidden_reader(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("raw reader must not be reached")

    monkeypatch.setattr(subject, "_read_raw_game", forbidden_reader)
    access = FrozenSplitAccess(final_sessions=frozenset({record.session_id}))

    with pytest.raises(FinalTestAccessError, match="final-test is sealed"):
        access.read_raw_game(tmp_path, record, _raw_boundary(record))

    assert called is False


def test_final_guard_rejects_decisions_and_features_without_fallback(
    tmp_path: Path,
) -> None:
    record = _record("sealed-final")
    access = FrozenSplitAccess(final_sessions=frozenset({record.session_id}))
    produced = False

    def producer() -> str:
        nonlocal produced
        produced = True
        return "forbidden"

    with pytest.raises(FinalTestAccessError, match="decision load"):
        access.load_decisions(tmp_path, record, _raw_boundary(record))
    with pytest.raises(FinalTestAccessError, match="derived feature load"):
        access.derive_features(record.session_id, producer)

    assert produced is False


def test_nonfinal_feature_producer_is_allowed() -> None:
    access = FrozenSplitAccess(final_sessions=frozenset({"sealed-final"}))
    assert access.derive_features("train-session", lambda: 17) == 17


def test_cost_projection_uses_frozen_upper_mean_and_safety_multiplier() -> None:
    result = project_cost(
        total_decisions=1_000,
        state_build_seconds=2.0,
        sample_decisions=100,
        query_counts=[3, 5, 4, 4],
        query_seconds=[0.03, 0.05, 0.04, 0.04],
        safety_multiplier=1.25,
    )

    assert result["logical_decisions"] == 1_000
    assert result["queries_per_decision_upper_mean_95"] > 4.0
    assert result["projected_queries"] > 5_000
    assert result["projected_active_seconds"] > 75.0


def test_cost_projection_fails_closed_on_one_observation() -> None:
    with pytest.raises(B2FreezeError, match="insufficient observations"):
        project_cost(
            total_decisions=100,
            state_build_seconds=1.0,
            sample_decisions=2,
            query_counts=[3],
            query_seconds=[0.1],
            safety_multiplier=1.25,
        )


def test_characterization_keeps_strict_outcome_base_explicit() -> None:
    games = [
        _game("g1", "a", "b", "2026-02-01", moves=18),
        _game("g2", "a", "c", "2026-02-02", moves=45),
        _game("g3", "d", "a", "2026-02-03", moves=101),
    ]
    result = characterize_metadata(
        games,
        {"g1": "W", "g2": "D", "g3": None},
    )

    assert result["games"] == 3
    assert result["player_keys"] == 4
    assert result["logical_plies"] == 164
    assert result["strict_outcome_eligible_games"] == 2
    assert result["strict_outcome_distribution"] == {"W": 1, "B": 0, "D": 1}
    assert result["game_length_logical_plies"]["bins"] == {
        "1-18": 1,
        "19-40": 0,
        "41-60": 1,
        "61-100": 0,
        "101+": 1,
    }
