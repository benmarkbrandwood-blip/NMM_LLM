from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.evaluation.human_f0h0_design_b_supplement import (
    B2Candidate,
    build_b2_profiles,
    build_random_baseline,
    measure_b1_three_way,
    measure_support_metadata,
    replay_required_games,
    summarize_overlap,
)
from learned_ai.evaluation.human_f0h0_split_retest import (
    GameRecord,
    build_player_graph,
)


def _game(
    session: str,
    white: str,
    black: str,
    *,
    played_on: str,
    moves: int = 4,
) -> GameRecord:
    return GameRecord(
        session_id=session,
        canonical_file=f"human_{session}.jsonl",
        file_sha256="0" * 64,
        file_size=0,
        played_on=date.fromisoformat(played_on),
        move_count=moves,
        white_player=white,
        black_player=black,
    )


def _time_fixture() -> list[GameRecord]:
    return [
        _game("g0", "a", "b", played_on="2026-01-01"),
        _game("g1", "c", "d", played_on="2026-03-10"),
        _game("g2", "c", "d", played_on="2026-04-05"),
        _game("g3", "e", "f", played_on="2026-04-06"),
        _game("g4", "g", "h", played_on="2026-06-01"),
        _game("g5", "a", "b", played_on="2026-05-15"),
    ]


def test_support_metadata_keeps_behavior_and_outcome_bases_separate() -> None:
    games = _time_fixture()
    outcomes = {game.session_id: game.session_id in {"g1", "g4"} for game in games}
    measured, subsets = measure_support_metadata(
        games,
        outcomes,
        cut_dates=["2026-03-01", "2026-05-01"],
    )

    march = measured["2026-03-01"]
    assert march["train"]["games"] == 1
    assert march["train"]["decisions"] == 4
    assert march["strong_post"]["games"] == 4
    assert march["strong_post"]["player_keys"] == 6
    assert march["strong_post"]["outcome_eligible_games"] == 2
    assert {game.session_id for game in subsets["support::2026-03-01::strong"]} == {
        "g1",
        "g2",
        "g3",
        "g4",
    }

    may = measured["2026-05-01"]
    assert may["train"]["games"] == 4
    assert may["strong_post"]["games"] == 1
    assert may["strong_post"]["outcome_eligible_games"] == 1


def test_b2_segments_the_march_pool_then_reapplies_player_novelty() -> None:
    games = _time_fixture()
    outcomes = {game.session_id: game.session_id in {"g1", "g4"} for game in games}
    candidate = B2Candidate(
        candidate_id="fixture",
        cut_one=date(2026, 4, 1),
        cut_two=date(2026, 6, 1),
    )
    measured, profiles, subsets = build_b2_profiles(
        games,
        outcomes,
        candidates=[candidate],
    )

    row = measured["fixture"]
    assert row["march_test_pool_games"] == 4
    assert row["segments"]["selection"]["all_segment_games"] == 1
    assert row["segments"]["one-time-confirmation"]["all_segment_games"] == 2
    assert row["segments"]["final-test"]["all_segment_games"] == 1
    assert sum(
        segment["all_segment_games"] for segment in row["segments"].values()
    ) == row["march_test_pool_games"]

    assert set(profiles["fixture"].values()) == {
        "train",
        "selection",
        "one-time-confirmation",
        "final-test",
    }
    assert {game.session_id for game in subsets["b2::fixture::selection"]} == {
        "g1"
    }
    assert {
        game.session_id
        for game in subsets["b2::fixture::one-time-confirmation"]
    } == {"g3"}
    assert {game.session_id for game in subsets["b2::fixture::final-test"]} == {
        "g4"
    }


def test_random_baseline_is_disjoint_exact_and_reproducible() -> None:
    games = _time_fixture()
    first, first_summary = build_random_baseline(
        games,
        left_games=3,
        right_games=2,
        seed="fixture-random",
    )
    second, second_summary = build_random_baseline(
        list(reversed(games)),
        left_games=3,
        right_games=2,
        seed="fixture-random",
    )

    assert first == second
    assert first_summary == second_summary
    assert list(first.values()).count("random-left") == 3
    assert list(first.values()).count("random-right") == 2
    assert len(first) == 5


def test_three_way_player_cut_accounts_for_every_game() -> None:
    games = [
        _game(f"g{index}", f"p{index}", f"p{(index + 1) % 8}", played_on="2026-03-10")
        for index in range(8)
    ]
    graph = build_player_graph(games)
    measured = measure_b1_three_way(
        graph,
        games,
        ratios={
            "selection": 0.5,
            "one-time-confirmation": 0.25,
            "final-test": 0.25,
        },
        seed="fixture-b1",
        outer_restarts=2,
        inner_restarts=2,
        max_iterations=10,
        louvain_seed=17,
    )

    assert measured["player_counts"] == {
        "selection": 4,
        "one-time-confirmation": 2,
        "final-test": 2,
    }
    assert (
        sum(measured["internal_games"].values())
        + measured["cross_partition_discard_games"]
        == len(games)
    )


def test_overlap_is_decision_weighted_and_pairwise() -> None:
    # Two groups share orbit x; only left sees orbit y.
    packed = {
        "x": (1 | (3 << 2)) | (2 | (1 << 34)),
        "y": 1 | (2 << 2),
    }
    result = summarize_overlap(packed, ("left", "right"))
    rows = {row["partition"]: row for row in result["partition_overlap"]}

    assert rows["left"]["decisions"] == 5
    assert rows["left"]["decisions_on_shared_ring16_orbits"] == 3
    assert rows["left"]["decision_weighted_overlap_rate"] == 3 / 5
    assert rows["right"]["decision_weighted_overlap_rate"] == 1.0
    assert result["pairwise_overlap"][0]["shared_unique_ring16_orbits"] == 1


def _notation(move: dict) -> str:
    source = move.get("from")
    target = move.get("to")
    base = str(target) if source is None else f"{source}-{target}"
    capture = move.get("capture")
    return base if capture is None else f"{base}x{capture}"


def _write_raw_game(path: Path, session_id: str) -> tuple[int, str]:
    board = BoardState.new_game()
    moves = []
    for logical_ply in range(4):
        move = dict(get_all_legal_moves(board)[logical_ply])
        moves.append(
            {
                "board_fen_before": board.to_fen_string(),
                "color": board.turn,
                "turn": logical_ply // 2 + 1,
                "type": get_game_phase(board, board.turn),
                "from": move.get("from"),
                "to": move.get("to"),
                "capture": move.get("capture"),
                "notation": _notation(move),
            }
        )
        board = board.apply_move(move)
    raw = (json.dumps({"session_id": session_id, "moves": moves}) + "\n").encode()
    path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_raw_replay_measures_phase_and_same_metric_ring16(tmp_path: Path) -> None:
    games = []
    for index, group in enumerate(("left", "right")):
        session = f"raw-{index}"
        filename = f"human_{session}.jsonl"
        size, digest = _write_raw_game(tmp_path / filename, session)
        games.append(
            GameRecord(
                session_id=session,
                canonical_file=filename,
                file_sha256=digest,
                file_size=size,
                played_on=date(2026, 3, index + 1),
                move_count=4,
                white_player=f"{group}-white",
                black_player=f"{group}-black",
            )
        )
    replay = replay_required_games(
        repository_root=tmp_path,
        games=games,
        outcome_eligible={game.session_id: False for game in games},
        phase_subsets={"fixture": games},
        profile_memberships={
            "fixture": {"raw-0": "left", "raw-1": "right"}
        },
        profile_groups={"fixture": ("left", "right")},
    )

    assert replay["raw_files_opened"] == 2
    assert replay["strict_replayed_decisions"] == 8
    assert replay["phase_counts"]["fixture"] == {
        "placement": 8,
        "movement": 0,
        "flying": 0,
    }
    overlap = replay["ring16_profiles"]["fixture"]["partition_overlap"]
    assert all(row["decision_weighted_overlap_rate"] == 1.0 for row in overlap)
