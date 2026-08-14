from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.evaluation.human_f0h0_split_retest import (
    PARTITIONS,
    GameRecord,
    build_player_graph,
    decision_player_partition,
    detect_communities,
    graph_structure,
    measure_design_a,
    measure_design_c_counts,
    measure_design_c_ring16,
    nearest_rank_quantiles,
    time_structure,
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


def _fixture_games() -> list[GameRecord]:
    return [
        _game("g1", "a", "b", played_on="2026-01-01"),
        _game("g2", "b", "c", played_on="2026-01-08"),
        _game("g3", "a", "c", played_on="2026-02-01"),
        _game("g4", "d", "e", played_on="2026-02-08"),
        _game("g5", "e", "f", played_on="2026-03-01"),
        _game("g6", "d", "f", played_on="2026-03-08"),
        _game("g7", "c", "d", played_on="2026-04-01"),
    ]


def test_graph_and_time_measurements_are_exact() -> None:
    games = _fixture_games()
    data = build_player_graph(games)
    structure = graph_structure(data, games)
    assert structure["connected_components"] == 1
    assert structure["giant_component"]["player_keys"] == 6
    assert structure["giant_component"]["games"] == 7
    assert structure["degree"]["distinct_opponents"]["quantiles"]["max"] == 3
    assert nearest_rank_quantiles([1, 2, 3, 4])["p50"] == 2

    measured = time_structure(games, cut_dates=["2026-03-01"])
    cut = measured["candidate_cuts"][0]
    assert cut["pre_games"] == 4
    assert cut["post_games_time_only"] == 3
    assert cut["player_classes"] == {
        "pre_only": 2,
        "post_only": 1,
        "spanning": 3,
    }
    assert cut["post_games_both_players_unseen_pre_cut"] == 0


def test_design_a_keeps_target_size_and_accounts_every_game() -> None:
    games = _fixture_games()
    data = build_player_graph(games)
    communities, _summary = detect_communities(data, seed=7, resolution=1.0)
    measurements = measure_design_a(
        data,
        games,
        communities,
        targets=[1 / 3],
        seed="fixture-a",
        restarts=3,
        maximum_kl_iterations=10,
    )
    row = measurements[0]
    assert row["measured_holdout_players"] == 2
    assert (
        row["holdout_internal_games"]
        + row["train_internal_games"]
        + row["cross_cut_discard_games"]
        == len(games)
    )
    assert row["cross_cut_discard_fraction"] == (
        row["cross_cut_discard_games"] / len(games)
    )


def test_design_c_assigns_each_decision_to_its_actor() -> None:
    games = _fixture_games()
    players = {player for game in games for player in game.players}
    ratios = {
        "train": 0.5,
        "selection": 1 / 6,
        "one-time-confirmation": 1 / 6,
        "final-test": 1 / 6,
    }
    membership, summary = decision_player_partition(
        players,
        ratios=ratios,
        seed="fixture-c",
    )
    counts = measure_design_c_counts(games, membership)
    assert sum(summary["player_counts"].values()) == len(players)
    assert sum(
        row["decisions"] for row in counts["partition_counts"].values()
    ) == sum(game.move_count for game in games)
    assert (
        counts["same_partition_games"] + counts["cross_partition_games"]
        == len(games)
    )
    assert set(summary["player_counts"]) == set(PARTITIONS)


def _notation(move: dict) -> str:
    source = move.get("from")
    target = move.get("to")
    base = str(target) if source is None else f"{source}-{target}"
    capture = move.get("capture")
    return base if capture is None else f"{base}x{capture}"


def _write_raw_game(path: Path) -> tuple[int, str]:
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
    raw = (json.dumps({"session_id": "raw-game", "moves": moves}) + "\n").encode()
    path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_design_c_ring16_replay_reports_trajectory_distance(tmp_path: Path) -> None:
    size, digest = _write_raw_game(tmp_path / "human_raw-game.jsonl")
    game = GameRecord(
        session_id="raw-game",
        canonical_file="human_raw-game.jsonl",
        file_sha256=digest,
        file_size=size,
        played_on=date(2026, 1, 1),
        move_count=4,
        white_player="white",
        black_player="black",
    )
    measured = measure_design_c_ring16(
        repository_root=tmp_path,
        games=[game],
        membership={"white": "train", "black": "selection"},
    )
    assert measured["strict_replayed_games"] == 1
    assert measured["strict_replayed_decisions"] == 4
    pair = next(
        row
        for row in measured["pairwise_overlap"]
        if row["left"] == "train" and row["right"] == "selection"
    )
    assert pair["cross_partition_trajectory_games"] == 1
    distances = pair["cross_partition_trajectory_decision_pairs_by_ply_distance"]
    assert distances["1"] == 3
    assert distances["3-4"] == 1
