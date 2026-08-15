from __future__ import annotations

import math

from game.board import POSITIONS, BoardState
from learned_ai.evaluation import human_feature_deviation as v1
from learned_ai.evaluation import human_feature_deviation_design_round as design


def test_fixed_size_cut_preserves_locked_exploration_player() -> None:
    graph = design.nx.Graph()
    graph.add_edge("a", "b", games=10)
    graph.add_edge("b", "c", games=1)
    graph.add_edge("c", "d", games=10)

    result, _iterations = design._improve_fixed_size_cut(
        graph,
        {"c", "d"},
        locked_exploration={"a"},
        maximum_iterations=10,
        shortlist=4,
    )

    assert len(result) == 2
    assert "a" not in result


def test_precision_screen_does_not_claim_mathematical_impossibility() -> None:
    metrics = {
        "research-confirmation": {
            "participating_players": 290,
            "player_decision_concentration": {"kish_effective_units": 46.78},
        }
    }

    result = design._precision_analysis(metrics)

    assert result["mathematically_impossible_claim"] is False
    assert result["structural_reachability_decision"].startswith("not_certified")
    coefficient = result["equal_player_upper_structural_count"][
        "mde_per_unit_player_level_standard_deviation"
    ]
    assert math.isclose(coefficient, 2.8015852181129683 / math.sqrt(290))


def test_material_and_mill_are_affine_within_atomic_choice_set() -> None:
    board = BoardState.new_game()
    rows = [
        v1.action_feature_scores(
            board,
            {"from": None, "to": destination, "capture": None},
        )
        for destination in POSITIONS
    ]
    audit = design.CollinearityAudit()

    audit.observe(rows)

    assert audit.exact_affine_choice_sets == 1
    assert audit.maximum_within_choice_residual_range == 0.0


def test_simultaneous_double_mill_is_possible_but_distinct_from_fork() -> None:
    positions = {position: "" for position in POSITIONS}
    for position in ("a7", "g7", "d6", "d5"):
        positions[position] = "W"
    board = BoardState.from_setup(positions, turn="W", phase="place")
    move = {"from": None, "to": "d7", "capture": None}

    old = v1.action_feature_scores(board, move)
    new = design.extended_action_feature_scores(board, move)

    assert old["creates_double_mill"] == 1.0
    assert new["closes_mill"] == 1.0
    assert tuple(new) == design.V2_FEATURE_NAMES


def test_extended_feature_panel_varies_on_opening_geometry() -> None:
    board = BoardState.new_game()
    rows = [
        design.extended_action_feature_scores(
            board,
            {"from": None, "to": destination, "capture": None},
        )
        for destination in POSITIONS
    ]

    assert all(tuple(row) == design.V2_FEATURE_NAMES for row in rows)
    assert len({row["destination_degree"] for row in rows}) > 1
    assert all(math.isfinite(value) for row in rows for value in row.values())


def test_oracle_inventory_contract_is_board_then_database(monkeypatch) -> None:
    observed: list[tuple[object, object]] = []

    def fake_inventory(board: object, database: object) -> tuple[str, list, int]:
        observed.append((board, database))
        return "D", [], 0

    monkeypatch.setattr(design, "_oracle_inventory_positional", fake_inventory)
    board = BoardState.new_game()
    database = object()

    design._query_inventory(board, database)

    assert observed == [(board, database)]
