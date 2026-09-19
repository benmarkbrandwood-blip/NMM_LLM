"""
Stage 1 tests for PostGameAssessor (heuristic signal only).

Uses difficulty=1, depth=3 and a short synthetic game record to keep
each test fast (no Sentinel, Malom, or trajectory dependencies).
"""

import pytest
from game.board import BoardState
from game.rules import get_all_legal_moves
from ai.post_game_assessor import PostGameAssessor, MoveAnnotation, PostGameAnnotation


# ── Synthetic game record builder ─────────────────────────────────────────────

def _move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


def build_game_record(n_plies: int = 6) -> dict:
    """Play n_plies legal moves from the opening position and return a game record.

    Moves are chosen as the first legal move at each position — deterministic and fast.
    The game record format matches what GameDebriefer / app.py produce.
    """
    board = BoardState.new_game()
    moves_raw = []
    for ply in range(n_plies):
        legal = get_all_legal_moves(board)
        if not legal:
            break
        move = legal[0]
        notation = _move_notation(move)
        moves_raw.append({
            "from": move.get("from"),
            "to": move["to"],
            "capture": move.get("capture"),
            "color": board.turn,
            "type": "placement",
            "turn": ply,
            "notation": notation,
            "board_fen_before": "",
            "was_blunder": False,
        })
        board = board.apply_move(move)

    return {
        "winner": None,
        "human_color": "W",
        "moves": moves_raw,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def assessor():
    return PostGameAssessor(difficulty=1, depth=3)


@pytest.fixture(scope="module")
def annotation(assessor):
    record = build_game_record(n_plies=6)
    return assessor.assess(record)


# ── Stage 1 tests ─────────────────────────────────────────────────────────────

class TestPostGameAssessorStage1:
    def test_returns_post_game_annotation(self, annotation):
        assert isinstance(annotation, PostGameAnnotation)

    def test_move_count(self, annotation):
        assert len(annotation.moves) == 6

    def test_all_moves_are_annotations(self, annotation):
        for ann in annotation.moves:
            assert isinstance(ann, MoveAnnotation)

    def test_heuristic_curve_length(self, annotation):
        assert len(annotation.heuristic_curve) == len(annotation.moves)

    def test_sentinel_curve_length(self, annotation):
        assert len(annotation.sentinel_curve) == len(annotation.moves)

    def test_sentinel_curve_all_none_stage1(self, annotation):
        """Stage 1 has no Sentinel; all sentinel_curve entries must be None."""
        assert all(v is None for v in annotation.sentinel_curve)

    def test_r_h_non_negative(self, annotation):
        for ann in annotation.moves:
            assert ann.r_h >= 0.0, f"ply {ann.ply}: r_h={ann.r_h} < 0"

    def test_r_h_at_most_one(self, annotation):
        for ann in annotation.moves:
            assert ann.r_h <= 1.0 + 1e-9, f"ply {ann.ply}: r_h={ann.r_h} > 1"

    def test_score_played_in_range(self, annotation):
        for ann in annotation.moves:
            assert 0.0 <= ann.score_played <= 1.0 + 1e-9, \
                f"ply {ann.ply}: score_played={ann.score_played}"

    def test_score_best_is_one(self, annotation):
        for ann in annotation.moves:
            assert ann.score_best == 1.0

    def test_r_h_equals_one_minus_score_played(self, annotation):
        for ann in annotation.moves:
            assert abs(ann.r_h - (1.0 - ann.score_played)) < 1e-9, \
                f"ply {ann.ply}: r_h={ann.r_h}, 1-score_played={1-ann.score_played}"

    def test_ply_indices(self, annotation):
        for i, ann in enumerate(annotation.moves):
            assert ann.ply == i

    def test_colors_alternate(self, annotation):
        colors = [ann.color for ann in annotation.moves]
        for i in range(1, len(colors)):
            assert colors[i] != colors[i - 1], \
                f"Expected alternating colors at ply {i}: {colors}"

    def test_turning_point_oracle_is_heuristic(self, annotation):
        assert annotation.turning_point_oracle == "heuristic"

    def test_turning_point_ply_in_range(self, annotation):
        if annotation.turning_point_ply is not None:
            assert 0 <= annotation.turning_point_ply < len(annotation.moves)

    def test_sentinel_fields_none(self, annotation):
        for ann in annotation.moves:
            assert ann.sentinel_score_white is None
            assert ann.sentinel_played is None
            assert ann.sentinel_best is None
            assert ann.r_s is None

    def test_malom_fields_default(self, annotation):
        for ann in annotation.moves:
            assert ann.wdl_before is None
            assert ann.wdl_after is None
            assert ann.oracle_source == "none"
            assert ann.quality == "clean"

    def test_trajectory_fields_none(self, annotation):
        for ann in annotation.moves:
            assert ann.traj_delta_played is None
            assert ann.traj_delta_best is None
            assert ann.r_t is None

    def test_policy_fields_none(self, annotation):
        for ann in annotation.moves:
            assert ann.policy_prob is None
            assert ann.policy_top_move is None
            assert not ann.is_unconventional

    def test_heuristic_curve_values_are_floats(self, annotation):
        for v in annotation.heuristic_curve:
            assert isinstance(v, float)

    def test_move_played_strings_nonempty(self, annotation):
        for ann in annotation.moves:
            assert ann.move_played, f"ply {ann.ply}: empty move_played"

    def test_best_alt_is_string_or_none(self, annotation):
        for ann in annotation.moves:
            assert ann.best_alt is None or isinstance(ann.best_alt, str)

    def test_turning_point_quality_format(self, annotation):
        if annotation.turning_point_ply is not None:
            assert annotation.turning_point_quality.startswith("r_h:")

    def test_empty_game_record(self, assessor):
        result = assessor.assess({"moves": []})
        assert result.moves == []
        assert result.heuristic_curve == []
        assert result.sentinel_curve == []
        assert result.turning_point_ply is None
