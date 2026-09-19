"""
Stage 1 + Stage 2 tests for PostGameAssessor.

Stage 1: heuristic signal only.
Stage 2: Sentinel integration via a lightweight mock advisor.

Uses difficulty=1, depth=3 and a short synthetic game record to keep
each test fast.
"""

import pytest
from unittest.mock import MagicMock
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

    def test_generalist_fields_none_stage1(self, annotation):
        for ann in annotation.moves:
            assert ann.generalist_policy_prob is None
            assert ann.generalist_top_move is None
            assert ann.generalist_value_after is None


# ── Stage 2: Sentinel integration ─────────────────────────────────────────────

def _make_mock_sentinel(played_quality: float = 0.6, best_quality: float = 0.8):
    """Create a mock SentinelAdvisor that returns fixed scores."""
    from learned_ai.sentinel.infer import SentinelAdvice
    advice = SentinelAdvice(
        move_scores=[played_quality, best_quality],
        best_sentinel_move_idx=1,
        played_move_idx=0,
        played_move_quality=played_quality,
        best_available_quality=best_quality,
        opportunity_gap=max(0.0, best_quality - played_quality),
        player="W",
        advisory_message="missed_opportunity",
    )
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.advise.return_value = advice
    return mock


@pytest.fixture(scope="module")
def sentinel_assessor():
    return PostGameAssessor(difficulty=1, depth=3, sentinel=_make_mock_sentinel())


@pytest.fixture(scope="module")
def sentinel_annotation(sentinel_assessor):
    record = build_game_record(n_plies=6)
    return sentinel_assessor.assess(record)


class TestPostGameAssessorStage2:
    def test_sentinel_curve_populated(self, sentinel_annotation):
        assert len(sentinel_annotation.sentinel_curve) == len(sentinel_annotation.moves)
        assert any(v is not None for v in sentinel_annotation.sentinel_curve)

    def test_sentinel_played_populated(self, sentinel_annotation):
        for ann in sentinel_annotation.moves:
            assert ann.sentinel_played is not None, f"ply {ann.ply}: sentinel_played is None"

    def test_sentinel_best_populated(self, sentinel_annotation):
        for ann in sentinel_annotation.moves:
            assert ann.sentinel_best is not None

    def test_r_s_non_negative(self, sentinel_annotation):
        for ann in sentinel_annotation.moves:
            assert ann.r_s is not None
            assert ann.r_s >= 0.0, f"ply {ann.ply}: r_s={ann.r_s} < 0"

    def test_r_s_equals_gap(self, sentinel_annotation):
        for ann in sentinel_annotation.moves:
            expected = max(0.0, ann.sentinel_best - ann.sentinel_played)
            assert abs(ann.r_s - expected) < 1e-9

    def test_sentinel_score_white_normalised_white_ply(self, sentinel_annotation):
        """For a White ply sentinel_score_white == sentinel_played."""
        white_plies = [a for a in sentinel_annotation.moves if a.color == "W"]
        for ann in white_plies:
            assert abs(ann.sentinel_score_white - ann.sentinel_played) < 1e-9, \
                f"ply {ann.ply}: sentinel_score_white={ann.sentinel_score_white}, played={ann.sentinel_played}"

    def test_sentinel_score_white_normalised_black_ply(self, sentinel_annotation):
        """For a Black ply sentinel_score_white == 1 - sentinel_played."""
        black_plies = [a for a in sentinel_annotation.moves if a.color == "B"]
        for ann in black_plies:
            expected = 1.0 - ann.sentinel_played
            assert abs(ann.sentinel_score_white - expected) < 1e-9, \
                f"ply {ann.ply}: sentinel_score_white={ann.sentinel_score_white}, expected={expected}"

    def test_sentinel_curve_matches_annotations(self, sentinel_annotation):
        for i, ann in enumerate(sentinel_annotation.moves):
            assert sentinel_annotation.sentinel_curve[i] == ann.sentinel_score_white

    def test_heuristic_fields_unaffected(self, sentinel_annotation):
        """Heuristic fields must be unchanged when Sentinel is present."""
        for ann in sentinel_annotation.moves:
            assert ann.r_h >= 0.0
            assert ann.r_h <= 1.0 + 1e-9
            assert ann.score_best == 1.0

    def test_no_sentinel_gives_none_fields(self):
        """Without a sentinel advisor, all sentinel fields remain None."""
        assessor_no_sentinel = PostGameAssessor(difficulty=1, depth=3, sentinel=None)
        record = build_game_record(n_plies=4)
        result = assessor_no_sentinel.assess(record)
        for ann in result.moves:
            assert ann.sentinel_score_white is None
            assert ann.sentinel_played is None
            assert ann.r_s is None
        assert all(v is None for v in result.sentinel_curve)

    def test_oracle_still_heuristic_stage2(self, sentinel_annotation):
        """Turning point oracle stays 'heuristic' until Stage 5."""
        assert sentinel_annotation.turning_point_oracle == "heuristic"


# ── Stage 2: GapNet blunder-zone density ──────────────────────────────────────

def _make_mock_gap_net(raw_output: float = 0.4):
    """Mock gap_net with fixed tanh output → blunder_zone_score = (raw+1)/2."""
    mock = MagicMock()
    mock.predict.return_value = raw_output
    return mock


@pytest.fixture(scope="module")
def gap_net_assessor():
    return PostGameAssessor(
        difficulty=1, depth=3,
        gap_net=_make_mock_gap_net(raw_output=0.4),
    )


@pytest.fixture(scope="module")
def gap_net_annotation(gap_net_assessor):
    record = build_game_record(n_plies=6)
    return gap_net_assessor.assess(record)


class TestPostGameAssessorGapNet:
    def test_blunder_zone_populated(self, gap_net_annotation):
        for ann in gap_net_annotation.moves:
            assert ann.blunder_zone_score is not None, \
                f"ply {ann.ply}: blunder_zone_score is None"

    def test_blunder_zone_in_range(self, gap_net_annotation):
        for ann in gap_net_annotation.moves:
            assert 0.0 <= ann.blunder_zone_score <= 1.0, \
                f"ply {ann.ply}: blunder_zone_score={ann.blunder_zone_score} out of [0,1]"

    def test_blunder_zone_conversion(self, gap_net_annotation):
        """(raw+1)/2 conversion: raw=0.4 → score=0.7."""
        expected = (0.4 + 1.0) / 2.0
        for ann in gap_net_annotation.moves:
            assert abs(ann.blunder_zone_score - expected) < 1e-6

    def test_blunder_zone_none_without_gap_net(self):
        assessor = PostGameAssessor(difficulty=1, depth=3, gap_net=None)
        record = build_game_record(n_plies=4)
        result = assessor.assess(record)
        for ann in result.moves:
            assert ann.blunder_zone_score is None

    def test_gap_net_predict_called_with_board_and_color(self):
        mock_gn = _make_mock_gap_net()
        assessor = PostGameAssessor(difficulty=1, depth=3, gap_net=mock_gn)
        record = build_game_record(n_plies=4)
        assessor.assess(record)
        assert mock_gn.predict.call_count == 4
        for call in mock_gn.predict.call_args_list:
            board_arg, color_arg = call[0]
            assert isinstance(board_arg, BoardState)
            assert color_arg in ("W", "B")

    def test_boundary_raw_minus_one(self):
        """raw=-1.0 → blunder_zone_score=0.0 (floor)."""
        mock_gn = _make_mock_gap_net(raw_output=-1.0)
        assessor = PostGameAssessor(difficulty=1, depth=3, gap_net=mock_gn)
        result = assessor.assess(build_game_record(n_plies=2))
        for ann in result.moves:
            assert abs(ann.blunder_zone_score - 0.0) < 1e-6

    def test_boundary_raw_plus_one(self):
        """raw=+1.0 → blunder_zone_score=1.0 (ceiling)."""
        mock_gn = _make_mock_gap_net(raw_output=1.0)
        assessor = PostGameAssessor(difficulty=1, depth=3, gap_net=mock_gn)
        result = assessor.assess(build_game_record(n_plies=2))
        for ann in result.moves:
            assert abs(ann.blunder_zone_score - 1.0) < 1e-6

    def test_sentinel_unaffected_by_gap_net(self, gap_net_annotation):
        """Adding gap_net must not change sentinel fields (they stay None here)."""
        for ann in gap_net_annotation.moves:
            assert ann.sentinel_played is None
            assert ann.r_s is None

    def test_combined_sentinel_and_gap_net(self):
        """Both can be active simultaneously without interfering."""
        assessor = PostGameAssessor(
            difficulty=1, depth=3,
            sentinel=_make_mock_sentinel(),
            gap_net=_make_mock_gap_net(raw_output=0.6),
        )
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.sentinel_played is not None
            assert ann.blunder_zone_score is not None
            assert abs(ann.blunder_zone_score - (0.6 + 1.0) / 2.0) < 1e-6


# ── Stage 3: Malom adjudication ───────────────────────────────────────────────

def _make_mock_malom(wdl_before: str = "W", wdl_after: str = "W",
                     transition: str = "win_preserved", available: bool = True,
                     unavailable_reason: str = None):
    """Build a mock MalomDB that returns fixed values for every position."""
    parent_val = MagicMock()
    parent_val.outcome = wdl_before

    omv = MagicMock()
    omv.outcome = wdl_after

    regret = MagicMock()
    regret.available = available
    regret.omv = omv
    regret.wdl_transition = transition
    regret.unavailable_reason = unavailable_reason

    mock = MagicMock()
    mock.is_available.return_value = True
    mock.query_value.return_value = parent_val
    mock.query_regret.return_value = regret
    return mock


class TestPostGameAssessorStage3:
    def test_no_malom_leaves_fields_default(self):
        assessor = PostGameAssessor(difficulty=1, depth=3)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.wdl_before is None
            assert ann.wdl_after is None
            assert ann.oracle_source == "none"
            assert ann.quality == "clean"

    def test_win_preserved_is_clean(self):
        malom = _make_mock_malom(wdl_before="W", wdl_after="W",
                                 transition="win_preserved")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.oracle_source == "malom_full"
            assert ann.quality == "clean"
            assert ann.wdl_before == "W"
            assert ann.wdl_after == "W"

    def test_draw_preserved_is_clean(self):
        malom = _make_mock_malom(wdl_before="D", wdl_after="D",
                                 transition="draw_preserved")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.quality == "clean"

    def test_win_to_loss_is_confirmed_poor(self):
        malom = _make_mock_malom(wdl_before="W", wdl_after="L",
                                 transition="win_to_loss")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.quality == "confirmed_poor"
            assert ann.oracle_source == "malom_full"

    def test_win_to_draw_is_confirmed_poor(self):
        malom = _make_mock_malom(wdl_before="W", wdl_after="D",
                                 transition="win_to_draw")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.quality == "confirmed_poor"

    def test_draw_to_loss_is_confirmed_poor(self):
        malom = _make_mock_malom(wdl_before="D", wdl_after="L",
                                 transition="draw_to_loss")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.quality == "confirmed_poor"

    def test_already_losing_is_skipped(self):
        """wdl_before=L → abstained, quality clean, wdl fields preserved."""
        malom = _make_mock_malom(wdl_before="L", wdl_after="L",
                                 transition="all_losing")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.quality == "clean"
            assert ann.abstained_reason == "already_losing"
            assert ann.oracle_source == "none"

    def test_unavailable_malom_sets_abstained(self):
        malom = _make_mock_malom(available=False,
                                 unavailable_reason="parent_value_unavailable")
        # query_value returns None to trigger the unavailable path
        malom.query_value.return_value = None
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.wdl_before is None
            assert ann.oracle_source == "none"
            assert ann.abstained_reason == "parent_value_unavailable"

    def test_label_inconsistency_fails_closed(self):
        malom = _make_mock_malom(wdl_before="W", wdl_after="W",
                                 transition="label_inconsistency")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.oracle_source == "none"
            assert ann.wdl_before is None
            assert ann.wdl_after is None
            assert "label_inconsistency" in (ann.abstained_reason or "")

    def test_malom_not_available_leaves_fields_default(self):
        malom = MagicMock()
        malom.is_available.return_value = False
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.wdl_before is None
            assert ann.oracle_source == "none"

    def test_wdl_values_are_valid_strings(self):
        malom = _make_mock_malom(wdl_before="W", wdl_after="D",
                                 transition="win_to_draw")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            if ann.wdl_before is not None:
                assert ann.wdl_before in ("W", "D", "L")
            if ann.wdl_after is not None:
                assert ann.wdl_after in ("W", "D", "L")

    def test_malom_called_with_board_before_move(self):
        """query_value must be called with the board BEFORE apply_move."""
        malom = _make_mock_malom(transition="win_preserved")
        assessor = PostGameAssessor(difficulty=1, depth=3, malom_db=malom)
        assessor.assess(build_game_record(n_plies=4))
        assert malom.query_value.call_count == 4
        for call in malom.query_value.call_args_list:
            board_arg = call[0][0]
            assert isinstance(board_arg, BoardState)


# ── Stage 4: Trajectory, Human policy, Generalist AI ─────────────────────────

import numpy as _np


def _make_mock_trajectory(notation_to_delta: dict):
    """Mock TrajectoryDB.query() returning a fixed {notation: delta} dict."""
    mock = MagicMock()
    mock.query.return_value = notation_to_delta
    return mock


def _make_mock_policy(top_idx: int = 0):
    """Mock HumanMovePolicyAdvisor.probs() that sizes its output to len(candidates)."""
    def _side_effect(board, candidates, elo_band):
        n = len(candidates)
        probs = _np.ones(n, dtype=_np.float32) * (0.4 / max(n - 1, 1))
        probs[min(top_idx, n - 1)] = 0.6
        probs /= probs.sum()
        return probs
    mock = MagicMock()
    mock.probs.side_effect = _side_effect
    return mock


def _make_mock_generalist(top_idx: int = 0):
    """Mock GeneralistAgent.score_moves() that sizes its output to len(candidates)."""
    def _side_effect(board, candidates, color):
        n = len(candidates)
        scores = [0.2] * n
        scores[min(top_idx, n - 1)] = 0.6
        return scores
    mock = MagicMock()
    mock.score_moves.side_effect = _side_effect
    return mock


class TestPostGameAssessorStage4Trajectory:
    def test_trajectory_fields_populated_when_covered(self):
        record = build_game_record(n_plies=4)
        # Plant a delta for the notation of the first played move
        first_notation = record["moves"][0]["notation"]
        traj = _make_mock_trajectory({first_notation: 0.3, "other": 0.1})
        assessor = PostGameAssessor(difficulty=1, depth=3, trajectory_db=traj)
        result = assessor.assess(record)
        ann = result.moves[0]
        assert ann.traj_delta_best == 0.3
        assert ann.traj_delta_played == 0.3
        assert abs(ann.r_t - 0.0) < 1e-9  # played was best

    def test_trajectory_regret_positive_when_not_best(self):
        record = build_game_record(n_plies=4)
        first_notation = record["moves"][0]["notation"]
        traj = _make_mock_trajectory({first_notation: 0.1, "better_move": 0.4})
        assessor = PostGameAssessor(difficulty=1, depth=3, trajectory_db=traj)
        result = assessor.assess(record)
        ann = result.moves[0]
        assert abs(ann.r_t - 0.3) < 1e-6

    def test_trajectory_none_when_no_coverage(self):
        traj = _make_mock_trajectory({})
        assessor = PostGameAssessor(difficulty=1, depth=3, trajectory_db=traj)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.traj_delta_played is None
            assert ann.traj_delta_best is None
            assert ann.r_t is None

    def test_traj_delta_played_none_when_notation_not_in_hints(self):
        # Hints exist but played move's notation is absent (low coverage for that move)
        traj = _make_mock_trajectory({"other_move": 0.2})
        assessor = PostGameAssessor(difficulty=1, depth=3, trajectory_db=traj)
        result = assessor.assess(build_game_record(n_plies=2))
        for ann in result.moves:
            assert ann.traj_delta_best == 0.2
            assert ann.traj_delta_played is None
            assert ann.r_t is None

    def test_no_trajectory_db_leaves_fields_none(self):
        assessor = PostGameAssessor(difficulty=1, depth=3)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.traj_delta_played is None
            assert ann.r_t is None


class TestPostGameAssessorStage4Policy:
    def test_policy_fields_populated(self):
        policy = _make_mock_policy()
        assessor = PostGameAssessor(difficulty=1, depth=3, policy_advisor=policy)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.policy_prob is not None
            assert ann.policy_top_move is not None
            assert ann.policy_top_prob is not None
            assert ann.policy_prob_source == "learned"

    def test_policy_prob_in_range(self):
        policy = _make_mock_policy()
        assessor = PostGameAssessor(difficulty=1, depth=3, policy_advisor=policy)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert 0.0 <= ann.policy_prob <= 1.0 + 1e-6

    def test_policy_top_prob_ge_played_prob(self):
        policy = _make_mock_policy()
        assessor = PostGameAssessor(difficulty=1, depth=3, policy_advisor=policy)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.policy_top_prob >= ann.policy_prob - 1e-6

    def test_no_policy_advisor_leaves_fields_none(self):
        assessor = PostGameAssessor(difficulty=1, depth=3)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.policy_prob is None
            assert ann.policy_top_move is None

    def test_policy_called_with_elo_band(self):
        policy = _make_mock_policy()
        assessor = PostGameAssessor(difficulty=1, depth=3,
                                    policy_advisor=policy, policy_elo_band="upper")
        assessor.assess(build_game_record(n_plies=2))
        for call in policy.probs.call_args_list:
            assert call[0][2] == "upper"


class TestPostGameAssessorStage4Generalist:
    def test_generalist_fields_populated(self):
        gen = _make_mock_generalist()
        assessor = PostGameAssessor(difficulty=1, depth=3, generalist=gen)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.generalist_policy_prob is not None
            assert ann.generalist_top_move is not None

    def test_generalist_self_assessed_flag_ai_side(self):
        """When human_color=W, AI is B — Black plies should be self-assessed."""
        gen = _make_mock_generalist()
        assessor = PostGameAssessor(difficulty=1, depth=3, generalist=gen)
        record = build_game_record(n_plies=4)
        record["human_color"] = "W"
        result = assessor.assess(record)
        for ann in result.moves:
            if ann.color == "B":
                assert ann.generalist_self_assessed
            else:
                assert not ann.generalist_self_assessed

    def test_generalist_self_assessed_false_no_human_color(self):
        gen = _make_mock_generalist()
        assessor = PostGameAssessor(difficulty=1, depth=3, generalist=gen)
        record = build_game_record(n_plies=4)
        record.pop("human_color", None)
        result = assessor.assess(record)
        for ann in result.moves:
            assert not ann.generalist_self_assessed

    def test_no_generalist_leaves_fields_none(self):
        assessor = PostGameAssessor(difficulty=1, depth=3)
        result = assessor.assess(build_game_record(n_plies=4))
        for ann in result.moves:
            assert ann.generalist_policy_prob is None
            assert ann.generalist_top_move is None

    def test_generalist_score_moves_returns_none_graceful(self):
        gen = MagicMock()
        gen.score_moves.return_value = None
        assessor = PostGameAssessor(difficulty=1, depth=3, generalist=gen)
        result = assessor.assess(build_game_record(n_plies=2))
        for ann in result.moves:
            assert ann.generalist_policy_prob is None

    def test_all_stage4_signals_together(self):
        """Trajectory + policy + generalist can all run simultaneously."""
        record = build_game_record(n_plies=4)
        first_notation = record["moves"][0]["notation"]
        assessor = PostGameAssessor(
            difficulty=1, depth=3,
            trajectory_db=_make_mock_trajectory({first_notation: 0.2, "x": 0.3}),
            policy_advisor=_make_mock_policy(),
            generalist=_make_mock_generalist(),
        )
        result = assessor.assess(record)
        assert all(a.policy_prob is not None for a in result.moves)
        assert all(a.generalist_policy_prob is not None for a in result.moves)
        # trajectory: first ply has coverage; rest may or may not
        assert result.moves[0].traj_delta_best == 0.3
