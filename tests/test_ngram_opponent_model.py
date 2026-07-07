"""tests/test_ngram_opponent_model.py — SE-13: N-gram opponent move predictor."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai.ngram_opponent_model import NGramOpponentModel


def _make_game(moves: list[tuple[str, str]]) -> dict:
    """Build a minimal game record from a list of (color, notation) tuples."""
    return {
        "moves": [
            {"color": c, "notation": n}
            for c, n in moves
        ]
    }


def _alternating_game(w_moves: list[str], b_moves: list[str]) -> dict:
    """Interleave W and B moves into an alternating game record."""
    moves = []
    for i in range(max(len(w_moves), len(b_moves))):
        if i < len(w_moves):
            moves.append(("W", w_moves[i]))
        if i < len(b_moves):
            moves.append(("B", b_moves[i]))
    return _make_game(moves)


class TestNGramUpdate(unittest.TestCase):

    def test_bigram_built_from_game(self):
        """After update, bigram counts reflect the same-color move sequence."""
        model = NGramOpponentModel()
        game = _alternating_game(["d6", "f4", "b6"], ["d2", "b4", "f6"])
        model.update(game)
        # W: d6 → f4 → b6; bigrams: (W,d6)→f4, (W,f4)→b6
        bg_key = ("W", "d6")
        self.assertIn(bg_key, model._bigrams)
        self.assertEqual(model._bigrams[bg_key].get("f4", 0), 1)

    def test_trigram_built_from_game(self):
        """After update, trigrams reflect 3-move same-color sequences."""
        model = NGramOpponentModel()
        game = _alternating_game(["d6", "f4", "b6"], ["d2", "b4", "f6"])
        model.update(game)
        tg_key = ("W", "d6", "f4")
        self.assertIn(tg_key, model._trigrams)
        self.assertEqual(model._trigrams[tg_key].get("b6", 0), 1)

    def test_game_count_increments(self):
        model = NGramOpponentModel()
        self.assertEqual(model.game_count, 0)
        model.update(_alternating_game(["a1"], []))
        self.assertEqual(model.game_count, 1)
        model.update(_alternating_game(["b2"], []))
        self.assertEqual(model.game_count, 2)

    def test_update_ignores_empty_notations(self):
        """Moves with missing notation are silently skipped."""
        model = NGramOpponentModel()
        game = _make_game([("W", "d6"), ("B", ""), ("W", "f4")])
        model.update(game)
        # No crash, W only has d6 and f4 (B entry with "" is skipped)
        bg_key = ("W", "d6")
        self.assertIn(bg_key, model._bigrams)

    def test_multiple_games_accumulate_counts(self):
        model = NGramOpponentModel()
        game = _alternating_game(["d6", "f4"], [])
        model.update(game)
        model.update(game)
        self.assertEqual(model._bigrams[("W", "d6")]["f4"], 2)


class TestNGramPredict(unittest.TestCase):

    def setUp(self):
        self.model = NGramOpponentModel()
        # Train on a few games so we have enough data
        for _ in range(3):
            self.model.update(_alternating_game(
                ["d6", "f4", "b6", "f2", "d3"],
                ["d2", "b4", "f6", "b2", "d5"],
            ))

    def test_predict_returns_dict(self):
        notations = ["d6", "d2", "f4", "b4"]
        result = self.model.predict("W", notations)
        self.assertIsInstance(result, dict)

    def test_predict_probabilities_sum_to_one(self):
        notations = ["d6", "d2", "f4", "b4"]
        result = self.model.predict("W", notations)
        if result:
            self.assertAlmostEqual(sum(result.values()), 1.0, places=5)

    def test_predict_returns_empty_for_unknown_context(self):
        """Unknown n-gram context → empty dict."""
        result = self.model.predict("W", ["zzz", "xxx"])
        self.assertEqual(result, {})

    def test_predict_uses_trigram_when_available(self):
        """With enough data, trigram context takes priority over bigram."""
        # After d6,f4 (W moves), next W move should be b6
        notations = ["d6", "d2", "f4", "b4"]  # W: d6, f4; B: d2, b4
        result = self.model.predict("W", notations)
        # Trigram key (W, d6, f4) → b6 with count 3
        self.assertIn("b6", result)
        self.assertAlmostEqual(result["b6"], 1.0, places=5)

    def test_predict_falls_back_to_bigram(self):
        """When trigram context is absent, bigram is used."""
        # W moves: only one known move (f4), so no valid trigram context
        notations = ["f4"]  # W: f4 only (no prev W move → no trigram)
        result = self.model.predict("W", notations)
        # Bigram key (W, f4) → b6
        self.assertIn("b6", result)

    def test_predict_color_b(self):
        """predict works for Black too."""
        notations = ["d6", "d2", "f4", "b4"]  # B: d2, b4
        result = self.model.predict("B", notations)
        self.assertIsInstance(result, dict)
        if result:
            self.assertIn("f6", result)

    def test_predict_empty_history(self):
        result = self.model.predict("W", [])
        self.assertEqual(result, {})

    def test_predict_min_count_filters(self):
        """With min_count=10, sparse contexts return empty dict."""
        result = self.model.predict("W", ["d6", "d2", "f4"], min_count=10)
        # Only 3 games → count=3 < 10
        self.assertEqual(result, {})


class TestNGramSaveLoad(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        """Saving and loading preserves bigram/trigram counts and game_count."""
        model = NGramOpponentModel()
        game = _alternating_game(["d6", "f4", "b6"], ["d2", "b4", "f6"])
        for _ in range(5):
            model.update(game)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ngram.json"
            model.save(path)
            self.assertTrue(path.exists())

            loaded = NGramOpponentModel()
            loaded.load(path)
            self.assertEqual(loaded.game_count, 5)
            self.assertEqual(loaded._bigrams[("W", "d6")]["f4"], 5)
            self.assertEqual(loaded._trigrams[("W", "d6", "f4")]["b6"], 5)

    def test_load_nonexistent_file_is_no_op(self):
        """Loading a nonexistent file doesn't crash."""
        model = NGramOpponentModel()
        model.load("/tmp/does_not_exist_ngram.json")
        self.assertEqual(model.game_count, 0)


class TestNGramLoadFromGames(unittest.TestCase):

    def test_load_from_games_dir(self):
        """load_from_games processes *.jsonl files in a directory."""
        game = _alternating_game(["d6", "f4", "b6"], ["d2", "b4", "f6"])
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "game_test.jsonl"
            p.write_text(json.dumps(game) + "\n", encoding="utf-8")

            model = NGramOpponentModel()
            model.load_from_games(tmpdir)
            self.assertEqual(model.game_count, 1)

    def test_load_from_nonexistent_dir_no_crash(self):
        """Nonexistent directory is handled gracefully."""
        model = NGramOpponentModel()
        model.load_from_games("/tmp/does_not_exist_games_dir")
        self.assertEqual(model.game_count, 0)


class TestNGramPonderIntegration(unittest.TestCase):
    """SE-13: ngram_model param in PonderManager.start()."""

    def test_ponder_uses_ngram_model(self):
        """PonderManager.start() calls ngram_model.predict() for the board's color."""
        from unittest.mock import MagicMock
        from ai.ponder import PonderManager
        from game.board import BoardState
        import time

        board = BoardState.new_game()
        # Manually reach move phase
        w = ["a1", "a4", "a7", "b2", "b4", "b6", "c3", "c5", "d1"]
        b = ["d3", "d5", "d7", "e4", "e5", "f2", "f4", "f6", "g1"]
        for i in range(9):
            board = board.apply_move({"from": None, "to": w[i], "capture": None})
            board = board.apply_move({"from": None, "to": b[i], "capture": None})

        ngram_model = MagicMock()
        ngram_model.predict.return_value = {}

        game_ai = MagicMock()
        game_ai.color = "B"
        game_ai.difficulty = 1
        game_ai._weights = MagicMock()
        game_ai._value_net = None
        game_ai._endgame_solved_db = None
        game_ai._neural_evaluator = None
        game_ai.choose_move.return_value = {"to": "d6"}

        pm = PonderManager()
        pm.start(board=board, game_ai=game_ai, game_notations=[], ngram_model=ngram_model)
        time.sleep(0.1)
        pm.stop()

        ngram_model.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
