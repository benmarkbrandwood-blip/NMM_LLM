"""
ai/ngram_opponent_model.py — SE-13: N-gram opponent move predictor.

Builds per-color bigram and trigram frequency tables from game JSONL records.
predict() returns a probability dict for the next move by a given color, keyed
by move notation, used to boost prediction quality in PonderManager.start().
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class NGramOpponentModel:
    """Bigram/trigram move-frequency model built from game records.

    Move sequences are per-color: White always occupies even-indexed turns
    (0, 2, 4, …) and Black odd-indexed turns (1, 3, 5, …) in each game.

    n-gram keys are tuples of same-color notation strings, NOT full game
    notation lists — so bigram ("d6", "f4") means "White played d6 then f4"
    regardless of Black's intervening moves.
    """

    def __init__(self) -> None:
        # {(color, prev_notation): {next_notation: count}}
        self._bigrams: dict[tuple, dict[str, int]] = {}
        # {(color, prev_prev_notation, prev_notation): {next_notation: count}}
        self._trigrams: dict[tuple, dict[str, int]] = {}
        self._game_count: int = 0

    # ── Training ────────────────────────────────────────────────────────────

    def update(self, game_record: dict) -> None:
        """Incorporate one game record into the model."""
        moves = game_record.get("moves", [])
        # Partition moves by color preserving order
        by_color: dict[str, list[str]] = {"W": [], "B": []}
        for m in moves:
            c = m.get("color", "")
            n = m.get("notation", "")
            if c in by_color and n:
                by_color[c].append(n)

        for c, seq in by_color.items():
            for i in range(1, len(seq)):
                bg_key = (c, seq[i - 1])
                d = self._bigrams.setdefault(bg_key, {})
                d[seq[i]] = d.get(seq[i], 0) + 1
            for i in range(2, len(seq)):
                tg_key = (c, seq[i - 2], seq[i - 1])
                d = self._trigrams.setdefault(tg_key, {})
                d[seq[i]] = d.get(seq[i], 0) + 1

        self._game_count += 1

    def load_from_games(self, games_dir: Path | str) -> None:
        """Load every *.jsonl file in `games_dir` (recursively) into the model."""
        games_dir = Path(games_dir)
        if not games_dir.exists():
            log.warning("NGramOpponentModel: games dir not found: %s", games_dir)
            return
        loaded = 0
        for path in sorted(games_dir.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.update(json.loads(line))
                    loaded += 1
                except Exception as exc:
                    log.debug("NGramOpponentModel: skipping line in %s — %s", path.name, exc)
        log.info("NGramOpponentModel: loaded %d games from %s", loaded, games_dir)

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(
        self,
        color: str,
        game_notations: list[str],
        min_count: int = 2,
    ) -> dict[str, float]:
        """Return {notation: probability} for the next move by `color`.

        `game_notations` is the full alternating notation list (even indices = W,
        odd = B).  Trigram evidence is used when available; falls back to bigram;
        returns {} when there is not enough data.
        """
        # Extract this color's recent moves from the alternating notation list
        offset = 0 if color == "W" else 1
        color_moves = game_notations[offset::2]
        if not color_moves:
            return {}

        # Try trigram first
        if len(color_moves) >= 2:
            tg_key = (color, color_moves[-2], color_moves[-1])
            tg_counts = self._trigrams.get(tg_key, {})
            total = sum(tg_counts.values())
            if total >= min_count:
                return {n: c / total for n, c in tg_counts.items()}

        # Fall back to bigram
        bg_key = (color, color_moves[-1])
        bg_counts = self._bigrams.get(bg_key, {})
        total = sum(bg_counts.values())
        if total >= min_count:
            return {n: c / total for n, c in bg_counts.items()}

        return {}

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Persist the model to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "game_count": self._game_count,
            "bigrams": {
                json.dumps(list(k)): v for k, v in self._bigrams.items()
            },
            "trigrams": {
                json.dumps(list(k)): v for k, v in self._trigrams.items()
            },
        }
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        log.info("NGramOpponentModel: saved %d games to %s", self._game_count, path)

    def load(self, path: Path | str) -> None:
        """Load a previously saved model from JSON."""
        path = Path(path)
        if not path.exists():
            log.warning("NGramOpponentModel: model file not found: %s", path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._game_count = data.get("game_count", 0)
        self._bigrams = {
            tuple(json.loads(k)): v
            for k, v in data.get("bigrams", {}).items()
        }
        self._trigrams = {
            tuple(json.loads(k)): v
            for k, v in data.get("trigrams", {}).items()
        }
        log.info("NGramOpponentModel: loaded %d games from %s", self._game_count, path)

    @property
    def game_count(self) -> int:
        return self._game_count
