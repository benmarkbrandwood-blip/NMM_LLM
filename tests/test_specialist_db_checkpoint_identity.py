"""Tests for cryptographic SpecialistDB checkpoint identities."""

from __future__ import annotations

from pathlib import Path

from game.board import BoardState
from learned_ai.data.specialist_db import SpecialistDB


def test_checkpoint_identity_is_stable_until_database_changes(tmp_path: Path) -> None:
    path = tmp_path / "specialist.sqlite"
    database = SpecialistDB(path)

    first = database.checkpoint_identity()
    repeated = database.checkpoint_identity()
    database.record_game(
        [BoardState.new_game()],
        "W",
        ["a1"],
        "gen",
        learner_color="W",
    )
    changed = database.checkpoint_identity()

    assert first == repeated
    assert first["label_version"] == "sector-corrected-v1"
    assert len(first["sha256"]) == 64
    assert changed["sha256"] != first["sha256"]
