"""Tests for runtime quarantine and read-only SpecialistDB boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from game.board import BoardState
from learned_ai.agents.specialist_router import SpecialistRouter
from learned_ai.data.runtime_quarantine import RuntimeGameQuarantine
from learned_ai.data.specialist_db import SpecialistDB


def test_runtime_quarantine_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    path = tmp_path / "runtime-games.jsonl"
    quarantine = RuntimeGameQuarantine(path)

    first_id = quarantine.append_game({"winner": "W", "moves": []}, source="test")
    second_id = quarantine.append_game({"winner": None, "moves": []}, source="test")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert first_id != second_id
    assert [record["sequence"] for record in records] == [0, 1]
    assert records[1]["previous_record_sha256"] == records[0]["record_sha256"]
    assert records[0]["trust_level"] == "quarantined_unreviewed"
    assert RuntimeGameQuarantine(path)._sequence == 2


def test_tampered_quarantine_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "runtime-games.jsonl"
    RuntimeGameQuarantine(path).append_game({"winner": "W"}, source="test")
    path.write_text(path.read_text().replace('"winner":"W"', '"winner":"B"'))

    with pytest.raises(RuntimeError, match="hash is invalid"):
        RuntimeGameQuarantine(path)


def test_read_only_specialist_db_rejects_runtime_writes(tmp_path: Path) -> None:
    path = tmp_path / "specialist.sqlite"
    SpecialistDB(path).close()
    database = SpecialistDB(path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        database.record_game([BoardState.new_game()], "W", [], "gen")

    database.close()


def test_router_quarantines_without_mutating_specialist_db(tmp_path: Path) -> None:
    database_path = tmp_path / "specialist.sqlite"
    writable = SpecialistDB(database_path)
    before = writable.stats()
    writable.close()
    database = SpecialistDB(database_path, read_only=True)
    quarantine = RuntimeGameQuarantine(tmp_path / "quarantine.jsonl")
    router = SpecialistRouter(
        None,
        None,
        None,
        {},
        specialist_db=database,
        runtime_quarantine=quarantine,
    )

    router.record_game_result({"human_color": "W", "winner": "B", "moves": []})

    assert database.stats() == before
    assert quarantine.path.read_text(encoding="utf-8").count("\n") == 1
    database.close()
