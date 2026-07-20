from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from learned_ai.validation.resume_parity import (
    ResumeParityError,
    _assert_equal,
    _read_jsonl,
)


def test_assert_equal_reports_nested_tensor_difference() -> None:
    import torch

    with pytest.raises(ResumeParityError, match=r"root\.weights"):
        _assert_equal({"weights": torch.tensor([1.0])}, {"weights": torch.tensor([2.0])}, "root")


def test_read_jsonl_ignores_only_checkpoint_source(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"game_id": "g1", "source_checkpoint": "a"}) + "\n",
        encoding="utf-8",
    )
    assert _read_jsonl(path) == [{"game_id": "g1"}]


def test_database_schema_fixture_documents_semantic_columns(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "CREATE TABLE positions (pos_hash TEXT PRIMARY KEY, wins INTEGER, "
            "draws INTEGER, losses INTEGER, malom_label TEXT, last_seen TEXT)"
        )
    assert path.exists()
