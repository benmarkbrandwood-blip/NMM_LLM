"""Exact semantic parity checks for continuous and resumed Generalist runs."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learned_ai.training.checkpoint_envelope import load_checkpoint


class ResumeParityError(AssertionError):
    """Raised when a resumed run differs from its continuous reference."""


@dataclass(frozen=True)
class ResumeParityReport:
    """Machine-readable summary of one successful parity audit."""

    checkpoint_fields: tuple[str, ...]
    log_records: int
    database_tables: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "checkpoint_fields": list(self.checkpoint_fields),
            "log_records": self.log_records,
            "database_tables": list(self.database_tables),
        }


def _assert_equal(left: Any, right: Any, path: str) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ResumeParityError(f"{path}: tensor type differs")
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ResumeParityError(f"{path}: tensor metadata differs")
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise ResumeParityError(f"{path}: tensor values differ")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ResumeParityError(f"{path}: array type differs")
        if not np.array_equal(left, right):
            raise ResumeParityError(f"{path}: array values differ")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ResumeParityError(f"{path}: mapping type differs")
        if set(left) != set(right):
            raise ResumeParityError(f"{path}: mapping keys differ")
        for key in sorted(left, key=str):
            _assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            raise ResumeParityError(f"{path}: sequence type differs")
        if len(left) != len(right):
            raise ResumeParityError(f"{path}: sequence length differs")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_equal(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise ResumeParityError(f"{path}: {left!r} != {right!r}")


def _normalized_trainer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    recovery = dict(result["recovery_state"])
    recovery.pop("source_checkpoint", None)
    recovery.pop("checkpoint_sequence", None)
    result["recovery_state"] = recovery
    return result


def _normalized_data_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    mutable_assets = dict(result["mutable_assets"])
    mutable_assets.pop("specialist_db", None)
    result["mutable_assets"] = mutable_assets
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        record.pop("source_checkpoint", None)
    return records


def _database_snapshot(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    queries = {
        "positions": (
            "SELECT pos_hash, wins, draws, losses, malom_label "
            "FROM positions ORDER BY pos_hash"
        ),
        "preferred_plays": (
            "SELECT tag, pos_sequence, win_rate, times_played, promoted "
            "FROM preferred_plays ORDER BY tag, pos_sequence"
        ),
        "winning_lines": (
            "SELECT move_seq, phase, result, wins, times_played, win_rate "
            "FROM winning_lines ORDER BY move_seq, phase, result"
        ),
        "meta": (
            "SELECT key, value FROM meta "
            "WHERE key <> 'training_lineage_root_run_id' ORDER BY key"
        ),
    }
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {name: connection.execute(query).fetchall() for name, query in queries.items()}


def verify_resume_parity(
    *,
    continuous_checkpoint: str | Path,
    resumed_checkpoint: str | Path,
    continuous_log: str | Path,
    resumed_logs: Sequence[str | Path],
    continuous_database: str | Path,
    resumed_database: str | Path,
) -> ResumeParityReport:
    """Require exact future-state and evidence parity after a segmented resume."""
    continuous = load_checkpoint(continuous_checkpoint, map_location="cpu")
    resumed = load_checkpoint(resumed_checkpoint, map_location="cpu")
    comparisons = {
        "model_state": (continuous.payload.model_state, resumed.payload.model_state),
        "optimizer_state": (
            continuous.payload.optimizer_state,
            resumed.payload.optimizer_state,
        ),
        "scheduler_state": (
            continuous.payload.scheduler_state,
            resumed.payload.scheduler_state,
        ),
        "scaler_state": (continuous.payload.scaler_state, resumed.payload.scaler_state),
        "rng_state": (continuous.payload.rng_state, resumed.payload.rng_state),
        "trainer_state": (
            _normalized_trainer_state(continuous.payload.trainer_state),
            _normalized_trainer_state(resumed.payload.trainer_state),
        ),
        "data_state": (
            _normalized_data_state(continuous.payload.data_state),
            _normalized_data_state(resumed.payload.data_state),
        ),
    }
    for name, (left, right) in comparisons.items():
        _assert_equal(left, right, name)

    continuous_records = _read_jsonl(Path(continuous_log))
    resumed_records: list[dict[str, Any]] = []
    for log in resumed_logs:
        resumed_records.extend(_read_jsonl(Path(log)))
    _assert_equal(continuous_records, resumed_records, "training_log")

    continuous_db = _database_snapshot(Path(continuous_database))
    resumed_db = _database_snapshot(Path(resumed_database))
    _assert_equal(continuous_db, resumed_db, "specialist_database")
    return ResumeParityReport(
        checkpoint_fields=tuple(comparisons),
        log_records=len(continuous_records),
        database_tables=tuple(continuous_db),
    )
