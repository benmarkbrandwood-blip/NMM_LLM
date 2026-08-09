"""Fail-closed exact-WDL action labels for policy-only supervision.

The Generalist policy receives one row per complete legal action.  This module
joins those rows to Malom results by the full ``{from, to, capture}`` identity
and marks every action that preserves the root WDL value.  It deliberately does
not rank tied preserving actions or add any Malom feature to inference inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class MalomPolicyLabelError(RuntimeError):
    """Exact per-action supervision could not be established safely."""


@dataclass(frozen=True)
class MalomPolicyLabels:
    """Exact WDL deltas aligned with the policy encoder's legal-move order."""

    root_wdl: str
    qualities: np.ndarray
    preserving_mask: np.ndarray
    preserving_count: int
    downgrading_count: int


_ROOT_WDL = {"W": "win", "D": "draw", "L": "loss"}
_WDL_RANK = {"win": 1.0, "draw": 0.0, "loss": -1.0}


def _action_key(move: Any, *, source: str) -> tuple[str | None, str, str | None]:
    if not isinstance(move, Mapping):
        raise MalomPolicyLabelError(f"{source} action is not an object")
    if any(field not in move for field in ("from", "to", "capture")):
        raise MalomPolicyLabelError(
            f"{source} action lacks from/to/capture identity"
        )
    from_pos = move["from"]
    to_pos = move["to"]
    capture = move["capture"]
    if from_pos is not None and not isinstance(from_pos, str):
        raise MalomPolicyLabelError(f"{source} action has invalid from value")
    if not isinstance(to_pos, str) or not to_pos:
        raise MalomPolicyLabelError(f"{source} action has invalid to value")
    if capture is not None and not isinstance(capture, str):
        raise MalomPolicyLabelError(f"{source} action has invalid capture value")
    return from_pos, to_pos, capture


def label_malom_preserving_actions(
    malom_db: Any,
    *,
    board: Any,
    player: str,
    legal_moves: Sequence[Mapping[str, Any]],
) -> MalomPolicyLabels:
    """Return exact WDL-preserving labels in ``legal_moves`` order.

    The teacher must cover the complete legal action set with one known WDL per
    action.  Missing, duplicate, extra, unknown, or minimax-inconsistent rows
    stop the caller instead of silently weakening supervision.
    """
    if malom_db is None or not callable(getattr(malom_db, "is_available", None)):
        raise MalomPolicyLabelError("Malom teacher is not available")
    try:
        available = bool(malom_db.is_available())
    except Exception as exc:
        raise MalomPolicyLabelError("Malom availability probe failed") from exc
    if not available:
        raise MalomPolicyLabelError("Malom teacher is not available")

    legal_keys: list[tuple[str | None, str, str | None]] = []
    legal_key_set: set[tuple[str | None, str, str | None]] = set()
    for move in legal_moves:
        key = _action_key(move, source="policy")
        if key in legal_key_set:
            raise MalomPolicyLabelError("duplicate policy action")
        legal_keys.append(key)
        legal_key_set.add(key)
    if not legal_keys:
        raise MalomPolicyLabelError("policy action set is empty")

    try:
        raw_root = malom_db.query_state(board)
        rows = malom_db.query_all_moves(board, player)
    except Exception as exc:
        raise MalomPolicyLabelError("Malom action query failed") from exc
    root_wdl = _ROOT_WDL.get(raw_root)
    if root_wdl is None:
        raise MalomPolicyLabelError(f"unknown Malom root WDL: {raw_root!r}")
    if not isinstance(rows, list):
        raise MalomPolicyLabelError("Malom action set is not a list")

    by_key: dict[tuple[str | None, str, str | None], str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MalomPolicyLabelError("Malom action row is not an object")
        key = _action_key(row.get("move"), source="Malom")
        if key in by_key:
            raise MalomPolicyLabelError("duplicate Malom action")
        outcome = row.get("wdl")
        if outcome not in _WDL_RANK:
            raise MalomPolicyLabelError(
                f"unknown WDL for Malom action {key!r}: {outcome!r}"
            )
        by_key[key] = outcome

    if set(by_key) != legal_key_set:
        missing = len(legal_key_set - set(by_key))
        extra = len(set(by_key) - legal_key_set)
        raise MalomPolicyLabelError(
            "Malom action set does not match policy action set "
            f"(missing={missing}, extra={extra})"
        )

    root_rank = _WDL_RANK[root_wdl]
    qualities = np.asarray(
        [_WDL_RANK[by_key[key]] - root_rank for key in legal_keys],
        dtype=np.float32,
    )
    if bool(np.any(qualities > 0.0)):
        raise MalomPolicyLabelError("Malom action has a positive WDL delta")
    if not bool(np.all(np.isin(qualities, (0.0, -1.0, -2.0)))):
        raise MalomPolicyLabelError("Malom action has an invalid WDL delta")
    preserving = qualities == 0.0
    preserving_count = int(preserving.sum())
    if preserving_count == 0:
        raise MalomPolicyLabelError("Malom action set has no preserving action")

    return MalomPolicyLabels(
        root_wdl=root_wdl,
        qualities=qualities,
        preserving_mask=preserving,
        preserving_count=preserving_count,
        downgrading_count=len(legal_keys) - preserving_count,
    )
