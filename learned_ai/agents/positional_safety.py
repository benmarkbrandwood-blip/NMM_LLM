"""Position-only Malom safety filtering for product move selection.

This module implements ``A_pos``: retain every legal move whose Malom W/D/L
tier equals the parent position's tier. It does not carry repetition or
no-progress history and therefore must not be described as full-rule safety.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, terminal_wdl
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION


_WDL_RANK = {"L": 0, "D": 1, "W": 2}


class PositionalSafetyError(RuntimeError):
    """A required trust, inventory, or Malom query contract failed."""


def _move_key(move: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return move.get("from"), move.get("to"), move.get("capture")


@dataclass(frozen=True)
class PositionalSafetyDecision:
    """One successful final-selection intervention record."""

    original_move: dict[str, Any]
    selected_move: dict[str, Any]
    original_score: float
    selected_score: float
    parent_tier: str
    legal_move_count: int
    safe_move_count: int
    query_count: int
    elapsed_ms: float
    intervened: bool
    positional_only: bool = True
    history_aware: bool = False


class PositionalSafetyFilter:
    """Filter scored legal moves to the trusted positional ``A_pos`` set."""

    def __init__(
        self,
        oracle: Any,
        *,
        label_version: str,
        manifest_sha256: str,
        content_sha256: str | None = None,
    ) -> None:
        if label_version != CURRENT_MALOM_LABEL_VERSION:
            raise PositionalSafetyError(
                "Malom label version must be sector-corrected-v1"
            )
        if (
            len(manifest_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in manifest_sha256)
        ):
            raise PositionalSafetyError("Malom manifest identity is invalid")
        required = ("query_value", "move_value", "terminal_move_value")
        missing = [
            name for name in required if not callable(getattr(oracle, name, None))
        ]
        if missing:
            raise PositionalSafetyError(
                f"Malom oracle surface is incomplete: {', '.join(missing)}"
            )
        self.oracle = oracle
        self.label_version = label_version
        self.manifest_sha256 = manifest_sha256
        self.content_sha256 = content_sha256

    def _query_value(self, board: BoardState, *, context: str):
        try:
            value = self.oracle.query_value(board)
        except Exception as exc:
            raise PositionalSafetyError(
                f"required {context} Malom query raised {type(exc).__name__}"
            ) from exc
        if value is None or getattr(value, "outcome", None) not in _WDL_RANK:
            raise PositionalSafetyError(
                f"required {context} Malom value is unavailable"
            )
        return value

    def filter_scores(
        self,
        board: BoardState,
        candidates: Sequence[Mapping[str, Any]],
        scores: Sequence[float],
    ) -> tuple[list[float], PositionalSafetyDecision]:
        """Return model scores renormalized over ``A_pos`` or fail closed.

        The model has already scored every legal move. This final operation
        preserves its ordering inside ``A_pos`` while assigning zero mass to
        positional W/D/L downgrades.
        """
        started = time.perf_counter()
        legal = [dict(move) for move in get_all_legal_moves(board)]
        if not legal:
            raise PositionalSafetyError("position has no legal moves to filter")
        if len(candidates) != len(legal) or len(scores) != len(candidates):
            raise PositionalSafetyError(
                "positional safety requires scores for every legal move"
            )

        candidate_keys = [_move_key(move) for move in candidates]
        legal_keys = [_move_key(move) for move in legal]
        if (
            len(set(candidate_keys)) != len(candidate_keys)
            or set(candidate_keys) != set(legal_keys)
        ):
            raise PositionalSafetyError(
                "candidate inventory differs from the complete legal move set"
            )

        numeric_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) or score < 0.0 for score in numeric_scores):
            raise PositionalSafetyError("model scores must be finite and non-negative")

        parent = self._query_value(board, context="parent")
        move_outcomes: dict[tuple[Any, Any, Any], str] = {}
        contexts: set[tuple[Any, Any, Any]] = set()
        query_count = 1
        for move in legal:
            after = board.apply_move(move)
            rules_outcome = terminal_wdl(after)
            try:
                if rules_outcome is not None:
                    value = self.oracle.terminal_move_value(parent, rules_outcome)
                else:
                    child = self._query_value(after, context="successor")
                    query_count += 1
                    value = self.oracle.move_value(parent, child)
            except PositionalSafetyError:
                raise
            except Exception as exc:
                raise PositionalSafetyError(
                    "required Malom move-value conversion failed"
                ) from exc
            outcome = getattr(value, "outcome", None)
            if outcome not in _WDL_RANK:
                raise PositionalSafetyError("candidate Malom value is unavailable")
            move_outcomes[_move_key(move)] = outcome
            contexts.add(
                (
                    getattr(value, "sector", None),
                    getattr(value, "sector_value", None),
                    getattr(value, "perspective", None),
                )
            )

        if len(contexts) != 1:
            raise PositionalSafetyError("candidate Malom values mix parent contexts")
        best_tier = max(move_outcomes.values(), key=_WDL_RANK.__getitem__)
        if best_tier != parent.outcome:
            raise PositionalSafetyError(
                "candidate inventory contradicts the parent Malom tier"
            )

        safe_indices = [
            index
            for index, key in enumerate(candidate_keys)
            if move_outcomes[key] == parent.outcome
        ]
        if not safe_indices:
            raise PositionalSafetyError("position has no W/D/L-preserving move")
        safe_total = sum(numeric_scores[index] for index in safe_indices)
        if not math.isfinite(safe_total) or safe_total <= 1e-12:
            raise PositionalSafetyError("model assigned no usable mass inside A_pos")

        filtered = [0.0] * len(numeric_scores)
        for index in safe_indices:
            filtered[index] = numeric_scores[index] / safe_total

        original_index = max(
            range(len(numeric_scores)),
            key=numeric_scores.__getitem__,
        )
        selected_index = max(safe_indices, key=numeric_scores.__getitem__)
        decision = PositionalSafetyDecision(
            original_move=dict(candidates[original_index]),
            selected_move=dict(candidates[selected_index]),
            original_score=numeric_scores[original_index],
            selected_score=numeric_scores[selected_index],
            parent_tier=parent.outcome,
            legal_move_count=len(legal),
            safe_move_count=len(safe_indices),
            query_count=query_count,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            intervened=original_index != selected_index,
        )
        return filtered, decision
