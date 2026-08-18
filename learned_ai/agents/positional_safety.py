"""Position-only Malom safety filtering for product move selection.

This module implements ``A_pos``: retain every legal move whose Malom W/D/L
tier equals the parent position's tier. It does not carry repetition or
no-progress history and therefore must not be described as full-rule safety.
"""

from __future__ import annotations

import math
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, terminal_wdl
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION


_WDL_RANK = {"L": 0, "D": 1, "W": 2}
_LOG = logging.getLogger("nmm.positional_safety")


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


@dataclass(frozen=True)
class PositionalSafetyInventory:
    """Trusted W/D/L tiers for one complete legal-move inventory."""

    legal_moves: tuple[dict[str, Any], ...]
    move_tiers: tuple[str, ...]
    safe_indices: tuple[int, ...]
    parent_tier: str
    query_count: int
    elapsed_ms: float


@dataclass(frozen=True)
class ProductSafetyOutcome:
    """Final product move and the JSON-safe decision that produced it."""

    move: dict[str, Any]
    decision: dict[str, Any]


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

    def inspect_moves(
        self,
        board: BoardState,
        candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> PositionalSafetyInventory:
        """Return the complete positional tier partition for legal moves."""
        started = time.perf_counter()
        legal = [dict(move) for move in get_all_legal_moves(board)]
        if not legal:
            raise PositionalSafetyError("position has no legal moves to filter")
        supplied = legal if candidates is None else [dict(move) for move in candidates]
        if len(supplied) != len(legal):
            raise PositionalSafetyError(
                "positional safety requires every legal move"
            )

        candidate_keys = [_move_key(move) for move in supplied]
        legal_keys = [_move_key(move) for move in legal]
        if (
            len(set(candidate_keys)) != len(candidate_keys)
            or set(candidate_keys) != set(legal_keys)
        ):
            raise PositionalSafetyError(
                "candidate inventory differs from the complete legal move set"
            )

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
        move_tiers = tuple(move_outcomes[key] for key in candidate_keys)
        safe_indices = tuple(
            index
            for index, tier in enumerate(move_tiers)
            if tier == parent.outcome
        )
        if not safe_indices:
            raise PositionalSafetyError("position has no W/D/L-preserving move")
        return PositionalSafetyInventory(
            legal_moves=tuple(supplied),
            move_tiers=move_tiers,
            safe_indices=safe_indices,
            parent_tier=parent.outcome,
            query_count=query_count,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

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
        inventory = self.inspect_moves(board, candidates)
        legal = list(inventory.legal_moves)
        if len(scores) != len(candidates):
            raise PositionalSafetyError(
                "positional safety requires scores for every legal move"
            )

        numeric_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) or score < 0.0 for score in numeric_scores):
            raise PositionalSafetyError("model scores must be finite and non-negative")

        safe_indices = list(inventory.safe_indices)
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
            parent_tier=inventory.parent_tier,
            legal_move_count=len(legal),
            safe_move_count=len(safe_indices),
            query_count=inventory.query_count,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            intervened=original_index != selected_index,
        )
        return filtered, decision


class ProductPositionalSafetyGate:
    """Single final-selection choke for every product-owned AI move source.

    The default difficulty ladder is constrained only at difficulties 9 and
    10. Explicit specialist/generalist routes are always constrained because
    selecting them is already an opt-in replacement for the normal ladder.
    Query failure is observable and returns the caller-provided classical
    fallback, preserving playability without claiming that filtering applied.
    """

    def __init__(self, *, high_difficulty_minimum: int = 9) -> None:
        self.high_difficulty_minimum = int(high_difficulty_minimum)
        self._filter: PositionalSafetyFilter | None = None
        self._disabled_reason = "Malom DB is unavailable at startup"
        self._state_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._counters = {
            "requests": 0,
            "applied_requests": 0,
            "interventions": 0,
            "low_difficulty_bypasses": 0,
            "unavailable_requests": 0,
            "runtime_failures": 0,
            "selection_failures": 0,
        }
        self._last_error = self._disabled_reason
        self._last_decision: dict[str, Any] | None = None

    def configure(
        self,
        oracle: Any,
        *,
        label_version: str,
        manifest_sha256: str,
        content_sha256: str | None = None,
    ) -> None:
        safety_filter = PositionalSafetyFilter(
            oracle,
            label_version=label_version,
            manifest_sha256=manifest_sha256,
            content_sha256=content_sha256,
        )
        with self._state_lock:
            self._filter = safety_filter
            self._disabled_reason = ""
            self._last_error = ""

    def disable(self, reason: str) -> None:
        detail = reason.strip() or "unspecified startup failure"
        with self._state_lock:
            self._filter = None
            self._disabled_reason = detail
            self._last_error = detail
        _LOG.error("Product A_pos final gate disabled: %s", detail)

    def is_enabled(self) -> bool:
        with self._state_lock:
            return self._filter is not None

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            safety_filter = self._filter
            return {
                "configured": safety_filter is not None,
                "enabled": safety_filter is not None,
                "mode": "A_pos",
                "scope": "difficulty-9-10-and-explicit-learned-routes",
                "high_difficulty_minimum": self.high_difficulty_minimum,
                "positional_only": True,
                "history_aware": False,
                "label_version": (
                    safety_filter.label_version if safety_filter is not None else None
                ),
                "manifest_sha256": (
                    safety_filter.manifest_sha256 if safety_filter is not None else None
                ),
                "content_sha256": (
                    safety_filter.content_sha256 if safety_filter is not None else None
                ),
                "disabled_reason": self._disabled_reason,
                **self._counters,
                "last_error": self._last_error,
                "last_decision": self._last_decision,
            }

    @staticmethod
    def _canonical_move(moves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def key(move: Mapping[str, Any]) -> tuple[str, str, str]:
            return tuple(str(move.get(field) or "") for field in ("from", "to", "capture"))

        return dict(min(moves, key=key))

    def _record(self, decision: dict[str, Any], *, error: str = "") -> None:
        with self._state_lock:
            self._last_decision = decision
            self._last_error = error

    def constrain(
        self,
        board: BoardState,
        original_move: Mapping[str, Any],
        *,
        source: str,
        difficulty: int,
        candidate_scores: Sequence[float] | None = None,
        safe_selector: Callable[[list[dict[str, Any]]], Mapping[str, Any]] | None = None,
        query_failure_move: Mapping[str, Any] | None = None,
    ) -> ProductSafetyOutcome:
        """Return the final move after applying the product ``A_pos`` gate."""
        started = time.perf_counter()
        legal = [dict(move) for move in get_all_legal_moves(board)]
        legal_keys = {_move_key(move) for move in legal}
        original = dict(original_move)
        fallback = dict(query_failure_move or original_move)
        if _move_key(original) not in legal_keys:
            raise PositionalSafetyError("original product move is not legal")
        if _move_key(fallback) not in legal_keys:
            raise PositionalSafetyError("query-failure fallback move is not legal")

        learned_source = source.startswith("specialist") or source.startswith(
            "generalist"
        )
        enforced = int(difficulty) >= self.high_difficulty_minimum or learned_source
        with self._state_lock:
            self._counters["requests"] += 1
            safety_filter = self._filter
            disabled_reason = self._disabled_reason
        base = {
            "source": source,
            "difficulty": int(difficulty),
            "original_move": original,
            "selected_move": fallback,
            "intervened": _move_key(original) != _move_key(fallback),
            "mode": "A_pos",
            "positional_only": True,
            "history_aware": False,
        }
        if not enforced:
            with self._state_lock:
                self._counters["low_difficulty_bypasses"] += 1
            decision = {
                **base,
                "status": "bypassed-low-difficulty",
                "selected_move": original,
                "intervened": False,
                "selection_rule": "original-unfiltered",
                "total_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
            self._record(decision)
            return ProductSafetyOutcome(original, decision)

        if safety_filter is None:
            with self._state_lock:
                self._counters["unavailable_requests"] += 1
            decision = {
                **base,
                "status": "unfiltered-malom-unavailable",
                "selection_rule": "classical-fallback-unfiltered",
                "failure": disabled_reason,
                "total_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
            self._record(decision, error=disabled_reason)
            _LOG.error(
                "Product A_pos unavailable source=%s difficulty=%s: %s",
                source,
                difficulty,
                disabled_reason,
            )
            return ProductSafetyOutcome(fallback, decision)

        try:
            with self._query_lock:
                inventory = safety_filter.inspect_moves(board, legal)
        except Exception as exc:
            detail = (
                str(exc)
                if isinstance(exc, PositionalSafetyError)
                else f"unexpected positional safety failure: {type(exc).__name__}"
            )
            with self._state_lock:
                self._counters["runtime_failures"] += 1
            decision = {
                **base,
                "status": "unfiltered-query-failure",
                "selection_rule": "classical-fallback-unfiltered",
                "failure": detail,
                "total_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
            self._record(decision, error=detail)
            _LOG.error(
                "Product A_pos query failed source=%s difficulty=%s: %s",
                source,
                difficulty,
                detail,
                exc_info=True,
            )
            return ProductSafetyOutcome(fallback, decision)

        safe_moves = [
            dict(inventory.legal_moves[index]) for index in inventory.safe_indices
        ]
        safe_keys = {_move_key(move) for move in safe_moves}
        selected = original
        selection_rule = "original-already-in-A_pos"
        selection_error = ""
        selection_started = time.perf_counter()
        if _move_key(original) not in safe_keys:
            if candidate_scores is not None:
                numeric = [float(value) for value in candidate_scores]
                if len(numeric) != len(legal) or any(
                    not math.isfinite(value) or value < 0.0 for value in numeric
                ):
                    selection_error = "invalid complete candidate score inventory"
                elif sum(numeric[index] for index in inventory.safe_indices) > 1e-12:
                    best_score = max(numeric[index] for index in inventory.safe_indices)
                    tied = [
                        legal[index]
                        for index in inventory.safe_indices
                        if numeric[index] == best_score
                    ]
                    selected = self._canonical_move(tied)
                    selection_rule = "model-argmax-inside-A_pos"
            if _move_key(selected) not in safe_keys and safe_selector is not None:
                try:
                    proposed = dict(safe_selector(safe_moves))
                    if _move_key(proposed) not in safe_keys:
                        raise PositionalSafetyError(
                            "restricted root research returned a move outside A_pos"
                        )
                    selected = proposed
                    selection_rule = "restricted-root-research"
                except Exception as exc:
                    selection_error = f"{type(exc).__name__}: {exc}"
            if _move_key(selected) not in safe_keys:
                selected = self._canonical_move(safe_moves)
                selection_rule = "canonical-safe-fallback"
                if selection_error:
                    with self._state_lock:
                        self._counters["selection_failures"] += 1
                    _LOG.error(
                        "A_pos root re-ranking failed; using canonical safe move: %s",
                        selection_error,
                    )

        intervened = _move_key(original) != _move_key(selected)
        with self._state_lock:
            self._counters["applied_requests"] += 1
            if intervened:
                self._counters["interventions"] += 1
        original_index = next(
            index
            for index, move in enumerate(inventory.legal_moves)
            if _move_key(move) == _move_key(original)
        )
        decision = {
            **base,
            "status": "applied",
            "selected_move": selected,
            "intervened": intervened,
            "selection_rule": selection_rule,
            "selection_error": selection_error or None,
            "parent_tier": inventory.parent_tier,
            "original_tier": inventory.move_tiers[original_index],
            "legal_move_count": len(inventory.legal_moves),
            "safe_move_count": len(inventory.safe_indices),
            "query_count": inventory.query_count,
            "elapsed_ms": inventory.elapsed_ms,
            "selection_elapsed_ms": (
                time.perf_counter() - selection_started
            ) * 1000.0,
            "total_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        self._record(decision, error=selection_error)
        if intervened:
            _LOG.warning(
                "Product A_pos rewrote source=%s difficulty=%s move=%s -> %s",
                source,
                difficulty,
                original,
                selected,
            )
        return ProductSafetyOutcome(selected, decision)
