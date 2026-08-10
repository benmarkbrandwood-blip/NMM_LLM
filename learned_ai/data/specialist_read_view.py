"""Explicit training-time read projections over a writable SpecialistDB."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping
from typing import Any

from learned_ai.data.specialist_db import SpecialistWdlEvidence


SPECIALIST_READ_MODES = ("full", "theoretical-only")

_MALOM_PRIORS = {
    "W": (0.90, 0.05, 0.05),
    "D": (0.05, 0.90, 0.05),
    "L": (0.05, 0.05, 0.90),
}

_STAT_KEYS = (
    "queries",
    "rows_present",
    "theoretical_available",
    "empirical_available",
    "projections_returned",
    "empirical_suppressed",
)


def project_training_wdl(
    evidence: SpecialistWdlEvidence | None,
    mode: str,
) -> tuple[float, float, float] | None:
    """Apply one explicit production training read policy."""
    if mode not in SPECIALIST_READ_MODES:
        raise ValueError(f"unsupported SpecialistDB read mode: {mode}")
    if evidence is None:
        return None
    if mode == "full" and evidence.empirical_distribution is not None:
        return tuple(float(value) for value in evidence.empirical_distribution)
    if evidence.theoretical_wdl is None:
        return None
    return _MALOM_PRIORS[evidence.theoretical_wdl.value]


class SpecialistTrainingReadView:
    """Counted query projection that delegates persistence to the same database."""

    def __init__(self, database: Any, mode: str) -> None:
        if mode not in SPECIALIST_READ_MODES:
            raise ValueError(f"unsupported SpecialistDB read mode: {mode}")
        self._database = database
        self.mode = mode
        self._counts_by_thread: dict[int, Counter[str]] = {}
        self._lock = threading.RLock()

    def _thread_counts(self) -> Counter[str]:
        thread_id = threading.get_ident()
        counts = self._counts_by_thread.get(thread_id)
        if counts is None:
            counts = Counter()
            self._counts_by_thread[thread_id] = counts
        return counts

    def query_wdl(
        self,
        board: Any,
        min_samples: int = 5,
    ) -> tuple[float, float, float] | None:
        with self._lock:
            counts = self._thread_counts()
            evidence = self._database.query_wdl_evidence(board, min_samples)
            projection = project_training_wdl(evidence, self.mode)
            counts["queries"] += 1
            if evidence is not None:
                counts["rows_present"] += 1
                if evidence.theoretical_wdl is not None:
                    counts["theoretical_available"] += 1
                if evidence.empirical_distribution is not None:
                    counts["empirical_available"] += 1
                    if self.mode == "theoretical-only":
                        counts["empirical_suppressed"] += 1
            if projection is not None:
                counts["projections_returned"] += 1
            return projection

    def record_game(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._database.record_game(*args, **kwargs)

    def label_position_malom(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._database.label_position_malom(*args, **kwargs)

    def snapshot_stats(self) -> dict[str, int | str]:
        with self._lock:
            counts = self._thread_counts()
            return {
                "mode": self.mode,
                **{key: int(counts[key]) for key in _STAT_KEYS},
            }


def specialist_read_stats_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, int | str]:
    """Return a fail-closed delta between two read-view counter snapshots."""
    if before is None and after is None:
        return {}
    if before is None or after is None or before.get("mode") != after.get("mode"):
        raise ValueError("SpecialistDB read snapshots are incompatible")
    result: dict[str, int | str] = {"mode": str(after["mode"])}
    for key in _STAT_KEYS:
        left = before.get(key)
        right = after.get(key)
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or right < left
        ):
            raise ValueError("SpecialistDB read counters are invalid")
        result[key] = right - left
    return result
