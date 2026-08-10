"""Read-only SpecialistDB projections for fixed-policy mechanism audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from learned_ai.data.specialist_db import SpecialistWdlEvidence


PROJECTION_MODES = (
    "full",
    "empirical_disabled",
    "malom_disabled",
    "all_disabled",
)
MATERIAL_ARGMAX_CHANGES = 3
MATERIAL_MEAN_TOTAL_VARIATION = 0.05

_MALOM_PRIORS = {
    "W": (0.90, 0.05, 0.05),
    "D": (0.05, 0.90, 0.05),
    "L": (0.05, 0.05, 0.90),
}


def project_wdl(
    evidence: SpecialistWdlEvidence | None,
    mode: str,
) -> tuple[float, float, float] | None:
    """Project separated theoretical/empirical evidence for one audit mode."""
    if mode not in PROJECTION_MODES:
        raise ValueError(f"unsupported SpecialistDB projection mode: {mode}")
    if evidence is None or mode == "all_disabled":
        return None

    theoretical = evidence.theoretical_wdl
    empirical = evidence.empirical_distribution
    if mode == "full":
        if empirical is not None:
            return tuple(float(value) for value in empirical)
        if theoretical is not None:
            return _MALOM_PRIORS[theoretical.value]
        return None
    if mode == "empirical_disabled":
        if theoretical is None:
            return None
        return _MALOM_PRIORS[theoretical.value]
    if mode == "malom_disabled":
        if empirical is None:
            return None
        return tuple(float(value) for value in empirical)
    raise AssertionError("projection mode exhaustiveness drifted")


@dataclass
class SpecialistEvidenceProjection:
    """Encoder-compatible view over one immutable SpecialistDB."""

    database: Any
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in PROJECTION_MODES:
            raise ValueError(f"unsupported SpecialistDB projection mode: {self.mode}")

    def query_wdl(
        self,
        board: Any,
        min_samples: int = 5,
    ) -> tuple[float, float, float] | None:
        evidence = self.database.query_wdl_evidence(board, min_samples)
        return project_wdl(evidence, self.mode)


def evidence_record(
    evidence: SpecialistWdlEvidence | None,
) -> dict[str, Any]:
    """Return the stable, separated record used in per-action evidence."""
    if evidence is None:
        return {
            "present": False,
            "perspective": None,
            "theoretical_wdl": None,
            "empirical_counts": [0, 0, 0],
            "empirical_samples": 0,
            "empirical_distribution": None,
            "empirical_modal_wdl": [],
            "theoretical_empirical_disagreement": False,
            "projection_hits": {mode: False for mode in PROJECTION_MODES},
        }

    counts = tuple(int(value) for value in evidence.empirical_counts)
    samples = sum(counts)
    theoretical = (
        None
        if evidence.theoretical_wdl is None
        else str(evidence.theoretical_wdl.value)
    )
    modal: list[str] = []
    if samples:
        maximum = max(counts)
        modal = [
            label
            for label, count in zip(("W", "D", "L"), counts, strict=True)
            if count == maximum
        ]
    disagreement = bool(
        theoretical is not None
        and evidence.empirical_distribution is not None
        and theoretical not in modal
    )
    return {
        "present": True,
        "perspective": str(evidence.perspective),
        "theoretical_wdl": theoretical,
        "empirical_counts": list(counts),
        "empirical_samples": samples,
        "empirical_distribution": (
            None
            if evidence.empirical_distribution is None
            else [float(value) for value in evidence.empirical_distribution]
        ),
        "empirical_modal_wdl": modal,
        "theoretical_empirical_disagreement": disagreement,
        "projection_hits": {
            mode: project_wdl(evidence, mode) is not None for mode in PROJECTION_MODES
        },
    }


def total_variation(left: Iterable[float], right: Iterable[float]) -> float:
    """Return total-variation distance for two finite action distributions."""
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("probability vectors must have the same 1-D shape")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("probability vectors must be finite")
    return float(0.5 * np.abs(left_array - right_array).sum())


def _quality_class(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value > 0:
        raise ValueError("Malom move quality cannot be positive")
    return "preserving" if value == 0 else "downgrading"


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changes: list[int] = []
    crossings: list[int] = []
    distances_scheduled: list[float] = []
    distances_temp1: list[float] = []
    direction_counts: Counter[str] = Counter()
    for row in rows:
        full = row["modes"]["full"]
        without_empirical = row["modes"]["empirical_disabled"]
        if full["argmax_index"] != without_empirical["argmax_index"]:
            changes.append(int(row["index"]))
        full_class = _quality_class(full["argmax_malom_quality"])
        without_class = _quality_class(without_empirical["argmax_malom_quality"])
        direction_counts[f"{full_class}_to_{without_class}"] += 1
        if row["critical"] and {full_class, without_class} == {
            "preserving",
            "downgrading",
        }:
            crossings.append(int(row["index"]))
        distances_scheduled.append(
            total_variation(
                full["probabilities_scheduled"],
                without_empirical["probabilities_scheduled"],
            )
        )
        distances_temp1.append(
            total_variation(
                full["probabilities_temp1"],
                without_empirical["probabilities_temp1"],
            )
        )
    count = len(rows)
    return {
        "states": count,
        "argmax_changes": len(changes),
        "argmax_change_indices": changes,
        "critical_preservation_crossings": len(crossings),
        "critical_preservation_crossing_indices": crossings,
        "mean_total_variation_scheduled": (
            float(np.mean(distances_scheduled)) if distances_scheduled else 0.0
        ),
        "max_total_variation_scheduled": (max(distances_scheduled, default=0.0)),
        "mean_total_variation_temp1": (
            float(np.mean(distances_temp1)) if distances_temp1 else 0.0
        ),
        "max_total_variation_temp1": max(distances_temp1, default=0.0),
        "malom_direction_changes": dict(sorted(direction_counts.items())),
    }


def summarize_primary_contrast(
    position_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen full-vs-empirical-disabled sensitivity rule."""
    if not position_rows:
        raise ValueError("at least one position row is required")
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        by_phase[str(row["phase"])].append(row)
    aggregate = _summarize_rows(position_rows)
    triggers = {
        "argmax_changes": (aggregate["argmax_changes"] >= MATERIAL_ARGMAX_CHANGES),
        "critical_preservation_crossing": (
            aggregate["critical_preservation_crossings"] >= 1
        ),
        "mean_total_variation_scheduled": (
            aggregate["mean_total_variation_scheduled"] >= MATERIAL_MEAN_TOTAL_VARIATION
        ),
    }
    return {
        "contrast": "full_vs_empirical_disabled",
        "thresholds": {
            "minimum_argmax_changes": MATERIAL_ARGMAX_CHANGES,
            "minimum_critical_preservation_crossings": 1,
            "minimum_mean_total_variation_scheduled": (MATERIAL_MEAN_TOTAL_VARIATION),
            "combination": "any",
        },
        "all": aggregate,
        "by_phase": {
            phase: _summarize_rows(rows) for phase, rows in sorted(by_phase.items())
        },
        "triggers": triggers,
        "decision": (
            "material" if any(triggers.values()) else "not_material_on_fixed_corpus"
        ),
    }
