"""Focused tests for the read-only SpecialistDB policy-mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from learned_ai.data.data_contract import TypedLabel
from learned_ai.data.specialist_db import SpecialistWdlEvidence
from learned_ai.validation.specialist_db_policy_mechanism import (
    PROJECTION_MODES,
    SpecialistEvidenceProjection,
    evidence_record,
    project_wdl,
    summarize_primary_contrast,
    total_variation,
)


def _label(value: str) -> TypedLabel:
    return TypedLabel(
        kind="theoretical_wdl",
        value=value,
        perspective="B",
        rules_version="test-rules",
        history_identity="test-history",
        source_identity="test-source",
        validity_version="sector-corrected-v1",
    )


def _evidence(
    *,
    theoretical: str | None,
    counts: tuple[int, int, int],
    distribution: tuple[float, float, float] | None,
) -> SpecialistWdlEvidence:
    return SpecialistWdlEvidence(
        perspective="B",
        theoretical_wdl=None if theoretical is None else _label(theoretical),
        empirical_counts=counts,
        empirical_distribution=distribution,
    )


def test_projection_modes_separate_theoretical_and_empirical_evidence() -> None:
    evidence = _evidence(
        theoretical="W",
        counts=(1, 1, 8),
        distribution=(0.1, 0.1, 0.8),
    )

    assert project_wdl(evidence, "full") == (0.1, 0.1, 0.8)
    assert project_wdl(evidence, "empirical_disabled") == (0.9, 0.05, 0.05)
    assert project_wdl(evidence, "malom_disabled") == (0.1, 0.1, 0.8)
    assert project_wdl(evidence, "all_disabled") is None


def test_full_projection_matches_legacy_low_support_prior() -> None:
    evidence = _evidence(
        theoretical="D",
        counts=(1, 0, 0),
        distribution=None,
    )

    assert project_wdl(evidence, "full") == (0.05, 0.9, 0.05)
    assert project_wdl(evidence, "empirical_disabled") == (0.05, 0.9, 0.05)
    assert project_wdl(evidence, "malom_disabled") is None


@dataclass
class _Database:
    evidence: SpecialistWdlEvidence
    observed_min_samples: int | None = None

    def query_wdl_evidence(self, _board, min_samples: int):
        self.observed_min_samples = min_samples
        return self.evidence


def test_encoder_projection_delegates_one_separated_query() -> None:
    database = _Database(
        _evidence(
            theoretical="L",
            counts=(0, 0, 0),
            distribution=None,
        )
    )
    projection = SpecialistEvidenceProjection(database, "empirical_disabled")

    assert projection.query_wdl(object(), min_samples=3) == (0.05, 0.05, 0.9)
    assert database.observed_min_samples == 3


def test_evidence_record_exposes_disagreement_and_projection_hits() -> None:
    record = evidence_record(
        _evidence(
            theoretical="W",
            counts=(1, 1, 8),
            distribution=(0.1, 0.1, 0.8),
        )
    )

    assert record["empirical_samples"] == 10
    assert record["empirical_modal_wdl"] == ["L"]
    assert record["theoretical_empirical_disagreement"] is True
    assert record["projection_hits"] == {
        mode: mode != "all_disabled" for mode in PROJECTION_MODES
    }


def _position(
    index: int,
    *,
    phase: str = "movement",
    full_index: int = 0,
    without_index: int = 0,
    full_quality: float | None = 0.0,
    without_quality: float | None = 0.0,
    full_probabilities: list[float] | None = None,
    without_probabilities: list[float] | None = None,
    critical: bool = True,
) -> dict:
    full_probabilities = full_probabilities or [0.6, 0.4]
    without_probabilities = without_probabilities or [0.6, 0.4]
    modes = {
        "full": {
            "argmax_index": full_index,
            "argmax_malom_quality": full_quality,
            "probabilities_scheduled": full_probabilities,
            "probabilities_temp1": full_probabilities,
        },
        "empirical_disabled": {
            "argmax_index": without_index,
            "argmax_malom_quality": without_quality,
            "probabilities_scheduled": without_probabilities,
            "probabilities_temp1": without_probabilities,
        },
    }
    return {
        "index": index,
        "phase": phase,
        "critical": critical,
        "modes": modes,
    }


def test_three_argmax_changes_trigger_material_decision() -> None:
    rows = [
        _position(index, full_index=0, without_index=1, critical=False)
        for index in range(1, 4)
    ]

    report = summarize_primary_contrast(rows)

    assert report["all"]["argmax_changes"] == 3
    assert report["triggers"]["argmax_changes"] is True
    assert report["decision"] == "material"


def test_one_critical_malom_direction_crossing_is_material() -> None:
    rows = [
        _position(
            7,
            full_index=0,
            without_index=1,
            full_quality=0.0,
            without_quality=-1.0,
        )
    ]

    report = summarize_primary_contrast(rows)

    assert report["all"]["critical_preservation_crossings"] == 1
    assert report["triggers"]["critical_preservation_crossing"] is True
    assert report["decision"] == "material"


def test_distribution_threshold_and_non_material_result_are_stable() -> None:
    material = summarize_primary_contrast(
        [
            _position(
                1,
                full_probabilities=[0.6, 0.4],
                without_probabilities=[0.5, 0.5],
                critical=False,
            )
        ]
    )
    quiet = summarize_primary_contrast(
        [
            _position(
                1,
                full_probabilities=[0.51, 0.49],
                without_probabilities=[0.5, 0.5],
                critical=False,
            )
        ]
    )

    assert material["all"]["mean_total_variation_scheduled"] == pytest.approx(0.1)
    assert material["decision"] == "material"
    assert quiet["decision"] == "not_material_on_fixed_corpus"
    assert total_variation([0.51, 0.49], [0.5, 0.5]) == pytest.approx(0.01)


@pytest.mark.parametrize("mode", ["unknown", "", "FULL"])
def test_unknown_projection_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        project_wdl(None, mode)
