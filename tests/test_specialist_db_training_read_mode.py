"""Focused tests for explicit SpecialistDB training read modes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from learned_ai.data.data_contract import TypedLabel
from learned_ai.data.specialist_db import SpecialistWdlEvidence
from learned_ai.data.specialist_read_view import (
    SpecialistTrainingReadView,
    project_training_wdl,
    specialist_read_stats_delta,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


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


def _evidence() -> SpecialistWdlEvidence:
    return SpecialistWdlEvidence(
        perspective="B",
        theoretical_wdl=_label("W"),
        empirical_counts=(1, 1, 8),
        empirical_distribution=(0.1, 0.1, 0.8),
    )


@dataclass
class _Database:
    evidence: SpecialistWdlEvidence | None
    recorded: int = 0
    labelled: int = 0

    def query_wdl_evidence(self, _board, min_samples: int):
        assert min_samples == 3
        return self.evidence

    def record_game(self, *_args, **_kwargs):
        self.recorded += 1

    def label_position_malom(self, *_args, **_kwargs):
        self.labelled += 1


def test_full_and_theoretical_only_have_explicit_projection_semantics() -> None:
    evidence = _evidence()

    assert project_training_wdl(evidence, "full") == (0.1, 0.1, 0.8)
    assert project_training_wdl(evidence, "theoretical-only") == (
        0.9,
        0.05,
        0.05,
    )


def test_theoretical_only_suppresses_empirical_reads_but_delegates_writes() -> None:
    database = _Database(_evidence())
    view = SpecialistTrainingReadView(database, "theoretical-only")
    before = view.snapshot_stats()

    assert view.query_wdl(object(), min_samples=3) == (0.9, 0.05, 0.05)
    view.record_game([], "D", [], "gen")
    view.label_position_malom(object(), "D")
    delta = specialist_read_stats_delta(before, view.snapshot_stats())

    assert delta == {
        "mode": "theoretical-only",
        "queries": 1,
        "rows_present": 1,
        "theoretical_available": 1,
        "empirical_available": 1,
        "projections_returned": 1,
        "empirical_suppressed": 1,
    }
    assert database.recorded == 1
    assert database.labelled == 1


def test_full_mode_reports_empirical_use_without_suppression() -> None:
    view = SpecialistTrainingReadView(_Database(_evidence()), "full")

    assert view.query_wdl(object(), min_samples=3) == (0.1, 0.1, 0.8)
    stats = view.snapshot_stats()

    assert stats["empirical_available"] == 1
    assert stats["empirical_suppressed"] == 0
    assert stats["projections_returned"] == 1


def test_rollout_counters_are_isolated_per_worker_thread() -> None:
    view = SpecialistTrainingReadView(_Database(_evidence()), "full")

    def run_queries(count: int) -> int:
        before = view.snapshot_stats()
        for _ in range(count):
            view.query_wdl(object(), min_samples=3)
        delta = specialist_read_stats_delta(before, view.snapshot_stats())
        return int(delta["queries"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(executor.map(run_queries, (2, 5)))

    assert observed == [2, 5]


@pytest.mark.parametrize("mode", ["", "disabled", "empirical-only"])
def test_unknown_training_read_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SpecialistTrainingReadView(_Database(None), mode)


def test_trainer_cli_defaults_to_legacy_full_and_accepts_ablation() -> None:
    parser = trainer._build_argument_parser()

    default = parser.parse_args(["--preflight", "smoke"])
    ablation = parser.parse_args(
        [
            "--preflight",
            "smoke",
            "--specialist-read-mode",
            "theoretical-only",
        ]
    )

    assert default.specialist_read_mode == "full"
    assert ablation.specialist_read_mode == "theoretical-only"


def test_managed_plan_propagates_the_explicit_read_mode(tmp_path) -> None:
    args = manager._build_parser().parse_args(
        [
            "prepare",
            "--control-dir",
            str(tmp_path / "control"),
            "--max-wall-hours",
            "1",
            "--objective",
            "focused test",
            "--experiment-id",
            "focused-test",
            "--seed",
            "61",
            "--max-ply",
            "120",
            "--mill-bonus-mode",
            "malom-preserving-only",
            "--specialist-read-mode",
            "theoretical-only",
        ]
    )

    common_args = manager._common_trainer_args(
        args,
        tmp_path / "training_paths.local.json",
    )
    index = common_args.index("--specialist-read-mode")

    assert common_args[index + 1] == "theoretical-only"
