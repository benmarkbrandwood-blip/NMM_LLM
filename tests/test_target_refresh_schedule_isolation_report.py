"""Focused tests for schedule-isolation runtime evidence publication."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import report_target_refresh_schedule_isolation_diagnostic as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )


def _arm_grid(tmp_path: Path) -> dict:
    arms = {}
    for seed in (67, 68, 69):
        for condition in ("refresh-once", "no-refresh"):
            control = tmp_path / f"{seed}-{condition}"
            segment = control / "segments" / "segment-0001"
            update_rows = [
                {
                    "post_fork_consumed_transition_count": count,
                    "batch_steps": 64,
                    "behaviour_temperature_min": 0.9 - count / 1_000_000,
                    "behaviour_temperature_mean": 0.9 - count / 1_000_000,
                    "behaviour_temperature_max": 0.9 - count / 1_000_000,
                }
                for count in range(64, 8192 + 1, 64)
            ]
            _write_jsonl(segment / "update_log.jsonl", update_rows)
            _write_jsonl(
                segment / "train_log.jsonl",
                [{"game_type": "vs_sanmill", "opponent_node_budget": 1000}],
            )
            arms[(seed, condition)] = {
                "control_dir": control.relative_to(tmp_path).as_posix()
            }
    return arms


def test_training_schedule_audit_requires_identical_transition_temperatures(
    tmp_path: Path, monkeypatch
) -> None:
    arms = _arm_grid(tmp_path)
    monkeypatch.setattr(report, "ROOT", tmp_path)

    audited = report._audit_paired_training_schedules(
        arms=arms,
        seeds=(67, 68, 69),
    )

    assert audited["67"]["exact_update_batches"] == 128
    assert audited["67"]["temperature_exposure_byte_equal"] is True
    assert audited["67"]["sanmill_node_budget"] == 1000


def test_training_schedule_temperature_drift_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    arms = _arm_grid(tmp_path)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    path = (
        tmp_path
        / arms[(68, "no-refresh")]["control_dir"]
        / "segments/segment-0001/update_log.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[31]["behaviour_temperature_mean"] += 0.001
    _write_jsonl(path, rows)

    with pytest.raises(
        report.ScheduleIsolationReportError,
        match="temperature exposure differs",
    ):
        report._audit_paired_training_schedules(
            arms=arms,
            seeds=(67, 68, 69),
        )


@pytest.mark.parametrize(
    ("training_outcome", "expected"),
    [
        (report.trainer.WIN_REWARD, ("win", 1.0)),
        (report.trainer.DRAW_SHORT, ("draw", 0.5)),
        (report.trainer.DRAW_LONG, ("draw", 0.5)),
        (report.trainer.LOSS_REWARD, ("loss", 0.0)),
    ],
)
def test_training_rewards_are_projected_to_match_scores(
    training_outcome, expected
) -> None:
    assert report._outcome_class(training_outcome) == expected


def test_completed_row_binds_referee_history_and_checkpoint_ids() -> None:
    scheduled = {
        "schema_version": "nmm.schedule-isolation-outcome-measurement.v1",
        "measurement_index": 0,
    }
    result = SimpleNamespace(
        outcome=report.trainer.WIN_REWARD,
        ply=7,
        termination_reason="loseFewerThanThree",
        step_diags=[],
        specialist_read_stats={},
    )
    start = SimpleNamespace(
        history_sha256="a" * 64,
        logical_ply_count=12,
    )
    end = SimpleNamespace(
        history_sha256="b" * 64,
        logical_ply_count=19,
    )

    row = report._complete_outcome_row(
        scheduled,
        result=result,
        start_state=start,
        end_state=end,
        candidate_checkpoint_id="candidate",
        anchor_checkpoint_id="anchor",
    )

    assert row["score"] == 1.0
    assert row["post_start_logical_plies"] == 7
    assert row["candidate_checkpoint_id"] == "candidate"
    assert row["anchor_checkpoint_id"] == "anchor"
