from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.serve_managed_training_monitor as monitor


def _row(
    game: int,
    *,
    outcome: float,
    game_type: str = "vs_sanmill",
    termination_reason: str = "draw_threefold_repetition",
) -> dict[str, object]:
    return {
        "game": game,
        "outcome": outcome,
        "game_type": game_type,
        "termination_reason": termination_reason,
        "learner_color": "W" if game % 2 else "B",
        # This legacy field name is deliberately misleading: the trainer value
        # is controlled by resolved_config.rolling_win, not by 200 games.
        "win_rate_200": 0.1234,
    }


def test_manifest_rolling_window_controls_mixed_training_metric(
    tmp_path: Path,
) -> None:
    segment = tmp_path / "segment-0001"
    segment.mkdir()
    (segment / "run-manifest.json").write_text(
        json.dumps({"resolved_config": {"rolling_win": 4}}),
        encoding="utf-8",
    )
    assert monitor._rolling_window_from_manifests([segment]) == 4

    rows = [
        _row(game, outcome=1.5 if game in {3, 4, 7} else -1.0)
        for game in range(1, 8)
    ]
    chart = monitor._chart_game_rows(rows, rolling_win=4)
    assert chart[-1]["mixed_win_rate"] == 0.5
    assert chart[-1]["mixed_window_games"] == 4
    assert chart[2]["mixed_win_rate"] is None


def test_source_summaries_separate_rules_draws_from_ply_truncations() -> None:
    rows = [
        _row(1, outcome=0.0, termination_reason="draw_threefold_repetition"),
        _row(2, outcome=0.0, termination_reason="draw_fifty_move"),
        _row(3, outcome=0.0, termination_reason="max-ply-truncation"),
        _row(4, outcome=1.5, termination_reason="win_fewer_than_three"),
    ]
    split = monitor._opponent_outcomes(rows, {"common_trainer_args": []})
    sanmill = split["bySource"]["vs_sanmill"]

    assert sanmill["overall"]["ruleDraw"] == 2
    assert sanmill["overall"]["maxPly"] == 1
    assert sanmill["recentSourceGames"]["window"] == 4
    assert sanmill["recentSourceGames"]["winRate"] == 0.25
    assert sanmill["recentSourceGames"]["scoreRate"] == 0.625


def test_manifest_rolling_window_rejects_segment_drift(tmp_path: Path) -> None:
    segments = []
    for index, window in enumerate((40, 200), start=1):
        segment = tmp_path / f"segment-{index:04d}"
        segment.mkdir()
        (segment / "run-manifest.json").write_text(
            json.dumps({"resolved_config": {"rolling_win": window}}),
            encoding="utf-8",
        )
        segments.append(segment)

    with pytest.raises(ValueError, match="disagree on rolling_win"):
        monitor._rolling_window_from_manifests(segments)


def test_dashboard_exposes_source_and_termination_kpis_with_help() -> None:
    assert 'id="frozenRecent"' in monitor.HTML
    assert 'id="sanmillRecent"' in monitor.HTML
    assert 'data-i18n="tableRuleDraws"' in monitor.HTML
    assert 'data-i18n="tableMaxPly"' in monitor.HTML
    assert "sourceRecent: {" in monitor.HTML
    assert "terminationSplit: {" in monitor.HTML


def test_dashboard_orders_outcome_bars_as_win_draw_loss() -> None:
    assert "const OUTCOME_BAR_ORDER=Object.freeze(['win','draw','loss']);" in monitor.HTML
    assert (
        "bars('outcomeBars',data.counts.outcomes,"
        "{win:COLORS.blue,draw:COLORS.yellow,loss:COLORS.magenta},"
        "OUTCOME_BAR_ORDER);"
    ) in monitor.HTML


def test_learning_rate_series_preserves_actual_step_boundary() -> None:
    rows = [
        {"game": 1, "lr": 1e-4},
        {"game": 50, "lr": 1e-4},
        {"game": 51, "lr": 5e-5},
        {"game": 5000, "lr": 5e-5},
    ]

    assert monitor._learning_rate_step_rows(rows) == [
        {"game": 1, "lr_x1e4": 1.0},
        {"game": 51, "lr_x1e4": 0.5},
        {"game": 5000, "lr_x1e4": 0.5},
    ]


def test_dashboard_renders_learning_rate_as_actual_steps() -> None:
    assert 'id="lrChartNote"' in monitor.HTML
    assert "lineChart('lrChart',data.series.learningRate||[]" in monitor.HTML
    assert "stepped:true" in monitor.HTML
    assert "learningRateNote(data.series.learningRate||[])" in monitor.HTML


def test_gpu_telemetry_keeps_only_managed_training_window_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "event_type": "managed_segment_started",
            "timestamp_utc": "2026-08-13T04:52:00Z",
        },
        {
            "event_type": "managed_segment_completed",
            "timestamp_utc": "2026-08-13T04:57:00Z",
        },
    ]
    windows = monitor._managed_training_windows(events)
    telemetry_path = tmp_path / "local-monitor" / "gpu-telemetry.jsonl"
    telemetry_path.parent.mkdir()
    rows = [
        {
            "game": 100,
            "timestampUtc": "2026-08-13T04:55:00Z",
            "gpuUtilPct": 75.0,
            "memoryUtilPct": 20.0,
        },
        {
            "game": 250,
            "timestampUtc": "2026-08-13T05:10:00Z",
            "gpuUtilPct": 5.0,
            "memoryUtilPct": 30.0,
        },
    ]
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    def unexpected_sample(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed runs must not sample the live GPU")

    monkeypatch.setattr(monitor.subprocess, "run", unexpected_sample)
    status = monitor._gpu_status(
        tmp_path,
        250,
        training_windows=windows,
        sampling_active=False,
    )

    assert status["available"] is True
    assert status["scope"] == "managed-training-window-whole-device"
    assert status["excludedOutsideTrainingWindow"] == 1
    assert [row["game"] for row in status["series"]] == [100]


def test_gpu_telemetry_is_unavailable_when_only_post_training_samples_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = monitor._managed_training_windows(
        [
            {
                "event_type": "managed_segment_started",
                "timestamp_utc": "2026-08-13T04:52:00Z",
            },
            {
                "event_type": "managed_plan_completed",
                "timestamp_utc": "2026-08-13T06:49:14Z",
            },
        ]
    )
    telemetry_path = tmp_path / "local-monitor" / "gpu-telemetry.jsonl"
    telemetry_path.parent.mkdir()
    telemetry_path.write_text(
        json.dumps(
            {
                "game": 5000,
                "timestampUtc": "2026-08-13T07:31:34Z",
                "gpuUtilPct": 10.0,
                "memoryUtilPct": 18.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed runs must not sample the live GPU")
        ),
    )

    status = monitor._gpu_status(
        tmp_path,
        5000,
        training_windows=windows,
        sampling_active=False,
    )

    assert status["available"] is False
    assert status["latest"] == {}
    assert status["series"] == []
    assert status["excludedOutsideTrainingWindow"] == 1


def test_dashboard_labels_gpu_as_training_window_whole_device_telemetry() -> None:
    assert "cardGpu:'训练期 GPU 遥测'" in monitor.HTML
    assert "panelGpu:'训练期间 GPU 与显存遥测（整卡）'" in monitor.HTML
    assert 'data-i18n="gpuTrainingNote"' in monitor.HTML
    assert "excludedOutsideTrainingWindow" in monitor.HTML
