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
