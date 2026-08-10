"""Focused tests for fixed-anchor, no-update development measurements."""

from __future__ import annotations

from types import SimpleNamespace

from scripts import train_s_gen_v2 as trainer


def test_measurement_cli_options_are_explicit_and_resume_bound() -> None:
    parser = trainer._build_argument_parser()
    baseline = parser.parse_args([])
    configured = parser.parse_args(
        [
            "--optimizer-update-bound",
            "34",
            "--measurement-anchor-game",
            "50",
            "--measurement-anchor-expected-update-count",
            "18",
            "--measurement-every-updates",
            "4",
            "--measurement-games-per-opponent",
            "8",
            "--measurement-sanmill-node-budget",
            "1000",
            "--measurement-temperature",
            "0.2",
        ]
    )

    assert baseline.optimizer_update_bound is None
    assert baseline.measurement_anchor_game is None
    assert configured.optimizer_update_bound == 34
    assert configured.measurement_anchor_game == 50
    assert configured.measurement_anchor_expected_update_count == 18
    assert configured.measurement_every_updates == 4
    assert configured.measurement_games_per_opponent == 8
    assert configured.measurement_sanmill_node_budget == 1_000
    assert configured.measurement_temperature == 0.2
    assert trainer.resume_config_sha256(configured) != trainer.resume_config_sha256(
        baseline
    )


def test_measurement_metrics_preserve_outcome_policy_and_read_evidence() -> None:
    steps = [
        SimpleNamespace(
            malom_quality=0.0,
            malom_chosen_dtm=None,
            chosen_prob=0.75,
            entropy=0.4,
            was_top1_policy=1,
            was_top1_heuristic=0,
        ),
        SimpleNamespace(
            malom_quality=None,
            malom_chosen_dtm=-1.0,
            chosen_prob=0.25,
            entropy=0.8,
            was_top1_policy=0,
            was_top1_heuristic=1,
        ),
        SimpleNamespace(
            malom_quality=None,
            malom_chosen_dtm=None,
            chosen_prob=0.5,
            entropy=0.6,
            was_top1_policy=1,
            was_top1_heuristic=1,
        ),
    ]
    result = trainer.RolloutResult(
        trajectory=[],
        step_diags=steps,
        outcome=trainer.DRAW_SHORT,
        ply=42,
        termination_reason="draw_threefold_repetition",
        branch_candidates=[],
        specialist_read_stats={"effective_known": 7, "mode": "fallback"},
    )

    metrics = trainer._development_measurement_metrics(result)

    assert metrics == {
        "outcome": trainer.DRAW_SHORT,
        "ply": 42,
        "termination_reason": "draw_threefold_repetition",
        "steps": 3,
        "chosen_probability_mean": 0.5,
        "entropy_mean": 0.6,
        "policy_top1_rate": 2 / 3,
        "heuristic_top1_rate": 2 / 3,
        "malom_preserving_rate": 0.5,
        "malom_known_steps": 2,
        "specialist_read_stats": {"effective_known": 7, "mode": "fallback"},
    }
