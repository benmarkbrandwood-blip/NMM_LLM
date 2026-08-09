"""Focused tests for deterministic mill-bonus ablation result analysis."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.evaluation.mill_bonus_ablation_result import (
    MillBonusAblationResultError,
    _validate_manifest,
    decide_paired_result,
    publish_result,
    summarize_game_rows,
    summarize_update_rows,
)
from learned_ai.training.run_contract import canonical_json_bytes


def _game_row(
    game: int,
    *,
    source: str,
    colour: str,
    downgrade: int = 0,
) -> dict:
    return {
        "game_id": f"game:{game:04d}",
        "game": game,
        "difficulty": 1,
        "learner_color": colour,
        "temperature": 0.9 - game / 10000,
        "outcome": -1.0,
        "ply": 30,
        "steps": 15,
        "update_policy_loss": None,
        "update_value_loss": None,
        "update_entropy": None,
        "reward_total_mean": -0.5,
        "reward_mill_bonus_mean": 0.0,
        "mill_bonus_awarded_total": 0.0,
        "chosen_prob_mean": 0.2,
        "entropy_mean": 2.0,
        "policy_top1_rate": 0.2,
        "heuristic_top1_rate": 0.1,
        "malom_preserving_move_rate": 0.8,
        "malom_downgrade_move_rate": 0.2,
        "game_type": source,
        "phase_bucket": "main",
        "is_branch": 0,
        "termination_reason": "lose_no_legal_moves",
        "opponent_node_budget": 1000 if source == "vs_sanmill" else None,
        "formed_mill_count": 1,
        "formed_mill_move_count": 1,
        "formed_mill_malom_unknown_count": 0,
        "formed_mill_malom_downgrade_count": downgrade,
        "formed_mill_malom_downgrade_rate": float(downgrade),
        "formed_mill_malom_known_place": 1,
        "formed_mill_malom_known_move": 0,
        "formed_mill_malom_known_fly": 0,
        "formed_mill_malom_downgrade_place": downgrade,
        "formed_mill_malom_downgrade_move": 0,
        "formed_mill_malom_downgrade_fly": 0,
    }


def _five_hundred_rows() -> tuple[list[dict], dict[str, int]]:
    schedule = {
        "frozen_black": 125,
        "frozen_white": 125,
        "sanmill_black": 125,
        "sanmill_white": 125,
    }
    classes = (
        ("vs_frozen", "B"),
        ("vs_frozen", "W"),
        ("vs_sanmill", "B"),
        ("vs_sanmill", "W"),
    )
    rows = [
        _game_row(
            game,
            source=classes[(game - 1) % 4][0],
            colour=classes[(game - 1) % 4][1],
            downgrade=int(game >= 301 and game % 4 == 0),
        )
        for game in range(1, 501)
    ]
    return rows, schedule


def test_summary_uses_complete_tail_and_rolling_windows() -> None:
    rows, schedule = _five_hundred_rows()

    summary = summarize_game_rows(
        rows,
        expected_games=500,
        expected_schedule_counts=schedule,
    )

    assert summary["primary"]["whole_run"] == {
        "downgrading_known_mill_actions": 50,
        "known_mill_actions": 500,
        "rate": 0.1,
    }
    assert summary["primary"]["tail_301_500"] == {
        "downgrading_known_mill_actions": 50,
        "known_mill_actions": 200,
        "rate": 0.25,
    }
    assert summary["primary"]["tail_by_phase"]["place"]["rate"] == 0.25
    assert summary["primary"]["tail_by_phase"]["move"]["rate"] is None
    rolling = summary["curves"]["rolling_50_complete_windows_only"]
    assert len(rolling) == 451
    assert rolling[0]["game"] == 50
    assert rolling[-1]["game"] == 500
    assert summary["curves"]["validation"]["available"] is False
    assert summary["wdl"]["all"]["losses"] == 500
    assert summary["wdl"]["by_opponent_and_colour"][
        "vs_sanmill:W"
    ]["games"] == 125


def test_summary_rejects_unreconciled_phase_support() -> None:
    rows, schedule = _five_hundred_rows()
    rows[20]["formed_mill_malom_known_place"] = 0

    with pytest.raises(
        MillBonusAblationResultError,
        match="phase Mill counts",
    ):
        summarize_game_rows(
            rows,
            expected_games=500,
            expected_schedule_counts=schedule,
        )


def test_summary_rejects_schedule_drift() -> None:
    rows, schedule = _five_hundred_rows()
    rows[0]["game_type"] = "vs_sanmill"
    rows[0]["opponent_node_budget"] = 1000

    with pytest.raises(
        MillBonusAblationResultError,
        match="scheduled opponent/colour counts differ",
    ):
        summarize_game_rows(
            rows,
            expected_games=500,
            expected_schedule_counts=schedule,
        )


def _arm(seed: int, mode: str, rate: float, *, safe: bool = True) -> dict:
    return {
        "seed": seed,
        "mill_bonus_mode": mode,
        "policy_health": {"passed": safe},
        "metrics": {
            "primary": {
                "tail_301_500": {
                    "downgrading_known_mill_actions": int(rate * 100),
                    "known_mill_actions": 100,
                    "rate": rate,
                }
            }
        },
    }


def test_paired_decision_requires_two_seeds_and_material_median() -> None:
    arms = [
        _arm(42, "legacy-unconditional", 0.30),
        _arm(42, "malom-preserving-only", 0.20),
        _arm(43, "legacy-unconditional", 0.25),
        _arm(43, "malom-preserving-only", 0.15),
        _arm(44, "legacy-unconditional", 0.20),
        _arm(44, "malom-preserving-only", 0.22),
    ]

    decision = decide_paired_result(arms, material_reduction=0.05)

    assert decision["verdict"] == "supports_malom_preserving_only"
    assert decision["pairs_favouring_corrected"] == 2
    assert decision["median_legacy_minus_corrected_rate"] == (
        pytest.approx(0.10)
    )

    arms[-1]["policy_health"]["passed"] = False
    unsafe_decision = decide_paired_result(arms, material_reduction=0.05)
    assert unsafe_decision["verdict"] == "inconclusive"
    assert unsafe_decision["corrected_arms_pass_safety"] is False


def test_update_curve_is_separate_and_finite() -> None:
    rows = [
        {
            "game": 10,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 2.0,
            "lr": 0.0001,
            "batch_steps": 64,
            "reason": "periodic",
        },
        {
            "game": 500,
            "policy_loss": 0.05,
            "value_loss": 0.1,
            "entropy": 1.5,
            "lr": 0.00005,
            "batch_steps": 32,
            "reason": "final_flush",
        },
    ]

    result = summarize_update_rows(rows, expected_games=500)

    assert result["updates"] == 2
    assert result["raw"][-1]["reason"] == "final_flush"
    assert result["validation"]["available"] is False

    rows[-1]["entropy"] = float("nan")
    with pytest.raises(MillBonusAblationResultError, match="must be finite"):
        summarize_update_rows(rows, expected_games=500)


def test_manifest_must_match_readiness_protocol_and_assets() -> None:
    mif = {
        "tag": "mif-suite-1.0",
        "releaseCommit": "a" * 40,
        "suiteJcsSha256": "sha256:" + "b" * 64,
        "releaseManifestSha256": "sha256:" + "c" * 64,
    }
    ruleset = {"semanticDigest": "sha256:" + "d" * 64}
    checks = {
        "malom": {"identity": "e" * 64},
        "specialist_db": {
            "identity": "f" * 64,
            "content_sha256": "1" * 64,
        },
        "human_db": {"identity": "2" * 64},
        "sanmill_training": {"identity": "3" * 64},
    }
    manifest = {
        "schema_version": "nmm.run-manifest.v1",
        "git_commit": "4" * 40,
        "git_dirty": False,
        "experiment_id": "experiment",
        "resolved_config": {
            "seed": 42,
            "mill_bonus_mode": "legacy-unconditional",
            "max_games": 5000,
            "segment_games": 500,
            "segment_stop_game": 500,
            "start_mode": "fresh",
            "referee_engine": "sanmill",
            "opponent_engine": "sanmill",
        },
        "checkpoint_policy": {"mifSuite": mif, "ruleset": ruleset},
        "assets": [
            {"logical_name": "mif_suite_1_0", "identity": mif["releaseManifestSha256"]},
            {"logical_name": "training_ruleset", "identity": ruleset["semanticDigest"]},
            {"logical_name": "malom_tablebase", "identity": "e" * 64},
            {"logical_name": "specialist_db", "identity": "f" * 64},
            {"logical_name": "human_db", "identity": "2" * 64},
            {"logical_name": "sanmill_training_runtime", "identity": "3" * 64},
        ],
    }
    contract = {
        "common_training_contract": {
            "max_games_schedule": 5000,
            "one_segment_games": 500,
        },
        "resources": {"completed_games_per_arm": 500},
        "rules_and_runtime": {
            "mif_tag": "mif-suite-1.0",
            "mif_release_commit": "a" * 40,
            "mif_suite_jcs_sha256": "b" * 64,
            "rules_semantic_digest": "sha256:" + "d" * 64,
        },
        "data_contract": {
            "malom_manifest_identity": "e" * 64,
            "human_db_identity": "2" * 64,
            "specialist_db_initial_template": {"sha256": "1" * 64},
        },
    }
    preflight = {
        "schema_version": "nmm.generalist-preflight.v1",
        "verdict": "needs_decision",
        "errors": [],
        "unresolved_decisions": [
            "long-run launch requires a frozen managed plan and separate "
            "product authorization"
        ],
        "resume_config_sha256": "5" * 64,
        "mifSuite": mif,
        "ruleset": ruleset,
        "checks": checks,
    }
    plan = SimpleNamespace(
        git_commit="4" * 40,
        experiment_id="experiment",
        resume_config_sha256="5" * 64,
    )
    arm = {"seed": 42, "mill_bonus_mode": "legacy-unconditional"}

    _validate_manifest(
        manifest,
        plan=plan,
        arm=arm,
        contract=contract,
        preflight=preflight,
    )

    preflight["unresolved_decisions"].append("another decision")
    with pytest.raises(
        MillBonusAblationResultError,
        match="readiness preflight content differs",
    ):
        _validate_manifest(
            manifest,
            plan=plan,
            arm=arm,
            contract=contract,
            preflight=preflight,
        )
    preflight["unresolved_decisions"].pop()

    manifest["assets"][-1]["identity"] = "6" * 64
    with pytest.raises(
        MillBonusAblationResultError,
        match="sanmill_training_runtime",
    ):
        _validate_manifest(
            manifest,
            plan=plan,
            arm=arm,
            contract=contract,
            preflight=preflight,
        )


def test_result_publication_is_canonical_and_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    report = {"schema_version": "test", "value": 1}

    publish_result(path, report)

    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert path.read_bytes() == canonical_json_bytes(report)
    with pytest.raises(MillBonusAblationResultError, match="already exists"):
        publish_result(path, report)
