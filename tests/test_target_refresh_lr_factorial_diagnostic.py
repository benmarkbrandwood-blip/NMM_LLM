"""Focused tests for the target-refresh/LR factorial diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.target_refresh_lr_factorial_result import (
    TargetRefreshLrResultError,
    decide_factorial_result,
    validate_paired_boundary,
)
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_lr_factorial_diagnostic import (
    CONDITION_FACTORS,
    DEFAULT_CONTRACT,
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    TargetRefreshLrDiagnosticError,
    build_prepare_commands,
    load_target_refresh_lr_contract,
    validate_prepare_commands,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / DEFAULT_CONTRACT


def _write_with_identity(path: Path, contract: dict) -> None:
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path.write_text(json.dumps(contract), encoding="utf-8")


def test_loader_accepts_the_frozen_two_seed_factorial() -> None:
    contract = load_target_refresh_lr_contract(CONTRACT_PATH)

    assert contract["plan_identity"] == (
        "94f6381a40ab86401cb0e957677dd3a21dde01ed9ffd4c69b3fa252b21787e58"
    )
    assert [
        (arm["seed"], arm["condition"])
        for arm in contract["arms"]
    ] == [
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    assert contract["authorization"]["launch_authorized"] is False


def test_frozen_result_implementation_hashes_match_repository_bytes() -> None:
    contract = load_target_refresh_lr_contract(CONTRACT_PATH)

    for record in contract["analysis"]["result_implementation"].values():
        if not isinstance(record, dict):
            continue
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_loader_rejects_identity_tampering(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["resources"]["maximum_completed_games_total"] = 801
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(TargetRefreshLrDiagnosticError, match="identity differs"):
        load_target_refresh_lr_contract(changed)


def test_loader_rejects_a_third_training_factor_after_rehash(
    tmp_path: Path,
) -> None:
    contract = copy.deepcopy(load_target_refresh_lr_contract(CONTRACT_PATH))
    contract["arms"][1]["unexpected_reward_scale"] = 0.5
    changed = tmp_path / "changed.json"
    _write_with_identity(changed, contract)

    with pytest.raises(
        TargetRefreshLrDiagnosticError,
        match="outside the factor allowlist",
    ):
        load_target_refresh_lr_contract(changed)


def test_prepare_commands_are_nonlaunching_and_bind_only_two_factors() -> None:
    contract = load_target_refresh_lr_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=ROOT / "data/training_paths.local.json",
        python_executable="python-under-test",
    )

    validate_prepare_commands(contract, commands)
    assert len(commands) == 8
    for arm, command in zip(contract["arms"], commands, strict=True):
        assert command[:3] == [
            "python-under-test",
            str(ROOT / "scripts/manage_generalist_run.py"),
            "prepare",
        ]
        assert "authorize" not in command
        assert "run-next" not in command
        args = manager._build_parser().parse_args(command[2:])
        assert args.seed == arm["seed"]
        assert args.target_refresh_every == arm["target_refresh_every_games"]
        assert args.lr_adaptation_mode == arm["lr_adaptation_mode"]
        assert args.specialist_read_mode == "theoretical-only"
        assert args.completion_game_bound == 100
        assert args.segment_games == 100
        assert args.max_games == 5000
        assert args.max_wall_hours == 0.25
        assert args.mill_bonus_mode == "malom-preserving-only"
        assert args.malom_policy_aux_coef == 0.0
        assert args.engine_profile == "sanmill-fixed-resource"


def test_command_validator_rejects_a_hidden_factor_change() -> None:
    contract = load_target_refresh_lr_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=ROOT / "data/training_paths.local.json",
        python_executable="python-under-test",
    )
    changed = copy.deepcopy(commands)
    index = changed[1].index("--target-refresh-every")
    changed[1][index + 1] = "5001"

    with pytest.raises(
        TargetRefreshLrDiagnosticError,
        match="target-refresh factor differs",
    ):
        validate_prepare_commands(contract, changed)


def _decision_arm(seed: int, condition: str, score: float) -> dict:
    return {
        "arm_id": f"seed{seed}-{condition}",
        "seed": seed,
        "condition": condition,
        "metrics": {
            "games": {
                "windows": {
                    "post_boundary_51_100": {
                        "by_opponent_source": {
                            "vs_frozen": {"score": float(score)}
                        }
                    }
                }
            }
        },
    }


def test_factorial_decision_reports_main_effects_and_interaction() -> None:
    scores = {
        64: {
            "refresh-adaptive": 0.05,
            "refresh-fixed": 0.25,
            "no-refresh-adaptive": 0.35,
            "no-refresh-fixed": 0.70,
        },
        65: {
            "refresh-adaptive": 0.10,
            "refresh-fixed": 0.30,
            "no-refresh-adaptive": 0.45,
            "no-refresh-fixed": 0.80,
        },
    }
    arms = [
        _decision_arm(seed, condition, scores[seed][condition])
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]

    decision = decide_factorial_result(arms)

    assert decision["classification"] == "factor_signal_detected"
    assert decision["supported_terms"] == [
        "target_refresh",
        "learning_rate",
        "interaction",
    ]
    assert decision["target_refresh"]["direction"] == "positive"
    assert decision["learning_rate"]["direction"] == "positive"
    assert decision["interaction"]["direction"] == "positive"


def _boundary_rows(condition: str) -> list[dict]:
    refresh_every, lr_mode = CONDITION_FACTORS[condition]
    rows = []
    for game in range(1, 101):
        rows.append(
            {
                "game": game,
                "game_type": "vs_sanmill" if game % 2 == 0 else "vs_frozen",
                "outcome": trainer.LOSS_REWARD,
                "target_age": game,
                "lr": 0.0001,
            }
        )
    if refresh_every == 50:
        for game in range(51, 101):
            rows[game - 1]["target_age"] = game - 50
    if lr_mode == "adaptive-search-opponent-win-rate":
        for game in range(51, 101):
            rows[game - 1]["lr"] = 0.00005
    return rows


def test_boundary_validator_requires_exact_pre_pairing_and_engaged_factors() -> None:
    arms = [
        {
            "seed": seed,
            "condition": condition,
            "game_rows": _boundary_rows(condition),
        }
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]

    report = validate_paired_boundary(arms, base_rate=0.0001)

    assert len(report) == 2
    assert all(item["first_50_games_byte_identical"] for item in report)
    assert all(item["adaptive_expected_game_51_lr"] == 0.00005 for item in report)


def test_boundary_validator_rejects_a_pre_boundary_difference() -> None:
    arms = [
        {
            "seed": seed,
            "condition": condition,
            "game_rows": _boundary_rows(condition),
        }
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    arms[1]["game_rows"][10]["outcome"] = trainer.DRAW_SHORT

    with pytest.raises(TargetRefreshLrResultError, match="before the intervention"):
        validate_paired_boundary(arms, base_rate=0.0001)
