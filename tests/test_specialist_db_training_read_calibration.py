"""Focused tests for the SpecialistDB read calibration contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.specialist_db_training_read_calibration import (
    DEFAULT_CONTRACT,
    SpecialistReadCalibrationError,
    build_prepare_commands,
    load_specialist_read_calibration_contract,
    validate_prepare_commands,
)
from scripts import manage_generalist_run as manager


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / DEFAULT_CONTRACT


def _write_with_identity(path: Path, contract: dict) -> None:
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path.write_text(json.dumps(contract), encoding="utf-8")


def test_loader_accepts_the_frozen_three_seed_pairing() -> None:
    contract = load_specialist_read_calibration_contract(CONTRACT_PATH)

    assert contract["plan_identity"] == (
        "032c2647b9211dd1292220c92431206097838c79beef582c5bd98e48fc85b772"
    )
    assert [(arm["seed"], arm["specialist_read_mode"]) for arm in contract["arms"]] == [
        (61, "full"),
        (61, "theoretical-only"),
        (62, "full"),
        (62, "theoretical-only"),
        (63, "full"),
        (63, "theoretical-only"),
    ]
    assert contract["authorization"]["launch_authorized"] is False


def test_loader_rejects_identity_tampering(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["resources"]["maximum_completed_games_total"] = 1501
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(SpecialistReadCalibrationError, match="identity differs"):
        load_specialist_read_calibration_contract(changed)


def test_loader_rejects_a_second_training_factor_after_rehash(
    tmp_path: Path,
) -> None:
    contract = copy.deepcopy(load_specialist_read_calibration_contract(CONTRACT_PATH))
    contract["arms"][1]["unexpected_reward_scale"] = 0.5
    changed = tmp_path / "changed.json"
    _write_with_identity(changed, contract)

    with pytest.raises(
        SpecialistReadCalibrationError,
        match="outside the allowlist",
    ):
        load_specialist_read_calibration_contract(changed)


def test_prepare_commands_are_nonlaunching_and_change_only_read_mode() -> None:
    contract = load_specialist_read_calibration_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=ROOT / "data/training_paths.local.json",
        python_executable="python-under-test",
    )

    validate_prepare_commands(contract, commands)
    assert len(commands) == 6
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
        assert args.specialist_read_mode == arm["specialist_read_mode"]
        assert args.completion_game_bound == 250
        assert args.segment_games == 250
        assert args.max_games == 5000
        assert args.max_wall_hours == 0.5
        assert args.mill_bonus_mode == "malom-preserving-only"
        assert args.malom_policy_aux_coef == 0.0
        assert args.engine_profile == "sanmill-fixed-resource"


def test_command_validator_rejects_a_hidden_mode_fallback() -> None:
    contract = load_specialist_read_calibration_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=ROOT / "data/training_paths.local.json",
        python_executable="python-under-test",
    )
    changed = copy.deepcopy(commands)
    index = changed[1].index("--specialist-read-mode")
    changed[1][index + 1] = "full"

    with pytest.raises(SpecialistReadCalibrationError, match="read mode differs"):
        validate_prepare_commands(contract, changed)
