"""Focused tests for fail-closed policy-auxiliary calibration preparation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.validation.malom_policy_auxiliary_calibration_readiness import (
    DEFAULT_CONTRACT,
    DEFAULT_REPORT,
    MalomPolicyAuxiliaryCalibrationReadinessError,
    _normalised_training_semantics,
    build_prepare_commands,
    inspect_gradient_evidence,
    load_calibration_contract,
)
from learned_ai.validation.mill_bonus_ablation_readiness import (
    assert_preparation_outputs_ignored,
    assert_preparation_targets_absent,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / DEFAULT_CONTRACT
PATHS_CONFIG = ROOT / "data/training_paths.local.json"


def test_loader_accepts_only_the_frozen_four_arm_contract() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)

    assert contract["plan_identity"] == (
        "4029c9ad0d10b4c9af7ffcca91a6ef9eb1647badf2c3fbf0db80da2f48dfa2f0"
    )
    assert [
        arm["malom_policy_aux_coef"]
        for arm in sorted(contract["arms"], key=lambda arm: arm["launch_order"])
    ] == [0.0, 0.03, 0.1, 0.3]


def test_loader_rejects_semantic_tampering(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["resources"]["maximum_completed_games_total"] = 401
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        MalomPolicyAuxiliaryCalibrationReadinessError,
        match="plan identity differs",
    ):
        load_calibration_contract(changed)


def test_gradient_evidence_binds_the_tracked_and_raw_reports() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    result = inspect_gradient_evidence(
        ROOT,
        contract,
        source={"head": "a36aff4686418823f0047248a586915dae39d28b"},
    )

    assert result["probe_identity"] == (
        "5ea60e2955a7a9b878ec4119648ed91ddcffd94687bb3c0976571e96048daa9c"
    )
    assert result["decision"]["training_launch_authorized"] is False
    assert result["result"]["fresh_seeds"] == [48, 49, 50]
    assert result["result"]["informative_states"] == 29
    assert result["result"]["mutation_checks_passed"] is True


def test_gradient_evidence_rejects_a_changed_raw_identity() -> None:
    contract = copy.deepcopy(load_calibration_contract(CONTRACT_PATH))
    contract["preparation_evidence"]["tracked_manifest"][
        "probe_sha256"
    ] = "0" * 64

    with pytest.raises(
        MalomPolicyAuxiliaryCalibrationReadinessError,
        match="probe reference differs|probe bytes differ",
    ):
        inspect_gradient_evidence(
            ROOT,
            contract,
            source={"head": "a36aff4686418823f0047248a586915dae39d28b"},
        )


def test_prepare_commands_encode_only_the_auxiliary_coefficient() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    ordered = sorted(contract["arms"], key=lambda arm: arm["launch_order"])
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=PATHS_CONFIG,
        python_executable="python-under-test",
    )

    assert len(commands) == 4
    normalised: set[str] = set()
    for arm, command in zip(ordered, commands, strict=True):
        assert command[:3] == [
            "python-under-test",
            str(ROOT / "scripts/manage_generalist_run.py"),
            "prepare",
        ]
        assert "authorize" not in command
        assert "run-next" not in command
        args = manager._build_parser().parse_args(command[2:])
        assert args.plan_id == arm["plan_id"]
        assert args.experiment_id == arm["experiment_id"]
        assert args.seed == 51
        assert args.mill_bonus_mode == "malom-preserving-only"
        assert args.malom_policy_aux_coef == arm["malom_policy_aux_coef"]
        assert args.max_games == 5000
        assert args.completion_game_bound == 100
        assert args.segment_games == 100
        assert args.max_wall_hours == 0.5
        assert args.max_ply == 120
        assert args.policy_health_gate
        common_args = manager._common_trainer_args(args, PATHS_CONFIG)
        parsed = trainer._build_argument_parser().parse_args(
            ["--preflight", "long-run", *common_args]
        )
        assert parsed.malom_policy_aux_coef == arm["malom_policy_aux_coef"]
        assert parsed.referee_engine == "sanmill"
        assert parsed.opponent_engine == "sanmill"
        assert parsed.curriculum_advance_policy == "fixed-resource"
        assert parsed.sanmill_node_ladder == (
            1000,
            5000,
            25000,
            100000,
            500000,
        )
        assert parsed.sanmill_stage_games == (500, 500, 500, 1000, 2500)
        assert parsed.minimal_rollouts
        assert parsed.no_recovery
        assert parsed.no_sentinel
        assert parsed.no_value_net
        assert parsed.no_gap_net
        assert parsed.no_s1a_warmstart
        assert parsed.no_imitation_mix
        assert parsed.no_s1b_refresher
        assert parsed.no_opening_forcing
        assert not parsed.ppo
        normalised.add(_normalised_training_semantics(parsed))
    assert len(normalised) == 1


def test_preparation_refuses_any_existing_target(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_calibration_contract(CONTRACT_PATH))
    for index, arm in enumerate(contract["arms"]):
        arm["control_dir"] = f"out/arm-{index}"
        arm["specialist_db"] = f"data/arm-{index}.sqlite"
    existing = tmp_path / contract["arms"][1]["specialist_db"]
    existing.parent.mkdir(parents=True)
    existing.touch()

    with pytest.raises(
        MalomPolicyAuxiliaryCalibrationReadinessError,
        match="preparation targets already exist",
    ):
        assert_preparation_targets_absent(
            tmp_path,
            contract,
            report_path=tmp_path / DEFAULT_REPORT,
        )


def test_real_preparation_outputs_are_all_git_ignored() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)

    assert_preparation_outputs_ignored(
        ROOT,
        contract,
        report_path=ROOT / DEFAULT_REPORT,
    )
