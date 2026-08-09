"""Focused tests for normalized auxiliary calibration preparation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.validation.malom_policy_auxiliary_normalized_calibration_readiness import (
    DEFAULT_CONTRACT,
    DEFAULT_REPORT,
    MalomPolicyAuxiliaryNormalizedCalibrationReadinessError,
    _normalised_training_semantics,
    build_prepare_commands,
    inspect_batch_capture_evidence,
    inspect_preparation_targets,
    load_normalized_calibration_contract,
    publish_source_readiness,
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


def test_loader_accepts_only_the_frozen_six_arm_contract() -> None:
    contract = load_normalized_calibration_contract(CONTRACT_PATH)

    assert contract["plan_identity"] == (
        "1b6f8d05047c4de9d6603d9ae1f26714cb1a23b3b96749e76136387a5f0b53ab"
    )
    assert [arm["seed"] for arm in contract["arms"]] == [55, 55, 56, 56, 57, 57]
    assert [arm["malom_policy_aux_mode"] for arm in contract["arms"]] == [
        "fixed",
        "policy-head-normalized",
        "fixed",
        "policy-head-normalized",
        "fixed",
        "policy-head-normalized",
    ]


def test_loader_rejects_semantic_tampering(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["resources"]["maximum_completed_games_total"] = 601
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationReadinessError,
        match="plan identity differs",
    ):
        load_normalized_calibration_contract(changed)


def test_batch_capture_evidence_binds_tracked_and_raw_results() -> None:
    contract = load_normalized_calibration_contract(CONTRACT_PATH)

    evidence = inspect_batch_capture_evidence(
        ROOT,
        contract,
        source_head="702c669a624f3ead7099126c6707e6513ed821c3",
    )

    assert evidence["result_identity"] == (
        "b0dfd3415c55196c59e71cf67e45b00ab5844e9f62fbc9f3bdc31b09a694bd86"
    )
    assert evidence["summary"]["fresh_seeds"] == [52, 53, 54]
    assert evidence["summary"]["batches"] == 19
    assert evidence["summary"]["informative_steps"] == 453


def test_batch_capture_evidence_rejects_a_changed_identity() -> None:
    contract = copy.deepcopy(load_normalized_calibration_contract(CONTRACT_PATH))
    contract["preparation_evidence"]["no_update_batch_capture"][
        "result_sha256"
    ] = "0" * 64

    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationReadinessError,
        match="SHA-256 differs",
    ):
        inspect_batch_capture_evidence(
            ROOT,
            contract,
            source_head="702c669a624f3ead7099126c6707e6513ed821c3",
        )


def test_prepare_commands_change_only_mode_within_each_seed() -> None:
    contract = load_normalized_calibration_contract(CONTRACT_PATH)
    commands = build_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=PATHS_CONFIG,
        python_executable="python-under-test",
    )

    assert len(commands) == 6
    pair_semantics: dict[int, set[str]] = {}
    global_semantics: set[str] = set()
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
        assert args.malom_policy_aux_coef == 0.0
        assert args.malom_policy_aux_mode == arm["malom_policy_aux_mode"]
        assert args.malom_policy_aux_target_ratio == 0.25
        assert args.malom_policy_aux_coef_cap == 0.25
        assert args.malom_policy_aux_denominator_floor == 1e-12
        assert args.completion_game_bound == 100
        assert args.max_wall_hours == pytest.approx(1 / 3)
        common_args = manager._common_trainer_args(args, PATHS_CONFIG)
        parsed = trainer._build_argument_parser().parse_args(
            ["--preflight", "long-run", *common_args]
        )
        trainer._configure_paths(parsed)
        trainer.validate_generalist_configuration(parsed)
        assert parsed.referee_engine == "sanmill"
        assert parsed.opponent_engine == "sanmill"
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
        pair_semantics.setdefault(arm["seed"], set()).add(
            _normalised_training_semantics(parsed, ignore_seed=False)
        )
        global_semantics.add(
            _normalised_training_semantics(parsed, ignore_seed=True)
        )

    assert all(len(values) == 1 for values in pair_semantics.values())
    assert len(global_semantics) == 1


def test_preparation_refuses_any_existing_target(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_normalized_calibration_contract(CONTRACT_PATH))
    for index, arm in enumerate(contract["arms"]):
        arm["control_dir"] = f"out/arm-{index}"
        arm["specialist_db"] = f"data/arm-{index}.sqlite"
    existing = tmp_path / contract["arms"][2]["specialist_db"]
    existing.parent.mkdir(parents=True)
    existing.touch()

    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationReadinessError,
        match="preparation targets already exist",
    ):
        assert_preparation_targets_absent(
            tmp_path,
            contract,
            report_path=tmp_path / DEFAULT_REPORT,
        )


def test_real_preparation_outputs_are_all_git_ignored() -> None:
    contract = load_normalized_calibration_contract(CONTRACT_PATH)

    assert_preparation_outputs_ignored(
        ROOT,
        contract,
        report_path=ROOT / DEFAULT_REPORT,
    )


def test_source_readiness_publisher_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "source-readiness.json"
    report = {"state": "implementation_complete_needs_publication"}

    publish_source_readiness(path, report)
    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationReadinessError,
        match="already exists",
    ):
        publish_source_readiness(path, report)


def test_source_audit_reports_existing_preparation_targets(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_normalized_calibration_contract(CONTRACT_PATH))
    for index, arm in enumerate(contract["arms"]):
        arm["control_dir"] = f"out/arm-{index}"
        arm["specialist_db"] = f"data/arm-{index}.sqlite"
    report_path = tmp_path / "out/readiness.json"

    assert inspect_preparation_targets(
        tmp_path, contract, report_path=report_path
    ) == {"absent": True, "existing": []}

    existing_dir = tmp_path / contract["arms"][0]["control_dir"]
    existing_dir.mkdir(parents=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.touch()
    observed = inspect_preparation_targets(
        tmp_path, contract, report_path=report_path
    )

    assert observed["absent"] is False
    assert observed["existing"] == [
        {
            "label": "readiness_report",
            "path": "out/readiness.json",
            "kind": "file",
        },
        {
            "label": "seed55-control:control_dir",
            "path": "out/arm-0",
            "kind": "directory",
        },
    ]
