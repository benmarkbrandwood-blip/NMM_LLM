"""Tests for staged equal-transition diagnostic preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.training.generalist_preflight import resume_config_sha256
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (
    SCHEDULE_ISOLATION_CONTRACT_SCHEMA,
    TargetRefreshEqualTransitionError,
    build_prefix_prepare_commands,
    load_equal_transition_contract,
    validate_prefix_prepare_commands,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-equal-transition-diagnostic-v1.json"
)


def _write_schedule_isolation_contract(tmp_path: Path) -> Path:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    seed_map = {64: 67, 65: 68, 66: 69}
    contract["schema_version"] = SCHEDULE_ISOLATION_CONTRACT_SCHEMA
    contract["pairing"]["seeds"] = [67, 68, 69]
    common = contract["common_training_contract"]
    common["sanmill_node_ladder"] = [1000]
    common["fixed_resource_stage_games"] = [5000]
    common["temperature_schedule_axis"] = "post-fork-transitions"
    common["post_fork_temperature_anneal_transitions"] = 106304
    for record in [*contract["prefixes"], *contract["arms"]]:
        old_seed = int(record["seed"])
        new_seed = seed_map[old_seed]
        record["seed"] = new_seed
        for field in (
            "arm_id",
            "control_dir",
            "experiment_id",
            "plan_id",
            "prefix_specialist_db",
            "resume_checkpoint",
            "specialist_db",
        ):
            if field not in record:
                continue
            record[field] = (
                record[field]
                .replace(f"seed{old_seed}", f"seed{new_seed}")
                .replace(f"s{old_seed}", f"s{new_seed}")
            )
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "schedule-isolation-contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def test_tracked_equal_transition_contract_is_self_consistent() -> None:
    contract = load_equal_transition_contract(CONTRACT)

    assert contract["plan_identity"] == (
        "b14d69db9a33b005c0a19fbb97e7f5b9a16364f1f74390ae85ff3e9d4edabb97"
    )
    assert [prefix["seed"] for prefix in contract["prefixes"]] == [64, 65, 66]
    assert len(contract["arms"]) == 6
    assert contract["authorization"]["launch_authorized"] is False


def test_contract_tampering_fails_closed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["resources"]["maximum_active_wall_hours_total"] = 7.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(TargetRefreshEqualTransitionError, match="plan identity"):
        load_equal_transition_contract(path)


def test_arm_difference_outside_allowlist_fails_even_with_new_identity(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["arms"][1]["resume_checkpoint"] = "different.pt"
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(TargetRefreshEqualTransitionError, match="lineage differs"):
        load_equal_transition_contract(path)


def test_prefix_prepare_commands_are_read_only_and_semantically_paired(
    tmp_path: Path,
) -> None:
    contract = load_equal_transition_contract(CONTRACT)
    commands = build_prefix_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=tmp_path / "training_paths.local.json",
        python_executable="python",
    )

    validate_prefix_prepare_commands(contract, commands)
    assert len(commands) == 3
    for expected_seed, command in zip((64, 65, 66), commands, strict=True):
        assert "authorize" not in command
        assert "run-next" not in command
        assert "run-all" not in command
        parsed = manager._build_parser().parse_args(command[2:])
        assert parsed.seed == expected_seed
        assert parsed.completion_game_bound == 50
        assert parsed.no_exact_resume is True
        assert parsed.exact_transition_batches is True
        assert parsed.target_refresh_fork_treatment == "capture"


def test_prefix_commands_build_valid_frozen_trainer_configuration(
    tmp_path: Path,
) -> None:
    contract = load_equal_transition_contract(CONTRACT)
    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text(
        json.dumps({"sanmill_training_checkout": str(tmp_path)}),
        encoding="utf-8",
    )
    commands = build_prefix_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=paths_config,
        python_executable="python",
    )

    resume_identities: set[str] = set()
    for command in commands:
        parsed = manager._build_parser().parse_args(command[2:])
        common = manager._common_trainer_args(parsed, paths_config)
        args = trainer._build_argument_parser().parse_args(
            ["--preflight", "long-run", *common]
        )
        trainer._configure_paths(args)
        trainer.validate_generalist_configuration(args)
        resume_identities.add(resume_config_sha256(args))
        assert args.post_fork_transition_bound is None
        assert args.start_mode == "fresh"
    assert len(resume_identities) == 3


def test_all_arm_paths_are_deferred_and_distinct() -> None:
    contract = load_equal_transition_contract(CONTRACT)

    paths = [arm["specialist_db"] for arm in contract["arms"]]
    assert len(paths) == len(set(paths)) == 6
    for seed in (64, 65, 66):
        seed_arms = [arm for arm in contract["arms"] if arm["seed"] == seed]
        assert seed_arms[0]["resume_checkpoint"] == seed_arms[1][
            "resume_checkpoint"
        ]
        assert seed_arms[0]["prefix_specialist_db"] == seed_arms[1][
            "prefix_specialist_db"
        ]


def test_contract_requires_exact_frozen_transition_boundaries(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["measurement_contract"]["transition_boundaries"] = [1024, 8192]
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        TargetRefreshEqualTransitionError,
        match="scientific values differ",
    ):
        load_equal_transition_contract(path)


def test_schedule_isolation_contract_freezes_transition_indexed_controls(
    tmp_path: Path,
) -> None:
    contract = load_equal_transition_contract(
        _write_schedule_isolation_contract(tmp_path)
    )

    assert contract["pairing"]["seeds"] == [67, 68, 69]
    assert [prefix["seed"] for prefix in contract["prefixes"]] == [67, 68, 69]
    common = contract["common_training_contract"]
    assert common["sanmill_node_ladder"] == [1000]
    assert common["fixed_resource_stage_games"] == [5000]
    assert common["temperature_schedule_axis"] == "post-fork-transitions"
    assert common["post_fork_temperature_anneal_transitions"] == 106304


def test_schedule_isolation_prefix_keeps_common_global_temperature(
    tmp_path: Path,
) -> None:
    contract = load_equal_transition_contract(
        _write_schedule_isolation_contract(tmp_path)
    )
    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text(
        json.dumps({"sanmill_training_checkout": str(tmp_path)}),
        encoding="utf-8",
    )
    commands = build_prefix_prepare_commands(
        root=ROOT,
        contract=contract,
        paths_config=paths_config,
        python_executable="python",
    )

    validate_prefix_prepare_commands(contract, commands)
    for expected_seed, command in zip((67, 68, 69), commands, strict=True):
        parsed = manager._build_parser().parse_args(command[2:])
        assert parsed.seed == expected_seed
        assert parsed.sanmill_node_ladder == "1000"
        assert parsed.sanmill_stage_games == "5000"
        common_args = manager._common_trainer_args(parsed, paths_config)
        args = trainer._build_argument_parser().parse_args(
            ["--preflight", "long-run", *common_args]
        )
        assert args.temperature_schedule_axis == "global-games"
        assert args.post_fork_temperature_anneal_transitions is None


def test_schedule_isolation_tampering_fails_closed(tmp_path: Path) -> None:
    path = _write_schedule_isolation_contract(tmp_path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["common_training_contract"][
        "post_fork_temperature_anneal_transitions"
    ] = 106303
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        TargetRefreshEqualTransitionError,
        match="schedule-isolation controls differ",
    ):
        load_equal_transition_contract(path)
