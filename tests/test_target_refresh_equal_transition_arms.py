"""Tests for deferred equal-transition arm preparation."""

from __future__ import annotations

import json
from pathlib import Path

from learned_ai.training.generalist_preflight import (
    resolved_resume_config,
    resume_config_sha256,
)
from learned_ai.validation.target_refresh_equal_transition_arms import (
    build_seed_arm_prepare_commands,
    prospective_arm_trainer_args,
)
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (
    SCHEDULE_ISOLATION_CONTRACT_SCHEMA,
    load_equal_transition_contract,
)
from scripts import manage_generalist_run as manager
from scripts import prepare_target_refresh_equal_transition_diagnostic as prepare
from learned_ai.training.run_contract import canonical_sha256


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
            if field in record:
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


def test_seed_arm_commands_freeze_exact_resume_without_launch(
    tmp_path: Path,
) -> None:
    contract = load_equal_transition_contract(CONTRACT)
    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text(
        json.dumps({"sanmill_training_checkout": str(tmp_path)}),
        encoding="utf-8",
    )

    commands = build_seed_arm_prepare_commands(
        root=ROOT,
        contract=contract,
        seed=64,
        paths_config=paths_config,
        python_executable="python",
    )

    assert len(commands) == 2
    for condition, command in zip(
        ("refresh-once", "no-refresh"), commands, strict=True
    ):
        assert "authorize" not in command
        assert "run-next" not in command
        assert "run-authorized" not in command
        parsed = manager._build_parser().parse_args(command[2:])
        assert parsed.target_refresh_fork_treatment == condition
        assert parsed.initial_resume_completed_games == 50
        assert parsed.no_exact_resume is True
        assert parsed.completion_game_bound == 600
        assert parsed.segment_games == 550
        assert parsed.post_fork_transition_bound == 8192
        assert parsed.exact_transition_batches is True


def test_seed_arm_commands_differ_only_in_isolated_resume_identities(
    tmp_path: Path,
) -> None:
    contract = load_equal_transition_contract(CONTRACT)
    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text(
        json.dumps({"sanmill_training_checkout": str(tmp_path)}),
        encoding="utf-8",
    )
    commands = build_seed_arm_prepare_commands(
        root=ROOT,
        contract=contract,
        seed=64,
        paths_config=paths_config,
        python_executable="python",
    )

    args = [
        prospective_arm_trainer_args(command, paths_config=paths_config)
        for command in commands
    ]
    assert args[0].target_refresh_fork_treatment == "refresh-once"
    assert args[1].target_refresh_fork_treatment == "no-refresh"
    assert resume_config_sha256(args[0]) != resume_config_sha256(args[1])
    normalised = []
    for item in args:
        semantics = resolved_resume_config(item)
        semantics["specialist_db"] = "<same-seed-byte-identical-clone>"
        normalised.append(semantics)
    assert normalised[0] == normalised[1]


def test_arm_prepare_cli_requires_separate_report(capsys) -> None:
    assert prepare.main(["--prepare-seed-arms", "64"]) == 1
    assert "requires --arm-report" in capsys.readouterr().err


def test_schedule_isolation_arm_commands_bind_transition_temperature(
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

    commands = build_seed_arm_prepare_commands(
        root=ROOT,
        contract=contract,
        seed=67,
        paths_config=paths_config,
        python_executable="python",
    )

    assert len(commands) == 2
    for condition, command in zip(
        ("refresh-once", "no-refresh"), commands, strict=True
    ):
        parsed = manager._build_parser().parse_args(command[2:])
        assert parsed.target_refresh_fork_treatment == condition
        assert parsed.sanmill_node_ladder == "1000"
        assert parsed.sanmill_stage_games == "5000"
        assert parsed.temperature_schedule_axis == "post-fork-transitions"
        assert parsed.post_fork_temperature_anneal_transitions == 106304
        args = prospective_arm_trainer_args(
            command,
            paths_config=paths_config,
        )
        assert args.temperature_schedule_axis == "post-fork-transitions"
        assert args.post_fork_temperature_anneal_transitions == 106304
