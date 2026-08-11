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
    contract["measurement_contract"]["outcome_measurement"] = {
        "candidate_colors": ["W", "B"],
        "common_random_numbers_within_pairs": True,
        "fixed_replay_corpus": (
            "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
        ),
        "fixed_replay_corpus_identity": (
            "ca4b410dd2913933d3ecbd8672fe274ea4a2f8ad42db3f039dabfa52af196aa4"
        ),
        "fixed_replay_corpus_sha256": (
            "9637efaae21074eefb4fab9e22550f5729999b30d03ed469dc88cf75aae07c2f"
        ),
        "games_per_checkpoint_condition_seed": 24,
        "held_out": False,
        "max_post_start_logical_plies": 120,
        "opponent": "common-game-50-anchor",
        "optimizer_updates": 0,
        "sampling_temperature": 0.2,
        "strict_replay_audit": (
            "docs/evidence/"
            "phase-replay-development-corpus-sanmill-audit-2026-08-11.json"
        ),
        "strict_replay_audit_identity": (
            "9d4c54270c6e66dd9e16b4dae5af9291b1fea6d1385856650e71119dc4c0dbbf"
        ),
        "strict_replay_audit_sha256": (
            "4634ba61a4e43c0b6d80a80c882aea5ca985b9bc8923e7895b39bf8ad557e42e"
        ),
        "total_games": 288,
        "training_games": 0,
        "transition_boundaries": [4096, 8192],
        "writes_training_data": False,
    }
    contract["analysis"]["outcome_classification"] = {
        "maximum_opposite_malom_mass_effect": 0.05,
        "maximum_opposite_phase_effect": 0.25,
        "maximum_truncation_rate_increase": 0.1,
        "minimum_aggregate_score_effect": 1.0 / 12.0,
        "minimum_per_seed_score_effect": 1.0 / 24.0,
        "minimum_supporting_seeds": 2,
    }
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
