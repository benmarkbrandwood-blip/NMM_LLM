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
    load_equal_transition_contract,
)
from scripts import manage_generalist_run as manager
from scripts import prepare_target_refresh_equal_transition_diagnostic as prepare


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-equal-transition-diagnostic-v1.json"
)


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
