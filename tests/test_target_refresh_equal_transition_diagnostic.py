"""Tests for staged equal-transition diagnostic preparation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.validation import target_refresh_equal_transition_diagnostic as diagnostic
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
    contract["lineage"]["main_review"] = {
        "cherry_picks_selected": [],
        "evidence": {
            "path": "docs/evidence/test-main-review.md",
            "sha256": "1" * 64,
        },
        "independent_dev_changes": [],
        "reason": "synthetic schedule-isolation test contract",
        "reviewed_tip": "2" * 40,
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


def test_schedule_isolation_requires_bound_main_review_evidence(
    tmp_path: Path,
) -> None:
    path = _write_schedule_isolation_contract(tmp_path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    del contract["lineage"]["main_review"]["evidence"]
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    contract["plan_identity"] = canonical_sha256(body)
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(
        TargetRefreshEqualTransitionError,
        match="main review evidence",
    ):
        load_equal_transition_contract(path)


def test_source_audit_requires_independent_dev_change_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    head = "a" * 40
    origin_main = "b" * 40
    required_dev_commit = "c" * 40
    reviewed_main_commit = "d" * 40
    outputs = {
        ("branch", "--show-current"): "dev",
        ("rev-parse", "HEAD"): head,
        ("rev-parse", "origin/dev"): head,
        ("rev-parse", "origin/main"): origin_main,
        ("rev-list", "--count", f"{origin_main}..{origin_main}"): "0",
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ): "",
    }
    monkeypatch.setattr(
        diagnostic,
        "_git_output",
        lambda _root, *arguments: outputs[arguments],
    )

    def fake_run(arguments, **_kwargs):
        if arguments[-2:] == [required_dev_commit, head]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    contract = {
        "lineage": {
            "main_review": {
                "reviewed_tip": origin_main,
                "independent_dev_changes": [
                    {
                        "dev_commit": required_dev_commit,
                        "main_commit": reviewed_main_commit,
                        "reason": "test",
                    }
                ],
            },
            "required_implementation_commits": [],
        }
    }

    with pytest.raises(
        TargetRefreshEqualTransitionError,
        match="independent dev change is absent",
    ):
        diagnostic._inspect_source(tmp_path, contract)


def test_source_audit_accepts_and_reports_unreviewed_main_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    head = "a" * 40
    reviewed_tip = "b" * 40
    origin_main = "c" * 40
    outputs = {
        ("branch", "--show-current"): "dev",
        ("rev-parse", "HEAD"): head,
        ("rev-parse", "origin/dev"): head,
        ("rev-parse", "origin/main"): origin_main,
        ("rev-list", "--count", f"{reviewed_tip}..{origin_main}"): "3",
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ): "",
    }
    monkeypatch.setattr(
        diagnostic,
        "_git_output",
        lambda _root, *arguments: outputs[arguments],
    )
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    contract = {
        "lineage": {
            "main_review": {
                "reviewed_tip": reviewed_tip,
                "independent_dev_changes": [],
            },
            "required_implementation_commits": [],
        }
    }

    source = diagnostic._inspect_source(tmp_path, contract)

    assert source["origin_main"] == origin_main
    assert source["origin_main_reviewed_tip"] == reviewed_tip
    assert source["origin_main_unreviewed_commits"] == 3


def test_source_audit_rejects_rewritten_reviewed_main_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    head = "a" * 40
    reviewed_tip = "b" * 40
    origin_main = "c" * 40
    outputs = {
        ("branch", "--show-current"): "dev",
        ("rev-parse", "HEAD"): head,
        ("rev-parse", "origin/dev"): head,
        ("rev-parse", "origin/main"): origin_main,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ): "",
    }
    monkeypatch.setattr(
        diagnostic,
        "_git_output",
        lambda _root, *arguments: outputs[arguments],
    )
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    contract = {
        "lineage": {
            "main_review": {
                "reviewed_tip": reviewed_tip,
                "independent_dev_changes": [],
            },
            "required_implementation_commits": [],
        }
    }

    with pytest.raises(
        TargetRefreshEqualTransitionError,
        match="reviewed origin/main tip is no longer an ancestor",
    ):
        diagnostic._inspect_source(tmp_path, contract)
