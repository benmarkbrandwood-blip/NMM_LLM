from __future__ import annotations

import copy
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_mature_fork_diagnostic import (
    DEFAULT_CONTRACT,
    DEFAULT_READINESS,
    PLAN_SCHEMA,
    MatureTargetRefreshDiagnosticError,
    TRAINER_TREATMENT,
    _preflight_experiment_digest_matches,
    build_arm_prepare_command,
    load_contract,
    validate_contract,
)


SHA = "a" * 64


def test_attempt_002_uses_fresh_default_paths() -> None:
    assert DEFAULT_CONTRACT.name.endswith("attempt-002.json")
    assert "attempt-002" in DEFAULT_READINESS.as_posix()


def _contract() -> dict:
    sources = []
    arms = []
    order = 0
    for seed in (67, 68, 69):
        sources.append(
            {
                "seed": seed,
                "checkpoint_path": f"out/source/s{seed}.pt",
                "checkpoint_file_sha256": SHA,
                "checkpoint_id": f"checkpoint-{seed}",
                "checkpoint_payload_sha256": SHA,
                "checkpoint_config_sha256": SHA,
                "checkpoint_experiment_id": f"source-{seed}",
                "checkpoint_run_id": f"source-run-{seed}",
                "game_count": 300 + seed,
                "update_count": 100,
                "optimizer_consumed_transitions": 9_000,
                "prior_post_fork_origin": 808,
                "pending_transition_count": 10,
                "specialist_db_path": f"data/source-s{seed}.sqlite",
                "specialist_db_sha256": SHA,
                "specialist_db_bytes": 45_056,
                "common_fork_path": f"out/mature/s{seed}/fork.pt",
            }
        )
        for condition in ("refresh-mature", "stale-control"):
            order += 1
            arms.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "launch_order": order,
                    "arm_id": f"seed{seed}-{condition}",
                    "experiment_id": f"mature-s{seed}",
                    "plan_id": f"mature-s{seed}-{condition}",
                    "control_dir": f"out/mature/s{seed}-{condition}",
                    "specialist_db": f"data/mature-s{seed}-{condition}.sqlite",
                }
            )
    body = {
        "schema_version": PLAN_SCHEMA,
        "status": "designed_unlaunched_needs_publication",
        "objective": "test a mature target refresh",
        "hypothesis": "a mature refresh helps",
        "lineage": {"required_implementation_commit": "b" * 40},
        "source_evidence": {},
        "sources": sources,
        "common_training_contract": {
            "algorithm": "A2C",
            "exact_transition_batch_size": 64,
            "post_mature_fork_transitions_per_arm": 8_192,
            "temperature_schedule_axis": "post-fork-transitions",
            "temperature_origin": 0.8379808850090307,
            "post_fork_temperature_anneal_transitions": 98_112,
            "sanmill_node_budget": 1_000,
            "max_games_schedule": 5_000,
            "max_logical_plies": 120,
            "specialist_read_mode": "theoretical-only",
            "target_refresh_after_fork": "none",
            "frozen_target_ratio": 0.6,
        },
        "arms": arms,
        "measurement_contract": {},
        "resources": {
            "maximum_training_games_total": 3_600,
            "maximum_training_games_per_arm": 600,
            "maximum_active_wall_hours_total": 4.0,
            "maximum_active_wall_hours_per_arm": 0.6,
            "maximum_no_update_games_total": 288,
        },
        "claim_boundary": "development mechanism evidence only",
        "authorization": {"launch_authorized": False},
        "stop_rules": {},
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def test_contract_is_closed_and_requires_six_ordered_arms() -> None:
    contract = _contract()
    assert validate_contract(contract)["plan_identity"] == contract["plan_identity"]

    tampered = copy.deepcopy(contract)
    tampered["arms"].reverse()
    tampered["plan_identity"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "plan_identity"}
    )
    with pytest.raises(MatureTargetRefreshDiagnosticError, match="arm order"):
        validate_contract(tampered)


def test_command_isolates_only_target_treatment_and_paths(tmp_path: Path) -> None:
    contract = _contract()
    source = contract["sources"][0]
    refresh, control = contract["arms"][:2]
    commands = []
    for arm in (refresh, control):
        commands.append(
            build_arm_prepare_command(
                root=tmp_path,
                contract=contract,
                source=source,
                arm=arm,
                branch_checkpoint=tmp_path / arm["arm_id"] / "fork.pt",
                paths_config=tmp_path / "paths.json",
                python_executable="python",
            )
        )
    assert (
        commands[0][commands[0].index("--target-refresh-fork-treatment") + 1]
        == (TRAINER_TREATMENT["refresh-mature"])
    )
    assert (
        commands[1][commands[1].index("--target-refresh-fork-treatment") + 1]
        == (TRAINER_TREATMENT["stale-control"])
    )
    for command in commands:
        assert command[command.index("--post-fork-transition-bound") + 1] == "8192"
        assert command[command.index("--post-fork-temperature-origin") + 1] == (
            "0.8379808850090307"
        )
        assert command[command.index("--sanmill-node-ladder") + 1] == "1000"
        assert command[command.index("--completion-game-bound") + 1] == "967"
        assert command.count("--policy-health-gate") == 1
        assert command[command.index("--policy-health-device") + 1] == "auto"


def test_contract_rejects_resource_or_temperature_drift() -> None:
    for mutate, message in (
        (
            lambda value: value["resources"].__setitem__(
                "maximum_training_games_total", 3_601
            ),
            "resource envelope",
        ),
        (
            lambda value: value["common_training_contract"].__setitem__(
                "temperature_origin", 0.1
            ),
            "temperature origin",
        ),
    ):
        contract = _contract()
        mutate(contract)
        contract["plan_identity"] = canonical_sha256(
            {key: value for key, value in contract.items() if key != "plan_identity"}
        )
        with pytest.raises(MatureTargetRefreshDiagnosticError, match=message):
            validate_contract(contract)


def test_preflight_digest_comparison_does_not_double_prefix() -> None:
    expected = "sha256:" + SHA

    assert _preflight_experiment_digest_matches(
        {"experimentDigest": expected}, expected
    )
    assert not _preflight_experiment_digest_matches(
        {"experimentDigest": "sha256:" + expected}, expected
    )


def test_repository_mature_fork_contract_is_canonical_and_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(
        root / "docs/experiments/sanmill-target-refresh-mature-fork-diagnostic-v1.json"
    )

    assert contract["plan_identity"] == (
        "7a0bd214c353d67bf52d3fb5c8d8c2184f4e6c647d49910a117539415cb2c0c0"
    )
    assert [source["game_count"] for source in contract["sources"]] == [
        439,
        327,
        518,
    ]
